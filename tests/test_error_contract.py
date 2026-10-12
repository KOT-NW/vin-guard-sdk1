"""Error contract tests: envelopes, transport failures, cleanup and no retry.

These are unit/contract tests over a fake transport (``httpx.MockTransport``);
they do not exercise a live gateway.
"""

import asyncio

import httpx
import pytest

from ai_gateway import (AsyncGatewayClient, ControlPlaneClient, GatewayClient,
                        GatewayConnectionError, GatewayError, GatewayTimeoutError,
                        GatewayTransportError)


def _client(handler, **kwargs):
    return GatewayClient(api_key="gw_live_k", base_url="https://gw.example/v1",
                         transport=httpx.MockTransport(handler), **kwargs)


# --- envelope shapes -------------------------------------------------------

def test_openai_envelope_keeps_diagnostic_payload_and_headers():
    payload = {"error": {"message": "bad", "type": "invalid_request_error",
                         "code": "invalid_request", "param": "messages",
                         "extra": {"hint": "check"}}}

    def handler(request):
        return httpx.Response(400, json=payload, headers={
            "X-Gateway-Request-Id": "req_openai", "Retry-After": "3"})

    with _client(handler) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model="auto", messages=[])
    err = exc.value
    assert (err.message, err.code, err.type, err.param, err.status) == (
        "bad", "invalid_request", "invalid_request_error", "messages", 400)
    assert err.request_id == "req_openai"
    assert err.retry_after == "3"
    assert err.payload == payload
    assert err.extensions == {"extra": {"hint": "check"}}
    assert err.header("x-gateway-request-id") == "req_openai"


def test_rfc7807_problem_details_extensions_and_validation_errors():
    payload = {
        "type": "https://errors.example/forbidden", "title": "Forbidden",
        "detail": "No access to tenant", "status": 403, "code": "forbidden",
        "traceId": "req_problem", "errors": [{"field": "orgId", "reason": "not a member"}],
        "instance": "/api/v1/orgs/org_x/usage",
    }

    def handler(request):
        return httpx.Response(403, json=payload,
                              headers={"Content-Type": "application/problem+json"})

    with ControlPlaneClient(base_url="https://cp.example/api/v1", dev_subject="dev",
                            transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc:
            client.get_usage("org_x")
    err = exc.value
    assert err.message == "No access to tenant"
    assert err.status == 403
    assert err.code == "forbidden"
    assert err.type == "https://errors.example/forbidden"
    assert err.request_id == "req_problem"
    assert err.errors == [{"field": "orgId", "reason": "not a member"}]
    assert err.extensions["instance"] == "/api/v1/orgs/org_x/usage"


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_status_is_authoritative_across_codes(status):
    def handler(request):
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with _client(handler) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model="auto", messages=[])
    assert exc.value.status == status


def test_request_id_priority_header_over_payload():
    def handler(request):
        return httpx.Response(500, json={"error": {"message": "x", "traceId": "from-body"}},
                              headers={"X-Request-Id": "from-header"})

    with _client(handler) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model="auto", messages=[])
    assert exc.value.request_id == "from-header"


@pytest.mark.parametrize("body,expected", [
    ({"error": {"message": "x", "traceId": "t1"}}, "t1"),
    ({"error": {"message": "x", "requestId": "r1"}}, "r1"),
    ({"error": {"message": "x", "request_id": "r2"}}, "r2"),
    ({"detail": "x", "trace_id": "t2"}, "t2"),
])
def test_request_id_payload_fallbacks(body, expected):
    with _client(lambda request: httpx.Response(500, json=body)) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model="auto", messages=[])
    assert exc.value.request_id == expected


def test_missing_request_id_is_none_not_a_crash():
    with _client(lambda request: httpx.Response(500, json={"error": {"message": "x"}})) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model="auto", messages=[])
    assert exc.value.request_id is None


# --- malformed bodies ------------------------------------------------------

@pytest.mark.parametrize("response", [
    httpx.Response(502, text="<html><body>Bad Gateway</body></html>",
                   headers={"content-type": "text/html"}),
    httpx.Response(500, text="upstream exploded"),
    httpx.Response(500),
    httpx.Response(500, json=["unexpected", "list"]),
    httpx.Response(400, json={"error": ["list-not-object"]}),
    httpx.Response(400, json={"error": "plain string error"}),
])
def test_malformed_bodies_never_hide_the_http_status(response):
    def handler(request):
        return response

    with _client(handler) as client:
        with pytest.raises(GatewayError) as exc:
            client.list_models()
    err = exc.value
    assert err.status == response.status_code
    assert err.message
    assert "Traceback" not in err.message


def test_html_body_is_bounded_and_kept_in_payload():
    html = "<html><body>" + ("x" * 5000) + "</body></html>"
    with _client(lambda request: httpx.Response(502, text=html,
                                                headers={"content-type": "text/html"})) as client:
        with pytest.raises(GatewayError) as exc:
            client.list_models()
    assert len(exc.value.message) <= 300
    assert exc.value.payload["error"]["message"].startswith("<html>")


# --- transport failures ----------------------------------------------------

def test_connection_error_is_a_transport_error_without_fake_status():
    original = httpx.ConnectError("connection refused")

    def handler(request):
        raise original

    with _client(handler) as client:
        with pytest.raises(GatewayConnectionError) as exc:
            client.health()
    err = exc.value
    assert isinstance(err, GatewayTransportError)
    assert err.status == 0
    assert err.cause is original
    assert err.__cause__ is original


@pytest.mark.parametrize("raised,phase", [
    (httpx.ConnectTimeout("slow connect"), "connect"),
    (httpx.ReadTimeout("slow read"), "read"),
    (httpx.WriteTimeout("slow write"), "write"),
    (httpx.PoolTimeout("no connection"), "pool"),
    (httpx.TimeoutException("unknown phase"), None),
])
def test_timeout_phase(raised, phase):
    def handler(request):
        raise raised

    with _client(handler) as client:
        with pytest.raises(GatewayTimeoutError) as exc:
            client.chat_completions(model="auto", messages=[])
    assert exc.value.phase == phase
    assert exc.value.status == 0
    assert exc.value.__cause__ is raised


def test_async_connection_error_is_translated():
    async def handler(request):
        raise httpx.ConnectError("nope")

    async def run():
        async with AsyncGatewayClient(api_key="x",
                                      transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(GatewayConnectionError):
                await client.list_models()

    asyncio.run(run())


def test_control_plane_transport_error_is_translated():
    def handler(request):
        raise httpx.ReadTimeout("boom")

    with ControlPlaneClient(base_url="https://cp.example/api/v1", access_token="jwt",
                            transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayTimeoutError) as exc:
            client.list_orgs()
    assert exc.value.phase == "read"


def test_input_value_errors_are_not_translated():
    def handler(request):
        pytest.fail("invalid input reached the network")

    with _client(handler) as client:
        with pytest.raises(ValueError, match="4KB"):
            client.chat_completions(model="auto", messages=[], metadata={"x": "y" * 4096})
        with pytest.raises(TypeError):
            client.chat_completions(model="auto", messages=[], temparature=1)


# --- cancellation ----------------------------------------------------------

def test_async_cancellation_is_not_swallowed():
    async def handler(request):
        await asyncio.sleep(30)
        return httpx.Response(200, json={})

    async def run():
        client = AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler))
        task = asyncio.ensure_future(client.chat_completions(model="auto", messages=[]))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.aclose()

    asyncio.run(run())


# --- stream cleanup --------------------------------------------------------

class _SyncStream(httpx.SyncByteStream):
    def __init__(self, lines):
        self._lines = lines
        self.closed = False

    def __iter__(self):
        yield from self._lines

    def close(self):
        self.closed = True


class _AsyncStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        await asyncio.sleep(30)

    async def aclose(self):
        self.closed = True


class _AsyncErrorStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes):
        self._data = data
        self.closed = False

    async def __aiter__(self):
        yield self._data

    async def aclose(self):
        self.closed = True


def test_sync_stream_is_closed_when_consumer_breaks():
    stream = _SyncStream([b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                          b"data: [DONE]\n\n"])

    def handler(request):
        return httpx.Response(200, stream=stream,
                              headers={"content-type": "text/event-stream"})

    with _client(handler) as client:
        gen = client.stream_chat_completions(model="auto", messages=[])
        assert next(gen)["choices"][0]["delta"]["content"] == "a"
        gen.close()
    assert stream.closed


def test_sync_stream_is_closed_on_http_error():
    stream = _SyncStream([b'{"error":{"message":"wait","code":"rate_limit_exceeded"}}'])

    def handler(request):
        return httpx.Response(429, stream=stream,
                              headers={"retry-after": "4", "x-gateway-request-id": "req_wait"})

    with _client(handler) as client:
        with pytest.raises(GatewayError) as exc:
            list(client.stream_chat_completions(model="auto", messages=[]))
    assert exc.value.retry_after == "4"
    assert exc.value.request_id == "req_wait"
    assert stream.closed


def test_async_stream_is_closed_when_consumer_breaks():
    stream = _AsyncStream()

    def handler(request):
        return httpx.Response(200, stream=stream,
                              headers={"content-type": "text/event-stream"})

    async def run():
        client = AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler))
        agen = client.stream_chat_completions(model="auto", messages=[])
        first = await agen.__anext__()
        assert first["choices"][0]["delta"]["content"] == "a"
        await agen.aclose()
        await client.aclose()

    asyncio.run(run())
    assert stream.closed


def test_async_stream_is_closed_on_http_error():
    stream = _AsyncErrorStream(b'{"error":{"message":"wait","code":"rate_limit_exceeded"}}')

    def handler(request):
        return httpx.Response(429, stream=stream,
                              headers={"retry-after": "4", "x-gateway-request-id": "req_wait"})

    async def run():
        client = AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler))
        with pytest.raises(GatewayError) as exc:
            async for _ in client.stream_chat_completions(model="auto", messages=[]):
                pass
        await client.aclose()
        return exc.value

    err = asyncio.run(run())
    assert err.retry_after == "4"
    assert err.request_id == "req_wait"
    assert stream.closed


# --- no automatic retries --------------------------------------------------

@pytest.mark.parametrize("status", [500, 429])
def test_post_chat_is_never_retried(status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"message": "no"}})

    with _client(handler) as client:
        with pytest.raises(GatewayError):
            client.chat_completions(model="auto", messages=[])
    assert len(calls) == 1


def test_async_post_chat_is_never_retried():
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "no"}})

    async def run():
        async with AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(GatewayError):
                await client.chat_completions(model="auto", messages=[])

    asyncio.run(run())
    assert len(calls) == 1
