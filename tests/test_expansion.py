"""Regression and wire-contract coverage for SDK expansion."""

import asyncio
import json

import httpx
import pytest

from ai_gateway import AsyncGatewayClient, ControlPlaneClient, GatewayClient, GatewayError
from ai_gateway._sse import parse_sse_lines


def test_async_chat_models_health_and_closing():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.url.path in ("/healthz", "/readyz"):
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"status": "ok"})
        assert request.headers["authorization"] == "Bearer gw_live_test"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "auto"}]})
        body = json.loads(request.content)
        assert body["max_tokens"] == 10
        assert request.headers["x-gateway-no-store"] == "true"
        assert json.loads(request.headers["x-gateway-metadata"]) == {"ticket": "xin chào"}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": None, "tool_calls": [{"id": "call_1"}]}}],
            "usage": {"total_tokens": 7}, "_gateway": None,
        }, headers={"x-gateway-request-id": "req_1"})

    async def run():
        client = AsyncGatewayClient(api_key="gw_live_test", transport=httpx.MockTransport(handler))
        async with client:
            result = await client.chat_completions(model="auto", messages=[], max_tokens=10,
                                                   no_store=True, metadata={"ticket": "xin chào"})
            assert result.content == ""
            assert result.choices[0]["message"]["tool_calls"][0]["id"] == "call_1"
            assert result.gateway.request_id == "req_1"
            assert result.usage.total_tokens == 7
            assert (await client.list_models())[0].id == "auto"
            await client.health()
            await client.ready()
        assert client._http.is_closed

    asyncio.run(run())
    assert seen == ["/v1/chat/completions", "/v1/models", "/healthz", "/readyz"]


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_errors_are_read_before_parsing(asynchronous):
    def handler(request):
        return httpx.Response(429, json={"error": {"message": "wait", "code": "rate_limit_exceeded"}},
                              headers={"retry-after": "4", "x-gateway-request-id": "req_wait"})

    async def run():
        async with AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
            async for _ in client.stream_chat_completions(model="auto", messages=[]):
                pass

    with pytest.raises(GatewayError) as exc:
        if asynchronous:
            asyncio.run(run())
        else:
            with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
                list(client.stream_chat_completions(model="auto", messages=[]))
    assert exc.value.retry_after == "4"
    assert exc.value.request_id == "req_wait"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_sse_and_non_streaming_backend(asynchronous):
    reply = httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n',
                           headers={"content-type": "text/event-stream; charset=utf-8"})

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return reply

    async def run():
        async with AsyncGatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
            return await client.stream_text(model="auto", messages=[])

    if asynchronous:
        assert asyncio.run(run()) == "hello"
    else:
        with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
            assert client.stream_text(model="auto", messages=[]) == "hello"
    reply = httpx.Response(200, json={"choices": []})
    with pytest.raises(GatewayError, match="non-SSE"):
        if asynchronous:
            asyncio.run(run())
        else:
            with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
                client.stream_text(model="auto", messages=[])


def test_multiline_and_invalid_sse_are_not_silently_discarded():
    assert list(parse_sse_lines([': ping', 'event: message', 'data: {"choices":',
                                 'data: []}', '', 'data: [DONE]', ''])) == [{"choices": []}]
    with pytest.raises(GatewayError, match="Invalid JSON"):
        list(parse_sse_lines(['data: broken', '']))
    with pytest.raises(GatewayError) as exc:
        list(parse_sse_lines(['data: {"error":{"message":"blocked","code":"guardrail_blocked"}}', '']))
    assert exc.value.code == "guardrail_blocked"


def test_metadata_limit_and_unknown_options_fail_before_network():
    def handler(request):
        pytest.fail("Invalid request reached the network")

    with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="4KB"):
            client.chat_completions(model="auto", messages=[], metadata={"text": "x" * 4096})
        with pytest.raises(TypeError):
            client.chat_completions(model="auto", messages=[], temparature=0)


def test_advanced_options_are_forwarded_and_never_mix_with_gateway_headers():
    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["type"] == "function"
        assert body["tool_choice"] == "auto"
        assert body["response_format"] == {"type": "json_object"}
        assert body["top_p"] == 0.9 and body["seed"] == 1
        assert body["stop"] == ["END"] and body["user"] == "user_a"
        assert "metadata" not in body
        assert json.loads(request.headers["x-gateway-metadata"]) == {"ticket": "t_1"}
        return httpx.Response(200, json={"choices": []})

    with GatewayClient(api_key="x", transport=httpx.MockTransport(handler)) as client:
        client.chat_completions(model="auto", messages=[],
                                tools=[{"type": "function", "function": {"name": "lookup"}}],
                                tool_choice="auto", response_format={"type": "json_object"},
                                top_p=0.9, seed=1, stop=["END"], user="user_a", metadata={"ticket": "t_1"})


def test_health_preserves_proxy_mount_prefix_and_ready_503_details():
    def handler(request):
        assert request.url.path == "/gateway/readyz"
        assert "authorization" not in request.headers
        return httpx.Response(503, json={"status": "not-ready"})

    with GatewayClient(api_key="x", base_url="https://example.test/gateway/v1/",
                        transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError) as exc:
            client.ready()
    assert exc.value.status == 503


def test_control_plane_pagination_refreshes_tokens_and_preserves_nulls():
    calls = []
    tokens = iter(["token-1", "token-2"])

    def handler(request):
        calls.append(request)
        assert request.url.path == "/api/v1/orgs/org_a/requests"
        if len(calls) == 1:
            assert request.headers["authorization"] == "Bearer token-1"
            return httpx.Response(200, json={"data": [{"projectId": None}],
                                            "pagination": {"nextCursor": "opaque+/=", "hasMore": True}})
        assert request.headers["authorization"] == "Bearer token-2"
        assert request.url.params["cursor"] == "opaque+/="
        assert request.url.params["projectId"] == "prj_a"
        return httpx.Response(200, json={"data": [{"requestId": "req_b"}],
                                        "pagination": {"nextCursor": None, "hasMore": False}})

    with ControlPlaneClient(base_url="https://cp.example/api/v1", access_token=lambda: next(tokens),
                            transport=httpx.MockTransport(handler)) as client:
        items = list(client.iter_items(client.list_requests, "org_a", limit=20, projectId="prj_a"))
    assert items == [{"projectId": None}, {"requestId": "req_b"}]


@pytest.mark.parametrize("cursor", [None, "repeat"])
def test_broken_pagination_fails_instead_of_truncating_or_looping(cursor):
    def handler(request):
        return httpx.Response(200, json={"data": [], "pagination": {"nextCursor": cursor, "hasMore": True}})

    with ControlPlaneClient(base_url="https://cp.example/api/v1", dev_subject="dev",
                            transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError):
            list(client.iter_items(client.list_orgs))


def test_control_plane_problem_details_and_usage_parameters():
    def handler(request):
        assert request.url.path == "/api/v1/orgs/org_a/usage"
        assert request.headers["x-dev-sub"] == "dev"
        assert "authorization" not in request.headers
        assert request.url.params["from"] == "2026-10-01T00:00:00Z"
        assert request.url.params["groupBy"] == "project"
        assert "granularity" not in request.url.params
        return httpx.Response(403, json={"title": "Forbidden", "detail": "No access",
                                        "code": "forbidden", "traceId": "req_cp"})

    with ControlPlaneClient(base_url="https://cp.example/api/v1", dev_subject="dev",
                            transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError, match="No access") as exc:
            client.get_usage("org_a", from_="2026-10-01T00:00:00Z", group_by="project")
    assert exc.value.status == 403
    assert exc.value.request_id == "req_cp"


@pytest.mark.parametrize("method,args,path", [
    ("me", (), "/me"), ("list_orgs", (), "/orgs"),
    ("list_projects", ("org_a",), "/orgs/org_a/projects"),
    ("list_api_keys", ("prj_a",), "/projects/prj_a/api-keys"),
    ("list_providers", ("org_a",), "/orgs/org_a/providers"),
    ("list_models", ("org_a",), "/orgs/org_a/models"),
    ("list_routing_policies", ("org_a",), "/orgs/org_a/routing-policies"),
    ("list_policy_versions", ("pol_a",), "/routing-policies/pol_a/versions"),
    ("list_requests", ("org_a",), "/orgs/org_a/requests"),
    ("get_request", ("req_a",), "/requests/req_a"),
    ("get_routing_insights", ("org_a",), "/orgs/org_a/routing/insights"),
    ("list_audit_logs", ("org_a",), "/orgs/org_a/audit-logs"),
])
def test_control_plane_endpoint_inventory(method, args, path):
    def handler(request):
        assert request.url.path == "/api/v1" + path
        assert request.method == "GET"
        return httpx.Response(200, json={"data": [], "pagination": {"nextCursor": None, "hasMore": False}})

    with ControlPlaneClient(base_url="https://cp.example/api/v1", access_token="jwt",
                            transport=httpx.MockTransport(handler)) as client:
        getattr(client, method)(*args)
