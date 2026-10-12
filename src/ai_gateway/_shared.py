"""Wire encoding shared by the sync and async clients."""

import json
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import httpx

from .errors import GatewayError, snippet, transport_error_from

GATEWAY_HEADERS = (
    "x-gateway-request-id", "x-gateway-provider", "x-gateway-model",
    "x-gateway-route-tier", "x-gateway-route-reason", "x-gateway-cost-usd",
    "x-gateway-latency-ms", "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens", "retry-after",
)


def gateway_headers(response: httpx.Response) -> dict[str, str]:
    return {key: response.headers[key] for key in GATEWAY_HEADERS if key in response.headers}


def _error_payload(response: httpx.Response) -> Any:
    """Parse an error body without ever masking the HTTP status.

    Non-JSON, empty or HTML bodies become a bounded plain-text message; the
    original HTTP status is always carried by :meth:`GatewayError.from_response`.
    """
    try:
        return response.json()
    except ValueError:
        text = response.text
        if text and text.strip():
            return {"error": {"message": snippet(text)}}
        return None


def raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    payload = _error_payload(response)
    raise GatewayError.from_response(response.status_code, payload, response.headers)


@contextmanager
def translate_transport_errors():
    """Map HTTPX transport failures to SDK errors, preserving cancellation.

    Only ``httpx.TransportError`` is caught, so ``KeyboardInterrupt``,
    ``asyncio.CancelledError`` and input ``ValueError``/``TypeError`` pass
    through untouched.
    """
    try:
        yield
    except httpx.TransportError as exc:
        raise transport_error_from(exc) from exc


@asynccontextmanager
async def translate_transport_errors_async():
    try:
        yield
    except httpx.TransportError as exc:
        raise transport_error_from(exc) from exc


def gateway_extras(*, tags=None, metadata=None, route_policy=None,
                   no_store=False, idempotency_key=None) -> dict[str, str]:
    extra = {}
    encoded = tags if isinstance(tags, str) else ",".join(f"{k}={v}" for k, v in (tags or {}).items())
    if encoded:
        extra["X-Gateway-Tags"] = encoded
    if metadata is not None:
        value = json.dumps(metadata, separators=(",", ":"), allow_nan=False)
        if len(value.encode("utf-8")) > 4096:
            raise ValueError("metadata must fit in the 4KB gateway header")
        extra["X-Gateway-Metadata"] = value
    if route_policy:
        extra["X-Gateway-Route-Policy"] = route_policy
    if no_store:
        extra["X-Gateway-No-Store"] = "true"
    if idempotency_key:
        extra["Idempotency-Key"] = idempotency_key
    return extra


def chat_request(*, model: str, messages: list[dict[str, Any]], stream=False,
                 temperature=None, max_tokens=None, top_p=None, stop=None,
                 tools=None, tool_choice=None, response_format=None, user=None,
                 seed=None, tags=None, metadata=None, route_policy=None,
                 no_store=False, idempotency_key=None):
    """Accept the documented chat subset; reject misspelled options."""
    body = {"model": model, "messages": messages, "stream": stream}
    options = dict(temperature=temperature, max_tokens=max_tokens, top_p=top_p,
                   stop=stop, tools=tools, tool_choice=tool_choice,
                   response_format=response_format, user=user, seed=seed)
    body.update({key: value for key, value in options.items() if value is not None})
    return body, gateway_extras(tags=tags, metadata=metadata, route_policy=route_policy,
                               no_store=no_store, idempotency_key=idempotency_key)


def health_url(base_url: str, endpoint: str) -> str:
    url = httpx.URL(base_url)
    if url.scheme not in ("http", "https") or not url.host:
        raise ValueError("base_url must be an absolute HTTP(S) URL")
    path = url.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return str(url.copy_with(path=f"{path}/{endpoint}", query=None, fragment=None))


def ensure_sse(response: httpx.Response) -> None:
    if response.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "text/event-stream":
        raise GatewayError("Gateway returned a non-SSE response; this backend does not stream yet",
                           code="streaming_not_supported", status=response.status_code,
                           request_id=response.headers.get("x-gateway-request-id"))
