"""Gateway errors: OpenAI envelopes, RFC 7807 and transport failures.

Backwards-compatible public attributes: ``message``, ``code``, ``type``,
``param``, ``status``, ``request_id`` and ``retry_after``.

Diagnostic attributes are additive and exist so callers can inspect the
server response without rewrapping it into a string:

``payload``
    The parsed response body (OpenAI envelope, RFC 7807 problem or raw
    text). It may contain server-supplied data, including sensitive values.
``headers``
    The response headers exactly as received. Use :meth:`GatewayError.header`
    for case-insensitive lookups.
``errors``
    RFC 7807 / OpenAI ``errors`` list when the server sent one.
``extensions``
    RFC 7807 extension members (or extra OpenAI error fields) that are not
    part of the reserved keys.

The SDK never logs these diagnostic attributes by default. They are kept on
the exception object for the caller, which is responsible for redaction and
for treating the payload as potentially sensitive.

Transport failures are normalised into a small hierarchy under
:class:`GatewayError`:

``GatewayTransportError``
    ``GatewayConnectionError`` (DNS/TCP/TLS/protocol errors)
    ``GatewayTimeoutError`` with a ``phase`` of ``connect``, ``read``,
    ``write`` or ``pool`` when HTTPX identifies the timeout kind.

These errors carry ``status == 0`` because no HTTP response was received;
the original ``httpx`` exception is preserved on ``.cause`` and attached as
the ``__cause__`` via ``raise ... from exc``.
"""

from __future__ import annotations

from typing import Any

import httpx

_RESERVED_PROBLEM = {"type", "title", "status", "detail", "code", "traceId"}
_RESERVED_OPENAI = {"message", "type", "code", "param", "errors"}
_REQUEST_ID_PAYLOAD_KEYS = ("traceId", "trace_id", "requestId", "request_id")


def _as_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    return None


def snippet(text: str, limit: int = 300) -> str:
    """Collapse whitespace and bound an untrusted body for a message."""
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1] + "\u2026"


def _header(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        try:
            value = getter(name)
        except Exception:
            value = None
        if value:
            return str(value)
    try:
        items = headers.items()
    except AttributeError:
        return None
    lowered = name.lower()
    for key, value in items:
        if value and str(key).lower() == lowered:
            return str(value)
    return None


class GatewayError(Exception):
    """Raised for non-2xx gateway responses (API_DESIGN §2.6)."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "gateway_error",
        error_type: str = "gateway_error",
        param=None,
        status: int = 0,
        request_id: str | None = None,
        retry_after: str | None = None,
        payload: Any = None,
        headers: Any = None,
        errors: list[Any] | None = None,
        extensions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.type = error_type
        self.param = param
        self.status = status
        self.request_id = request_id
        self.retry_after = retry_after
        self.payload = payload
        self.headers = headers
        self.errors = errors
        self.extensions = extensions

    @classmethod
    def from_response(cls, status: int, payload: Any, headers: Any = None) -> "GatewayError":
        """Build an error from an OpenAI envelope, RFC 7807 body or anything else.

        The HTTP status is authoritative and always taken from the response.
        A missing or malformed body never hides the status: the message falls
        back to ``Gateway request failed (HTTP <status>)``.
        """
        message: str | None = None
        code: str | None = None
        error_type: str | None = None
        param = None
        errors: list[Any] | None = None
        extensions: dict[str, Any] | None = None

        if isinstance(payload, dict):
            inner = payload.get("error")
            if isinstance(inner, dict):
                message = _as_text(inner.get("message")) or _as_text(inner.get("detail"))
                code = _as_text(inner.get("code"))
                error_type = _as_text(inner.get("type"))
                param = inner.get("param")
                if isinstance(inner.get("errors"), list):
                    errors = inner.get("errors")
                inner_extra = {k: v for k, v in inner.items() if k not in _RESERVED_OPENAI}
                if inner_extra:
                    extensions = dict(inner_extra)
            elif isinstance(inner, str):
                message = _as_text(inner)

            if message is None:
                message = _as_text(payload.get("detail")) or _as_text(payload.get("title"))
            if code is None:
                code = _as_text(payload.get("code"))
            if error_type is None:
                error_type = _as_text(payload.get("type"))
            if errors is None and isinstance(payload.get("errors"), list):
                errors = payload.get("errors")
            top_extra = {k: v for k, v in payload.items()
                         if k not in _RESERVED_PROBLEM and k not in ("error", "errors")}
            if top_extra:
                extensions = {**top_extra, **(extensions or {})} or None
        elif isinstance(payload, str):
            message = _as_text(payload)

        if message is None:
            message = f"Gateway request failed (HTTP {status})"
        if code is None:
            code = "gateway_error"
        if error_type is None:
            error_type = "gateway_error"

        request_id = _header(headers, "x-gateway-request-id") or _header(headers, "x-request-id")
        if request_id is None and isinstance(payload, dict):
            for source in (payload, payload.get("error") if isinstance(payload.get("error"), dict) else {}):
                for key in _REQUEST_ID_PAYLOAD_KEYS:
                    value = _as_text(source.get(key))
                    if value:
                        request_id = value
                        break
                if request_id is not None:
                    break

        return cls(
            message,
            code=code,
            error_type=error_type,
            param=param,
            status=status,
            request_id=request_id,
            retry_after=_header(headers, "retry-after"),
            payload=payload,
            headers=headers,
            errors=errors,
            extensions=extensions,
        )

    def header(self, name: str) -> str | None:
        """Case-insensitive response-header lookup; returns ``None`` if absent."""
        return _header(self.headers, name)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"[{self.status}/{self.code}] {self.message}"


class GatewayTransportError(GatewayError):
    """A request failed before a complete HTTP response was received.

    ``status`` stays ``0`` because the server did not return a status. The
    original HTTPX exception is available as ``.cause`` and ``__cause__``.
    """

    def __init__(self, message: str, *, cause: Exception | None = None,
                 request: httpx.Request | None = None) -> None:
        super().__init__(message, code="transport_error",
                         error_type="transport_error", status=0)
        self.cause = cause
        self.request = request


class GatewayConnectionError(GatewayTransportError):
    """Connection-level HTTPX failure (DNS, TCP, TLS, protocol, proxy)."""

    def __init__(self, message: str, *, cause: Exception | None = None,
                 request: httpx.Request | None = None) -> None:
        super().__init__(message, cause=cause, request=request)
        self.code = "connection_error"
        self.type = "connection_error"


class GatewayTimeoutError(GatewayTransportError):
    """Timeout failure; ``phase`` is connect/read/write/pool when HTTPX knows."""

    def __init__(self, message: str, *, phase: str | None = None,
                 cause: Exception | None = None,
                 request: httpx.Request | None = None) -> None:
        super().__init__(message, cause=cause, request=request)
        self.code = "timeout"
        self.type = "timeout"
        self.phase = phase


_TIMEOUT_PHASES = (
    (httpx.PoolTimeout, "pool"),
    (httpx.ConnectTimeout, "connect"),
    (httpx.ReadTimeout, "read"),
    (httpx.WriteTimeout, "write"),
)


def transport_error_from(exc: httpx.TransportError) -> GatewayError:
    """Translate an ``httpx.TransportError`` into the SDK hierarchy.

    Callers must re-raise the result with ``raise ... from exc`` so the
    original exception is preserved as ``__cause__``.
    """
    request = getattr(exc, "request", None)
    if isinstance(exc, httpx.TimeoutException):
        phase = next((name for kind, name in _TIMEOUT_PHASES if isinstance(exc, kind)), None)
        label = f"Gateway request timed out ({phase})" if phase else "Gateway request timed out"
        return GatewayTimeoutError(label, phase=phase, cause=exc, request=request)
    return GatewayConnectionError(
        f"Gateway connection failed ({type(exc).__name__})", cause=exc, request=request,
    )
