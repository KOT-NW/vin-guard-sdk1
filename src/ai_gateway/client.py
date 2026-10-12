"""Synchronous data-plane client; health probes use the service root."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from collections.abc import Iterator

import httpx

from ._shared import (chat_request, ensure_sse, gateway_headers, health_url,
                      raise_for_status, translate_transport_errors)
from ._sse import iter_text_chunks, parse_sse_lines
from .errors import GatewayError
from .types import GatewayInfo, Usage


@dataclass(frozen=True)
class ModelEntry:
    id: str
    owned_by: str = "gateway"
    type: str = "policy"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatCompletion:
    id: str
    model: str
    content: str
    choices: list[dict[str, Any]]
    usage: Usage
    gateway: GatewayInfo
    headers: dict[str, str]
    raw: dict[str, Any]


def _decode_json(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise GatewayError("Gateway returned an unexpected response body",
                           code="invalid_response", status=response.status_code,
                           request_id=response.headers.get("x-gateway-request-id"),
                           payload=None, headers=response.headers) from exc


def parse_chat_response(response: httpx.Response) -> ChatCompletion:
    payload = _decode_json(response)
    if not isinstance(payload, dict):
        raise GatewayError("Gateway returned an unexpected response body",
                           code="invalid_response", status=response.status_code,
                           request_id=response.headers.get("x-gateway-request-id"),
                           payload=payload, headers=response.headers)
    choices = payload.get("choices") or []
    if not isinstance(choices, list):
        choices = []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    gateway = payload.get("_gateway") if isinstance(payload.get("_gateway"), dict) else {}
    headers = gateway_headers(response)
    try:
        usage_typed = Usage(**{key: int(usage.get(key) or 0) for key in
                           ("prompt_tokens", "completion_tokens", "total_tokens")})
        cost_usd = float(gateway["costUsd"] if gateway.get("costUsd") is not None
                         else headers.get("x-gateway-cost-usd") or 0)
    except (ValueError, TypeError) as exc:
        raise GatewayError("Gateway returned an unexpected response body",
                           code="invalid_response", status=response.status_code,
                           request_id=headers.get("x-gateway-request-id"),
                           payload=payload, headers=response.headers) from exc
    return ChatCompletion(
        id=str(payload.get("id", "")), model=str(payload.get("model", "")),
        content=content if isinstance(content, str) else "", choices=choices,
        usage=usage_typed,
        gateway=GatewayInfo(
            request_id=gateway.get("requestId") or headers.get("x-gateway-request-id", ""),
            tier=gateway.get("tier") or headers.get("x-gateway-route-tier", ""),
            policy=gateway.get("policy"), confidence=gateway.get("confidence"),
            cost_usd=cost_usd,
        ), headers=headers, raw=payload,
    )


def parse_models(response: httpx.Response) -> list[ModelEntry]:
    known = {"id", "owned_by", "type"}
    payload = _decode_json(response)
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [ModelEntry(id=str(item.get("id", "")),
                       owned_by=str(item.get("owned_by", "gateway")),
                       type=str(item.get("type", "policy")),
                       extra={k: v for k, v in item.items() if k not in known})
            for item in data if isinstance(item, dict)]


class GatewayClient:
    """Auth, gateway headers, typed results and explicit resource ownership.

    Chat keyword options follow API_DESIGN §3.1. Tools, structured output,
    top_p, stop, user and seed are forwarded for future compatible backends;
    the current Java DTO only consumes temperature and max_tokens.
    """

    def __init__(self, *, api_key: str,
                 base_url: str = "https://gateway.example.com/v1",
                 timeout: float | httpx.Timeout = 30.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        if not api_key:
            raise ValueError("api_key is required (gw_live_...)")
        self._health_url = health_url(base_url, "healthz")
        self._ready_url = health_url(base_url, "readyz")
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})

    def chat_completions(self, *, model: str, messages: list[dict[str, Any]],
                         **options: Any) -> ChatCompletion:
        body, headers = chat_request(model=model, messages=messages, **options)
        if body["stream"]:
            raise ValueError("Use stream_chat_completions for stream=True")
        with translate_transport_errors():
            response = self._http.post("/chat/completions", json=body, headers=headers)
        raise_for_status(response)
        return parse_chat_response(response)

    def stream_chat_completions(self, *, model: str, messages: list[dict[str, Any]],
                                **options: Any) -> Iterator[dict[str, Any]]:
        body, headers = chat_request(model=model, messages=messages, stream=True, **options)
        with translate_transport_errors():
            with self._http.stream("POST", "/chat/completions", json=body, headers=headers) as response:
                if not response.is_success:
                    response.read()
                    raise_for_status(response)
                ensure_sse(response)
                yield from parse_sse_lines(response.iter_lines())

    def stream_text(self, **kwargs: Any) -> str:
        return "".join(iter_text_chunks(self.stream_chat_completions(**kwargs)))

    def list_models(self) -> list[ModelEntry]:
        with translate_transport_errors():
            response = self._http.get("/models")
        raise_for_status(response)
        return parse_models(response)

    def _probe(self, url: str) -> dict[str, Any]:
        with translate_transport_errors():
            response = self._http.send(httpx.Request("GET", url))
        raise_for_status(response)
        return _decode_json(response)

    def health(self) -> dict[str, Any]:
        return self._probe(self._health_url)

    def ready(self) -> dict[str, Any]:
        return self._probe(self._ready_url)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> GatewayClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
