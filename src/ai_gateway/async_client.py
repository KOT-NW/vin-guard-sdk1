"""Native HTTPX async counterpart of GatewayClient."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx

from ._shared import (chat_request, ensure_sse, health_url, raise_for_status,
                      translate_transport_errors_async)
from ._sse import SSEDecoder
from .client import ChatCompletion, ModelEntry, parse_chat_response, parse_models
from .errors import GatewayError


class AsyncGatewayClient:
    def __init__(self, *, api_key: str,
                 base_url: str = "http://136.81.141.176/v1",
                 timeout: float | httpx.Timeout = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not api_key:
            raise ValueError("api_key is required (gw_live_...)")
        self._health_url = health_url(base_url, "healthz")
        self._ready_url = health_url(base_url, "readyz")
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout,
                                       transport=transport,
                                       headers={"Authorization": f"Bearer {api_key}"})

    async def chat_completions(self, *, model: str, messages: list[dict[str, Any]],
                               **options: Any) -> ChatCompletion:
        body, headers = chat_request(model=model, messages=messages, **options)
        if body["stream"]:
            raise ValueError("Use stream_chat_completions for stream=True")
        async with translate_transport_errors_async():
            response = await self._http.post("/chat/completions", json=body, headers=headers)
        raise_for_status(response)
        return parse_chat_response(response)

    async def stream_chat_completions(self, *, model: str, messages: list[dict[str, Any]],
                                      **options: Any) -> AsyncIterator[dict[str, Any]]:
        body, headers = chat_request(model=model, messages=messages, stream=True, **options)
        async with translate_transport_errors_async():
            async with self._http.stream("POST", "/chat/completions", json=body, headers=headers) as response:
                if not response.is_success:
                    await response.aread()
                    raise_for_status(response)
                ensure_sse(response)
                decoder = SSEDecoder()
                async for line in response.aiter_lines():
                    event = decoder.feed(line)
                    if decoder.done:
                        return
                    if event is not None:
                        yield event
                event = decoder.feed("")
                if event is not None:
                    yield event

    async def stream_text(self, **kwargs: Any) -> str:
        parts = []
        async for event in self.stream_chat_completions(**kwargs):
            choices = event.get("choices") or []
            if choices and isinstance(choices[0], dict):
                parts.append((choices[0].get("delta") or {}).get("content") or "")
        return "".join(parts)

    async def list_models(self) -> list[ModelEntry]:
        async with translate_transport_errors_async():
            response = await self._http.get("/models")
        raise_for_status(response)
        return parse_models(response)

    async def _probe(self, url: str) -> dict[str, Any]:
        async with translate_transport_errors_async():
            response = await self._http.send(httpx.Request("GET", url))
        raise_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise GatewayError("Gateway returned an unexpected response body",
                               code="invalid_response", status=response.status_code,
                               request_id=response.headers.get("x-gateway-request-id"),
                               payload=None, headers=response.headers) from exc

    async def health(self) -> dict[str, Any]:
        return await self._probe(self._health_url)

    async def ready(self) -> dict[str, Any]:
        return await self._probe(self._ready_url)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> AsyncGatewayClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()
