"""OpenAI SSE decoding with multiline events and visible protocol errors."""

import json
from collections.abc import Iterable, Iterator
from typing import Any

from .errors import GatewayError


class SSEDecoder:
    def __init__(self) -> None:
        self._data: list[str] = []
        self.done = False

    def feed(self, line: str) -> dict[str, Any] | None:
        if self.done:
            return None
        if line.strip():
            if line.startswith("data:"):
                self._data.append(line[5:].removeprefix(" "))
            return None
        if not self._data:
            return None
        data = "\n".join(self._data)
        self._data.clear()
        if data.strip() == "[DONE]":
            self.done = True
            return None
        try:
            event = json.loads(data)
        except ValueError as exc:
            raise GatewayError("Invalid JSON in gateway SSE event", code="invalid_stream") from exc
        if not isinstance(event, dict):
            raise GatewayError("Expected an object in gateway SSE event", code="invalid_stream")
        if "error" in event:
            raise GatewayError.from_response(200, event, {})
        return event


def parse_sse_lines(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    decoder = SSEDecoder()
    for line in lines:
        event = decoder.feed(line)
        if decoder.done:
            return
        if event is not None:
            yield event
    event = decoder.feed("")
    if event is not None:
        yield event


def iter_text_chunks(events: Iterable[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        choices = event.get("choices") or []
        if choices and isinstance(choices[0], dict):
            content = (choices[0].get("delta") or {}).get("content")
            if content:
                yield content
