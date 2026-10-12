"""Typed views over chat responses (ChatDtos + API_DESIGN §3.1)."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class GatewayInfo:
    request_id: str = ""
    tier: str = ""
    policy: str | None = None
    confidence: float | None = None
    cost_usd: float = 0.0
