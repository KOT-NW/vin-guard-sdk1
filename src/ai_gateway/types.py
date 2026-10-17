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
    # P3 guardrail surface (design §10): hits served on this request plus
    # the rule-scan latency and flat Jev cost. Parsed from `_gateway`,
    # falling back to `x-gateway-*` headers when the body omits them.
    guardrails: list[dict] = field(default_factory=list)
    guardrail_ms: int | None = None
    guardrail_cost_usd: float = 0.0
