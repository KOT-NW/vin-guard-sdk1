"""Contract tests mirroring backend GatewayWebTest (chat happy path + headers)."""

import json

import httpx

from ai_gateway import GatewayClient


def _stub_transport(handler):
    return httpx.MockTransport(handler)


def test_chat_happy_path_returns_gateway_extension_and_headers():
    body = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1727434800,
        "model": "openai/stub-demo",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "stub-response for stub-demo"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7},
        "_gateway": {"requestId": "req_abc", "tier": "direct",
                     "policy": None, "confidence": None, "costUsd": 0.0},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer gw_live_key_1"
        payload = json.loads(request.content.decode())
        assert payload["model"] == "openai/stub-demo"
        return httpx.Response(
            200,
            json=body,
            headers={
                "X-Gateway-Request-Id": "req_abc",
                "X-Gateway-Provider": "openai",
                "X-Gateway-Model": "openai/stub-demo",
                "X-Gateway-Route-Tier": "direct",
            },
        )

    client = GatewayClient(
        api_key="gw_live_key_1",
        base_url="https://gateway.example.com/v1",
        transport=_stub_transport(handler),
    )
    try:
        resp = client.chat_completions(
            model="openai/stub-demo",
            messages=[{"role": "user", "content": "hi"}],
        )
    finally:
        client.close()

    assert resp.content == "stub-response for stub-demo"
    assert resp.gateway.tier == "direct"
    assert resp.gateway.request_id == "req_abc"
    assert resp.headers["x-gateway-provider"] == "openai"
    assert resp.headers["x-gateway-model"] == "openai/stub-demo"
    assert resp.usage.total_tokens == 7


def test_gateway_headers_forward_tags_metadata_policy():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        return httpx.Response(200, json={
            "id": "chatcmpl-x", "model": "auto", "choices": [],
            "usage": {}, "_gateway": {},
        })

    client = GatewayClient(
        api_key="gw_live_k", base_url="https://gateway.example.com/v1",
        transport=_stub_transport(handler),
    )
    try:
        client.chat_completions(
            model="auto",
            messages=[{"role": "user", "content": "hi"}],
            tags={"team": "support", "env": "prod"},
            metadata={"ticket": "t-123"},
            route_policy="default-auto",
            no_store=True,
            idempotency_key="idem-1",
        )
    finally:
        client.close()

    lowered = {k.lower(): v for k, v in seen.items()}
    assert lowered["x-gateway-tags"] == "team=support,env=prod"
    assert json.loads(lowered["x-gateway-metadata"]) == {"ticket": "t-123"}
    assert lowered["x-gateway-route-policy"] == "default-auto"
    assert lowered["x-gateway-no-store"] == "true"
    assert lowered["idempotency-key"] == "idem-1"
