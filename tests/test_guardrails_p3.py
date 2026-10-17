"""P3 guardrail SDK surface: safety_mode, _gateway guardrails, control-plane lifecycle."""

import json

import httpx
import pytest

from ai_gateway import GatewayClient, ControlPlaneClient
from ai_gateway.errors import GatewayError


def _stub(handler):
    return httpx.MockTransport(handler)


def test_chat_safety_mode_forwarded_and_guardrails_parsed():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content.decode()))
        return httpx.Response(200, json={
            "id": "chatcmpl-1", "model": "auto", "choices": [],
            "usage": {},
            "_gateway": {
                "requestId": "req_1", "tier": "direct",
                "guardrails": [{"policyId": "gpol_1", "action": "redact", "hits": 1}],
                "guardrailMs": 12, "guardrailCostUsd": 0.001,
            },
        })

    with GatewayClient(api_key="gw_live_k", base_url="https://gw.example/v1",
                       transport=_stub(handler)) as client:
        resp = client.chat_completions(model="auto", messages=[{"role": "user", "content": "hi"}],
                                       safety_mode="buffered")
    assert seen["safety_mode"] == "buffered"
    assert resp.gateway.guardrails == [{"policyId": "gpol_1", "action": "redact", "hits": 1}]
    assert resp.gateway.guardrail_ms == 12
    assert resp.gateway.guardrail_cost_usd == pytest.approx(0.001)


def test_chat_safety_mode_rejects_unknown():
    with GatewayClient(api_key="gw_live_k", base_url="https://gw.example/v1",
                       transport=_stub(lambda request: pytest.fail("no network"))) as client:
        with pytest.raises(ValueError, match="safety_mode"):
            client.chat_completions(model="auto", messages=[], safety_mode="strict")


def test_chat_guardrail_blocked_surfaces_code():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "Blocked by guardrail policy 'gpol_1'",
                                                  "type": "gateway_error", "code": "guardrail_blocked"}})
    with GatewayClient(api_key="gw_live_k", base_url="https://gw.example/v1",
                       transport=_stub(handler)) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model="auto", messages=[{"role": "user", "content": "x"}])
    assert exc.value.code == "guardrail_blocked"
    assert exc.value.status == 400


def test_control_plane_guardrail_lifecycle_paths():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith("/guardrail-policies") and request.method == "POST":
            return httpx.Response(201, json={"id": "gpol_1"})
        if path.endswith("/publish"):
            return httpx.Response(200, json={"id": "gpol_1"})
        if path.endswith("/test"):
            return httpx.Response(200, json={"action": "flag", "hits": 0})
        if path.endswith("/effective"):
            return httpx.Response(200, json={"projectId": "prj_1", "effective": []})
        if path.endswith("/global-guardrails"):
            return httpx.Response(200, json={"orgId": "org_1", "guardrails": []})
        if "/guardrail-policies" in path:
            return httpx.Response(200, json={"data": [], "pagination": {"nextCursor": None, "hasMore": False}})
        if path.endswith("/guardrails"):
            return httpx.Response(200, json={"projectId": "prj_1", "guardrails": []})
        return httpx.Response(404, json={"error": {"message": "no"}})

    with ControlPlaneClient(base_url="https://cp.example/api/v1", dev_subject="dev",
                            transport=_stub(handler)) as client:
        client.list_guardrail_policies("org_1")
        client.create_guardrail_policy("org_1", {"name": "p", "stage": "input", "mode": "SYNC_BLOCKING",
                                                "rules": []}, idempotency_key="k1")
        client.get_guardrail_policy("gpol_1")
        client.list_guardrail_versions("gpol_1")
        client.publish_guardrail_policy("gpol_1", 2, if_match=4)
        client.test_guardrail_policy("gpol_1", "hi", stage="input")
        client.attach_project_guardrails("prj_1", ["gpol_1"])
        client.get_effective_guardrails("prj_1")
        client.set_global_guardrails("org_1", ["gpol_1"])
    paths = [path for _, path in calls]
    assert any(p.endswith("/orgs/org_1/guardrail-policies") for p in paths)
    assert any(p.endswith("/publish") for p in paths)
    assert any(p.endswith("/effective") for p in paths)
    assert any(p.endswith("/global-guardrails") for p in paths)
