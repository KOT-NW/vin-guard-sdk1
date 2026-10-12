"""Models + health contract (mirrors ModelsController + HealthController)."""

import httpx

from ai_gateway import GatewayClient


def test_models_lists_policies_for_key():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={
            "object": "list",
            "data": [
                {"id": "auto", "object": "model",
                 "owned_by": "gateway", "type": "policy"},
                {"id": "policy/default-auto", "object": "model",
                 "owned_by": "gateway", "type": "policy"},
            ],
        })

    client = GatewayClient(
        api_key="gw_live_k", base_url="https://gateway.example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        models = client.list_models()
    finally:
        client.close()

    ids = [m.id for m in models]
    assert "auto" in ids
    assert "policy/default-auto" in ids


def test_health_and_ready():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        assert request.url.path in ("/healthz", "/readyz")
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(200, json={"status": "ready", "snapshotVersion": 7})

    client = GatewayClient(
        api_key="gw_live_k", base_url="https://gateway.example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        assert client.health() == {"status": "ok"}
        assert client.ready()["snapshotVersion"] == 7
    finally:
        client.close()
