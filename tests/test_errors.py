"""Error envelope: {"error": {message, type, code, param}} (API_DESIGN §2.6)."""

import httpx
import pytest

from ai_gateway import GatewayClient, GatewayError


@pytest.mark.parametrize(("status", "code"), [
    (400, "invalid_request"),
    (401, "invalid_api_key"),
    (403, "model_not_allowed"),
    (429, "rate_limit_exceeded"),
])
def test_gateway_error_carries_code_status_request_id(status, code):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={
            "error": {"message": "nope", "type": "gateway_error",
                      "code": code, "param": None},
        }, headers={"X-Gateway-Request-Id": "req_err"})

    client = GatewayClient(
        api_key="gw_live_k", base_url="https://gateway.example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(GatewayError) as exc_info:
            client.chat_completions(model="openai/other",
                                    messages=[{"role": "user", "content": "x"}])
    finally:
        client.close()

    err = exc_info.value
    assert err.code == code
    assert err.status == status
    assert err.request_id == "req_err"
