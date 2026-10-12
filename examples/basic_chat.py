"""Basic Level 1 usage (needs a live gateway + GW_API_KEY env)."""

import os

from ai_gateway import GatewayClient


def main() -> None:
    api_key = os.environ.get("GW_API_KEY", "gw_live_demo")
    base_url = os.environ.get("GW_BASE_URL", "http://localhost:8080/v1")
    with GatewayClient(api_key=api_key, base_url=base_url) as client:
        resp = client.chat_completions(
            model="auto",
            messages=[{"role": "user", "content": "Summarize this ticket..."}],
            tags={"team": "support", "env": "dev"},
        )
        print(resp.content)
        print("request:", resp.gateway.request_id, "tier:", resp.gateway.tier)
        print("provider:", resp.headers.get("x-gateway-provider"))


if __name__ == "__main__":
    main()
