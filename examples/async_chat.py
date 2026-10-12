"""Native async chat; configure GATEWAY_API_KEY and GATEWAY_BASE_URL."""

import asyncio
import os

from ai_gateway import AsyncGatewayClient


async def main():
    async with AsyncGatewayClient(
        api_key=os.environ["GATEWAY_API_KEY"],
        base_url=os.getenv("GATEWAY_BASE_URL", "http://localhost:8080/v1"),
    ) as client:
        response = await client.chat_completions(
            model="auto", messages=[{"role": "user", "content": "Explain gateway routing."}],
        )
        print(response.content)
        print(response.gateway)


if __name__ == "__main__":
    asyncio.run(main())
