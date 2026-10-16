"""Native asynchronous LangChain calls without worker-thread HTTP."""

import asyncio
import os

from ai_gateway.langchain import ChatGateway


async def main():
    async with ChatGateway(
        api_key=os.environ["GATEWAY_API_KEY"],
        base_url=os.getenv("GATEWAY_BASE_URL", "http://136.81.141.176/v1"),
    ) as model:
        results = await model.abatch(["Explain routing.", "Explain gateway budgets."],
                                     config={"max_concurrency": 2})
        for result in results:
            print(result.content, result.response_metadata["_gateway"])


if __name__ == "__main__":
    asyncio.run(main())
