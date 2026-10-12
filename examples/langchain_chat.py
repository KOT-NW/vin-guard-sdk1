"""Gateway routing inside a LangChain prompt pipeline."""

import os

from ai_gateway.langchain import ChatGateway
from langchain_core.prompts import ChatPromptTemplate


def main():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Explain briefly and clearly."), ("human", "{question}"),
    ])
    with ChatGateway(
        api_key=os.environ["GATEWAY_API_KEY"],
        base_url=os.getenv("GATEWAY_BASE_URL", "http://localhost:8080/v1"),
        model="auto", max_tokens=200, gateway_tags={"team": "sdk-demo"},
    ) as model:
        result = (prompt | model).invoke({"question": "How does an AI Gateway choose a model?"})
        print(result.content)
        print(result.response_metadata["_gateway"])
        print(result.usage_metadata)


if __name__ == "__main__":
    main()
