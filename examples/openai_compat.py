"""Level 0: any OpenAI SDK works by changing base_url + key (design §4.9)."""

import os

from openai import OpenAI


def main() -> None:
    client = OpenAI(
        api_key=os.environ.get("GW_API_KEY", "gw_live_demo"),
        base_url=os.environ.get("GW_BASE_URL", "http://136.81.141.176/v1"),
    )
    resp = client.chat.completions.create(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
    )
    print(resp.choices[0].message.content)


if __name__ == "__main__":
    main()
