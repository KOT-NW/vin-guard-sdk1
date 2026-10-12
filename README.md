# AI Gateway — Python SDK (Level 1)

Thin wrapper over the OpenAI-compatible data plane (`docs/API_DESIGN.md §3`,
`docs/ai-gateway-design.md §4.9`).

## Level 0 — use any OpenAI SDK as-is

```python
from openai import OpenAI

client = OpenAI(
    api_key="gw_live_...",
    base_url="https://gateway.example.com/v1",
)
resp = client.chat.completions.create(
    model="auto",  # or policy/<name> | <provider>/<model> | alias
    messages=[{"role": "user", "content": "Summarize this ticket..."}],
)
```

See `examples/openai_compat.py`.

## Level 1 — this package

```python
from ai_gateway import GatewayClient

client = GatewayClient(
    api_key="gw_live_...",
    base_url="https://gateway.example.com/v1",
)
resp = client.chat_completions(
    model="auto",
    messages=[{"role": "user", "content": "hi"}],
    tags={"team": "support", "env": "prod"},
    metadata={"ticket": "t-123"},
)
print(resp.content)
print(resp.gateway.request_id, resp.gateway.tier)
print(resp.headers.get("x-gateway-provider"))
```

Request options map to `API_DESIGN §2.5` headers:

| Arg | Header |
|---|---|
| `tags: dict \| str` | `X-Gateway-Tags: team=support,env=prod` |
| `metadata: dict` | `X-Gateway-Metadata: {...}` (≤4KB, opaque JSON) |
| `route_policy: str` | `X-Gateway-Route-Policy` |
| `no_store: bool` | `X-Gateway-No-Store: true` |
| `idempotency_key: str` | `Idempotency-Key` (non-streaming retries) |

Errors keep the fields `GatewayError` always had (`.message`, `.code`, `.type`,
`.param`, `.status`, `.request_id`, `.retry_after`) and understand both the
OpenAI envelope and RFC 7807 problem details (`detail`/`title`/`type`,
`errors`, extensions). Header lookups are case-insensitive and the HTTP status
is authoritative. Request ID priority is `X-Gateway-Request-Id` →
`X-Request-Id` → `traceId`/`requestId`/`request_id` in the body. The parsed body
stays on `err.payload` (plus `err.errors`/`err.extensions`) so callers can
inspect it; it may contain sensitive server data, so the SDK never logs it.

Failures before a response arrives raise `GatewayTransportError` subclasses:
`GatewayConnectionError` (connection) and `GatewayTimeoutError` (timeout with a
`phase` of `connect`/`read`/`write`/`pool`). Their `status` is `0` and the
original HTTPX exception is on `__cause__`/`.cause`. Migration note: code that
caught `httpx.TransportError`/`httpx.TimeoutException` should catch
`GatewayTransportError` (or the narrower classes) instead.

No automatic retries are performed; the current Java hot path does not enforce
the documented idempotency header. Retrying a chat POST may duplicate billing.

## Install and verify

```bash
python -m pip install -e './sdk-python[dev]'
python -m pytest sdk-python/tests -q
```

The default run skips `tests/integration` (real-stack smoke). CI has separate
`sdk-core` (no LangChain), `sdk-langchain` (extra required), `sdk-package`
(wheel/sdist installed into a clean venv outside the source tree) and
`sdk-smoke` jobs, across Python 3.10/3.12 and HTTPX 0.27.0/0.28.1, with
`langchain-core` 1.0.0/1.6.7. The adapter tests import LangChain directly, so a
missing dependency fails collection instead of skipping green.

## Async data plane

```python
import asyncio
import os
from ai_gateway import AsyncGatewayClient

async def main():
    async with AsyncGatewayClient(
        api_key=os.environ["GATEWAY_API_KEY"],
        base_url=os.getenv("GATEWAY_BASE_URL", "http://localhost:8080/v1"),
    ) as client:
        response = await client.chat_completions(
            model="auto", messages=[{"role": "user", "content": "Hello"}],
        )
        print(response.content, response.gateway.request_id)

asyncio.run(main())
```

`AsyncGatewayClient` mirrors chat, models, health, ready, streaming and text
collection. Use `aclose()` or `async with`; the sync client uses `close()` or `with`.
Both accept an HTTPX timeout and transport for contract testing.

Health probes call `/healthz` and `/readyz` at the service root, without the API
key. A proxy mount prefix is retained (`/gateway/v1` → `/gateway/healthz`).

`temperature` and `max_tokens` work with the current Java DTO. `top_p`, `stop`,
`tools`, `tool_choice`, `response_format`, `user`, and `seed` are documented
pass-through options for future backends: the current backend ignores them.
Gateway headers are forwarded; setting tags/no-store/metadata does not by itself
implement server-side enforcement. Metadata is checked against the 4KB limit.
Tool-only completions with `content: null` expose an empty `.content` string and
preserve tool calls in `.choices` and `.raw`.

Streaming helpers parse SSE including multiline events and surface malformed
events and stream errors. The current backend returns JSON even with
`stream: true`; helpers raise `GatewayError(code="streaming_not_supported")`
instead of silently returning no text. They do not simulate token streaming.

## Dashboard reads (serverless control plane)

This is a separate service and uses an OIDC access token, not a `gw_live_` key.
These endpoints require the serverless control-plane deployment. Both SDK
branches now include its source after synchronization with `origin/main` on
2026-10-07; source presence does not mean the service is already deployed.

```python
import os
from ai_gateway import ControlPlaneClient

with ControlPlaneClient(
    base_url=os.environ["CONTROL_PLANE_BASE_URL"],  # includes /api/v1
    access_token=os.environ["CONTROL_PLANE_ACCESS_TOKEN"],
) as dashboard:
    print(dashboard.me())
    for request in dashboard.iter_items(
        dashboard.list_requests, "org_example", limit=100,
        **{"from": "2026-10-01T00:00:00Z", "projectId": "prj_example"},
    ):
        print(request["requestId"])
    print(dashboard.get_usage("org_example", group_by="project").data)
```

An access-token callable can supply a refreshed token on every request. Local
development explicitly uses `dev_subject="..."` instead of `access_token`.
The client does not authenticate users or refresh tokens itself.

Read helpers: `me`, `list_orgs`, `list_projects`, `list_api_keys`,
`list_providers`, `list_models`, `list_routing_policies`, `list_policy_versions`,
`get_usage`, `list_requests`, `get_request`, `get_routing_insights`,
`list_audit_logs`. List helpers return `Page(data, next_cursor, has_more)`;
`iter_items` traverses all pages and fails on missing/repeated cursors. Query
kwargs use wire names (`projectId`, `minLatencyMs`, `cursor`, `limit`, etc.).
Usage is daily only; hourly granularity is not implemented by the control plane.
Wire dictionaries preserve nullable and additional fields. Key reads return
metadata only; no SDK secret persistence or key mutations are added here.

Research and prioritized endpoint/function suggestions:
[RESEARCH.md](RESEARCH.md).

Vietnamese architecture guide with component flows, methods, compatibility,
examples and troubleshooting: [SDK guide](../docs/SDK_PYTHON_GUIDE.vi.md).
For a documentation-style reading view, open the [offline HTML guide](../docs/sdk-python-guide.html).

## LangChain integration

The `sdk-python` branch contains the core SDK plus an optional native
`ChatGateway` model.

```bash
python -m pip install -e './sdk-python[dev,langchain]'
python -m pytest sdk-python/tests -q
```

```python
import os
from ai_gateway.langchain import ChatGateway
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer briefly."), ("human", "{question}"),
])
with ChatGateway(
    api_key=os.environ["GATEWAY_API_KEY"],
    base_url="http://localhost:8080/v1",
    model="auto", max_tokens=200,
    gateway_tags={"team": "support"},
    gateway_metadata={"ticket": "t-123"},
) as model:
    message = (prompt | model).invoke({"question": "How does routing work?"})
    print(message.content)
    print(message.response_metadata["_gateway"])
    print(message.response_metadata["gateway_headers"])
    print(message.usage_metadata)
```

`ChatGateway` supports `invoke`, native async `ainvoke`, `batch`/`abatch` and
prompt pipelines. `max_tokens` is sent unchanged to the Java DTO. Responses
preserve additional `_gateway` fields, headers, selected model, finish reason,
and LangChain token usage. `stream`/`astream` emit one **buffered full result**
via LangChain's fallback; they do not request server SSE or token streaming.

```python
async with ChatGateway(api_key=os.environ["GATEWAY_API_KEY"],
                       base_url="http://localhost:8080/v1") as model:
    messages = await model.abatch(["Explain routing", "Explain budgets"],
                                  config={"max_concurrency": 2})
```

Use `gateway_tags` and `gateway_metadata` for gateway headers; LangChain's
`config={"tags": [...], "metadata": {...}}` controls tracing only. Per-call
overrides work through `.bind(gateway_tags=..., route_policy=..., no_store=...)`
or invoke kwargs. Do not bind a fixed idempotency key to a reusable model.

Tools, stop sequences, structured output and other advanced chat options fail
clearly by default because the current backend ignores them. For a future
backend that implements them, `allow_advanced_options=True` enables forwarding,
`bind_tools`, tool-history conversion, and inherited `with_structured_output`.
This switch does not add server capabilities or execute tools.

Sync `with`/`close()` closes the owned sync client; async `with`/`aclose()` closes
both owned clients. If you use both sync and async methods on one instance,
finish with `await model.aclose()`. Injected `client`/`async_client` SDK clients
are caller-owned and remain open. API keys and clients are excluded from model
repr/dumps. LangChain caching is disabled by default; enable caching only with a
cache scoped to the same tenant/key and routing policy.

Only `from ai_gateway.langchain import ChatGateway` imports LangChain. Core
SDK users can install without the `langchain` extra. See
`examples/langchain_chat.py` and `examples/langchain_async.py`.

## End-to-end smoke (real stack)

`tests/integration/test_e2e_smoke.py` exercises the real
control plane → publisher → Valkey → Java gateway path with the backend's
`StubAdapter` as the only substituted LLM upstream. It creates an isolated
org/project/key over the control plane, publishes a snapshot, checks
`healthz`/`readyz`/models, chats sync and async (`openai/stub-sdk-smoke`),
reads project/key metadata, verifies tenant isolation, runs the LangChain
adapter, rejects a bad key, then revokes the key, republishes and expects
`401 key_revoked`.

The harness is opt-in: without `AIGW_SMOKE=1` the tests skip. With it set, a
missing config or unreachable service fails loudly. See
[tests/integration/README.md](tests/integration/README.md) for prerequisites
and the exact command. Authentication uses the local dev principal
(`x-dev-sub`/`x-dev-email`), which is not proof of Cognito JWT, and the smoke
does not assert trace/usage while the usage producer is not wired.
