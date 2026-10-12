# Python SDK expansion and LangChain integration

Research date: 2026-10-05. Base: `feat/sdk-python` at `6f7414b`.
Two deliverables: `sdk-python-expand` and `sdk-python-langchain`.
LangChain builds on the expansion commit so transport/error handling is shared.

Architecture update: 2026-10-07. Both branches now contain `origin/main` at
`dd0129f`, including the serverless control plane and direct Valkey snapshot
reads. The historical base constraints below describe the initial research.
Current flows, scaffold limitations, methods and compatibility are explained in
[the Vietnamese SDK guide](../docs/SDK_PYTHON_GUIDE.vi.md).

## Evidence and constraints

- `docs/API_DESIGN.md` §§2–4 defines chat, model catalog, health, headers,
  control-plane auth, pagination and errors. Backend code takes precedence
  over features described only in the design.
- `backend/.../gateway/web/ChatDtos.java` only reads model/messages/temperature/
  max_tokens/stream. Provider input currently has no tools, stop or structured
  output. The controller returns JSON, not SSE.
- `HealthController` serves `/healthz` and `/readyz` at the root. The previous
  SDK incorrectly requested `/v1/healthz` and `/v1/readyz`.
- The serverless route tables and read handlers were inspected on
  `codex/clarify-firehose-raw` at `8cbbf4f`. That checkout includes
  `docs/API_CONTRACT_AUDIT.md`; this SDK base does not include the control plane.
  M1/M2 make message-based routing simulation unreliable; M4 is daily-only
  usage; M5 requires preserving nulls; M7 requires visible pagination.
- [HTTPX async guide](https://www.python-httpx.org/async/) supports pooled
  `AsyncClient` requests and streaming with explicit closure.
- [HTTPX transports](https://www.python-httpx.org/advanced/transports/) supplies
  `MockTransport` for deterministic wire-contract tests without live credentials.
- [LangChain ChatOpenAI documentation](https://docs.langchain.com/oss/python/integrations/chat/openai)
  supports a custom `base_url`, but notes that nonstandard response fields are
  not preserved. It also converts `max_tokens` to `max_completion_tokens`.
  These are concrete reasons for a native gateway adapter: keep `_gateway` and
  send the `max_tokens` that the current Java DTO actually reads.
- [LangChain BaseChatModel reference](https://reference.langchain.com/python/langchain-core/language_models/chat_models)
  defines the custom model contract. Its installed source is checked alongside
  tests for invoke/ainvoke, prompt pipelines, batching and usage metadata.

## Recommended SDK functions and endpoints

| Priority | Function / endpoint | Why | Decision |
| --- | --- | --- | --- |
| P0 | Async chat, models, probes | FastAPI/async apps need nonblocking I/O | Implemented |
| P0 | Fix health URLs | Existing SDK probes target nonexistent paths | Implemented |
| P0 | Error details + Retry-After | Preserve failure context for callers | Implemented |
| P0 | Null content + raw tool calls | Prevent `None` becoming literal text | Implemented |
| P0 | Streaming content-type/error validation | Avoid silently losing JSON responses | Implemented; server SSE still pending |
| P1 | `GET /me`, orgs/projects | Discover identity and tenant scope | Read helpers implemented |
| P1 | `GET /orgs/{id}/usage` | Cost reports and budget automation | Implemented daily reads |
| P1 | Requests + `GET /requests/{id}` | Explain routing/fallbacks after a call | Implemented with cursor iteration |
| P1 | Providers, catalog, policies/versions | Discover available models and routing config | Read helpers implemented |
| P1 | Routing insights + audit logs | Compare route behavior and admin activity | Read helpers implemented |
| P1 | API-key metadata lists | Inspect keys without returning secrets | Read helper implemented |
| P1 | Native LangChain chat model | Preserve route/cost/usage in Runnable flows | Separate integration branch |
| P2 | Create/rotate/revoke keys | CI provisioning and service automation | Next iteration; idempotency + one-time secret handling required |
| P2 | Policy create/update/publish/simulate | Programmatic routing rollout | Defer until simulate M1/M2 resolved; honor If-Match/412 |
| P2 | Async control-plane reads | Large asynchronous monitoring clients | Follow-up after sync contract is exercised |
| P2 | Tools + response_format end to end | LangChain agents and structured output | SDK pass-through only; extend backend DTO/adapters first |
| P2 | `POST /v1/embeddings` | RAG/vector indexing | Design P2; no server implementation, no callable SDK stub |
| P3 | `/v1/responses`, `/v1/messages` | Native advanced provider protocols | Explicitly deferred by API_DESIGN §3.5 |

## LangChain choice (implemented on sdk-python-langchain)

Use a small `BaseChatModel` adapter over `GatewayClient` and
`AsyncGatewayClient`, with `langchain-core` as an optional dependency. Avoid a
mandatory full `langchain`/`langchain-openai` dependency for ordinary SDK users.
Support `invoke`, `ainvoke`, prompt pipes and batching; return gateway fields,
headers, selected model, finish reason and token usage on `AIMessage`.
Keep API keys excluded from repr/serialization and expose explicit closure.
The implementation uses `langchain-core>=1,<2` as an optional extra; the default
SDK import does not load it. HTTP requests remain on `/v1/chat/completions`.

For this Java backend, `stream`/`astream` must use LangChain's buffered fallback
and tools/stop/structured-output requests must fail clearly unless the caller
explicitly enables future-backend pass-through. Do not advertise token streaming,
agent tool execution, or embeddings as current server capabilities.

## Validation boundary

Contract tests validate paths, payloads, auth headers, token refresh hooks,
pagination, SSE parsing, error envelopes and LangChain conversion using mocked
HTTP responses. Wheel/sdist builds validate packaging. These do not prove an
OIDC deployment, a live provider, actual SSE, or gateway header enforcement.
No live requests with real credentials or changes to backend/schema are needed.

Verified results on Python 3.12 / HTTPX 0.28.1:

- Expansion commit: 34 tests passed; wheel and sdist built.
- LangChain branch: 50 tests passed with both `langchain-core` 1.6.6 and 1.0.0
  (the minimum package was isolated on PYTHONPATH with the same dependencies).
- Integration wheel/sdist built; wheel metadata keeps LangChain optional.
- All SDK modules parse with Python 3.10 syntax rules. Python 3.10 runtime and
  dependency combinations beyond the versions above were not exercised.
