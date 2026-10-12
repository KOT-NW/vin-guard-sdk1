# End-to-end smoke test (opt-in)

`test_e2e_smoke.py` runs the real control plane → publisher → Valkey → Java
gateway path. Only the LLM upstream is substituted: the request uses
`openai/stub-sdk-smoke`, answered by the backend's real `StubAdapter` (no
external provider, cost or credentials).

The offline helpers in `test_smoke_helpers.py` run without any service; the
opt-in gate below covers `test_e2e_smoke.py` only.

## Prerequisites

1. Docker + Floci up: `docker compose -f controlplane/infra/floci.yml up -d`.
   Floci publishes the emulated Postgres and Valkey on host ports `7001` and
   `6379`.
2. Control plane deployed: `bash controlplane/infra/deploy-local.sh` (needs
   Node >= 22, AWS CLI v2, `psql`, `zip`). The script bootstraps its Postgres
   from the host via `BOOTSTRAP_DATABASE_URL` (default
   `postgres://app:app@localhost:7001/postgres`) while the Lambdas keep using
   the in-network `floci:7001`; override the bootstrap URL if your host ports
   differ. It reads with a bounded deadline and fails with a redacted error
   instead of hanging.
3. Java gateway running against the same Postgres, Valkey and pepper:

   ```bash
   ( cd backend && ./mvnw -q -Dmaven.test.skip=true package )
   KEY_PEPPER=dev-pepper \
   DB_URL=jdbc:postgresql://localhost:7001/postgres DB_USER=app DB_PASSWORD=app \
   REDIS_HOST=localhost REDIS_PORT=6379 SPRING_FLYWAY_ENABLED=false \
   java -jar backend/target/*.jar
   ```

   `deploy-local.sh` writes the control-plane pepper as `dev-pepper`, but the
   Java default is `dev-pepper-change-me`, so `KEY_PEPPER` must match.
   `deploy-local.sh` also applies the schema (V1–V6) directly with `psql`, so
   that database has no Flyway history. `SPRING_FLYWAY_ENABLED=false` tells the
   standalone data plane to trust that pre-applied schema — only use it for
   this local harness. Run the gateway normally (Flyway enabled) against its own
   empty database.
4. SDK installed with the adapter: `pip install './sdk-python[dev,langchain]'`.

## Run

Set `AIGW_CONTROL_PLANE_URL` to the URL printed by `deploy-local.sh` (ends with
`/api/v1`) and run:

```bash
AIGW_SMOKE=1 \
AIGW_CONTROL_PLANE_URL='http://localhost:4566/execute-api/aigw-cp/$default/api/v1' \
AIGW_GATEWAY_URL='http://localhost:8080/v1' \
AIGW_REQUIRE_LANGCHAIN=1 \
AIGW_PUBLISH_COMMAND="aws --endpoint-url http://localhost:4566 --region us-east-1 lambda invoke --function-name aigw-publisher --payload {} --cli-binary-format raw-in-base64-out --no-cli-pager --output json /tmp/publish.json" \
python -m pytest sdk-python/tests/integration -q -s
```

The full default suite (unit + contract + integration) is run with:

```bash
python -m pytest sdk-python/tests -q
```

With no `AIGW_SMOKE=1`, the single integration test is reported as skipped;
prefer the explicit command above over recording a stale test count.

## Configuration

| Variable | Required | Default | Meaning |
| --- | --- | --- | --- |
| `AIGW_SMOKE` | yes to opt in | unset | Without `1` the integration test is skipped. |
| `AIGW_CONTROL_PLANE_URL` | yes | - | Full control-plane base including `/api/v1`. |
| `AIGW_GATEWAY_URL` | yes | - | Data-plane base including `/v1`. |
| `AIGW_REQUIRE_LANGCHAIN` | no | `0` | `1` fails if the LangChain extra is missing. |
| `AIGW_PUBLISH_COMMAND` | no | scheduler | Command to invoke the publisher Lambda. Must print the `aws lambda invoke` metadata JSON (`StatusCode`/`FunctionError`); exit 0 alone is not success. Without it the test waits for the real 1-minute scheduler. |
| `AIGW_PEPPER` | no | `dev-pepper` | Documented only; the harness never HMACs itself. |
| `AIGW_READY_TIMEOUT` / `AIGW_PUBLISH_TIMEOUT` | no | `180` | Seconds for readiness and snapshot publication. |

Once opted in, a missing config, unreachable service or broken prerequisite
fails the run; it never falls back to skip/xpass. The harness uses raw HTTPX
only for control-plane mutations the SDK does not expose (create org/project/
key and rotate) and `x-dev-sub`/`x-dev-email` only as the documented local dev
principal. It does not read the database or Valkey directly and does not assert
that chat produces usage/trace while the producer is not wired.

## Lifecycle, retention and key rotation

- The fixture is tracked as soon as each resource exists. If creation fails
  part-way, already-created resources are cleaned up best-effort and the
  original error is re-raised unchanged, with a `cleanup_report` attached.
- `DELETE /api-keys/{keyId}` only revokes: it keeps the key row. Because
  `DELETE /projects/{id}` returns `409 project is not empty` while any key row
  remains (even a revoked one), a shared local stack accumulates the fixture's
  org/project/key. Cleanup checks each response and prints the retained IDs
  (never the secret). There is no `DELETE /orgs` route.
- In CI the stack is ephemeral and `docker compose down -v` removes all fixture
  rows, so no cleanup is required there.
- Key rotation and revocation are verified through the raw control-plane API
  (`POST /api-keys/{keyId}/rotate`). The gateway now authenticates from the
  authoritative `gw:key:id:<keyId>` record only, so a stale `key:hash:<hmac>`
  index from an earlier generation can never grant access. `795dd26` made
  create/rotate HMAC the full presented key (`gw_live_<keyId>_<secret>`) to match
  the Java data plane.
- The smoke asserts the full lifecycle: overlap=0 rejects the previous secret
  immediately; overlap>0 accepts it during the window and rejects it after
  `prevExpiresAt`; revoke rejects the current, previous and every historical
  secret for the same key id (the latter with `invalid_api_key`).
- The publisher returns early when the outbox is empty, so an existing key gets
  its authoritative record via a deliberate backfill
  (`lambda invoke ... --payload '{"backfill": true}'
  --cli-binary-format raw-in-base64-out --no-cli-pager`). See
  `docs/KEY_RECORD_ROLLOUT_RUNBOOK.md` — backfill must succeed before the new
  Java reader is deployed.
- Keys created before `795dd26` stored `HMAC(secret-suffix)`, so they no longer
  authenticate against a data plane that hashes the full key. There is no
  automatic migration: only the hash is stored, so the original secret cannot be
  recovered. Re-issue (create or rotate) affected keys and republish.
