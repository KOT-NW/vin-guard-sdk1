"""Real end-to-end smoke: control plane -> publisher -> Valkey -> Java gateway.

The only substituted component is the LLM upstream: the request uses
``openai/stub-sdk-smoke`` so the backend's real ``StubAdapter`` answers without
external providers, cost or credentials. The control plane, the snapshot
publisher, Valkey, gateway authentication and HTTP transport are all real.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_gateway import (AsyncGatewayClient, ControlPlaneClient, GatewayClient,
                        GatewayError)
from .smoke_lib import (ControlPlaneHTTP, Fixture, SmokeConfig, cleanup,
                        create_fixture, gateway_ready_version, publish_and_wait,
                        revoke_key, rotate_key, wait_until)

pytestmark = pytest.mark.integration

STUB_MODEL = "openai/stub-sdk-smoke"
STUB_CONTENT = "stub-response for stub-sdk-smoke"
MESSAGES = [{"role": "user", "content": "smoke ping"}]
EXPECTED_PROMPT_TOKENS = len(MESSAGES) * 10
# Overlap window for the positive rotation case; the smoke then waits for it to
# lapse so the previous secret must be rejected.
ROTATE_OVERLAP_SECONDS = 30


def _new_client(config: SmokeConfig, secret: str) -> GatewayClient:
    return GatewayClient(api_key=secret, base_url=config.gateway_url)


def _expect_rejected(config: SmokeConfig, secret: str, *, code: str | None = None) -> GatewayError:
    with _new_client(config, secret) as client:
        with pytest.raises(GatewayError) as exc:
            client.chat_completions(model=STUB_MODEL, messages=MESSAGES)
    assert exc.value.status == 401
    if code is not None:
        assert exc.value.code == code, exc.value.code
    return exc.value


def _wait_rejected(config: SmokeConfig, secret: str, *, code: str, timeout: float) -> None:
    def predicate() -> bool | GatewayError:
        try:
            with _new_client(config, secret) as client:
                client.chat_completions(model=STUB_MODEL, messages=MESSAGES)
        except GatewayError as exc:
            return exc if exc.code == code else False
        return False

    wait_until(predicate, timeout=timeout,
               describe=f"secret rejected with {code} after overlap expiry")


def _assert_stub_completion(completion, *, prompt_tokens=EXPECTED_PROMPT_TOKENS) -> None:
    assert completion.content == STUB_CONTENT
    assert completion.model == STUB_MODEL
    assert completion.gateway.request_id.startswith("req_")
    assert completion.gateway.tier == "direct"
    assert completion.headers.get("x-gateway-provider") == "openai"
    assert completion.headers.get("x-gateway-model") == STUB_MODEL
    assert completion.usage.completion_tokens == 5
    assert completion.usage.prompt_tokens == prompt_tokens
    assert completion.usage.total_tokens == prompt_tokens + 5


def test_control_plane_to_gateway_smoke(smoke_service: SmokeConfig) -> None:
    config = smoke_service
    fixture: Fixture | None = None
    try:
        print("[smoke] 1. create isolated org/project/key via control plane", flush=True)
        fixture = create_fixture(config)

        print("[smoke] 2. wait for gateway, publish snapshot, wait for bump", flush=True)
        before = gateway_ready_version(config)
        publish_and_wait(config, before)

        with _new_client(config, fixture.secret) as client:
            assert client.health() == {"status": "ok"}
            ready = client.ready()
            assert ready["snapshotVersion"] > before

            print("[smoke] 3. models", flush=True)
            assert "auto" in {entry.id for entry in client.list_models()}

            print("[smoke] 4. sync chat", flush=True)
            _assert_stub_completion(client.chat_completions(model=STUB_MODEL, messages=MESSAGES))

        print("[smoke] 5. async chat", flush=True)

        async def async_chat() -> None:
            async with AsyncGatewayClient(api_key=fixture.secret,
                                          base_url=config.gateway_url) as client:
                completion = await client.chat_completions(model=STUB_MODEL, messages=MESSAGES)
                _assert_stub_completion(completion)

        asyncio.run(async_chat())

        print("[smoke] 6. control-plane reads", flush=True)
        with ControlPlaneClient(base_url=config.control_plane_url,
                                dev_subject=fixture.sub_a) as dashboard:
            assert dashboard.me()["sub"] == fixture.sub_a
            projects = list(dashboard.iter_items(dashboard.list_projects, fixture.org_id))
            assert fixture.project_id in {project["id"] for project in projects}
            keys = list(dashboard.iter_items(dashboard.list_api_keys, fixture.project_id))
            assert fixture.key_id in {key["id"] for key in keys}
            assert all("secret" not in key for key in keys)

        print("[smoke] 7. tenant isolation", flush=True)
        with ControlPlaneHTTP(config.control_plane_url, fixture.sub_b, fixture.email_b) as other:
            other.expect("GET", "/me")
        with ControlPlaneClient(base_url=config.control_plane_url,
                                dev_subject=fixture.sub_b) as intruder:
            with pytest.raises(GatewayError) as exc:
                intruder.list_projects(fixture.org_id)
            assert exc.value.status == 403
            assert exc.value.code == "no-membership"
            with pytest.raises(GatewayError) as exc:
                intruder.list_api_keys(fixture.project_id)
            assert exc.value.status == 403

        print("[smoke] 8. LangChain adapter through the real gateway", flush=True)
        _run_langchain(config, fixture.secret)

        print("[smoke] 9. invalid key rejected", flush=True)
        bogus = f"gw_live_key_bogus_{config.suffix}"
        with _new_client(config, bogus) as client:
            with pytest.raises(GatewayError) as exc:
                client.chat_completions(model=STUB_MODEL, messages=MESSAGES)
        assert exc.value.status == 401
        assert exc.value.code == "invalid_api_key"

        print("[smoke] 10. rotate overlap=0: new secret accepted, old rejected", flush=True)
        # 795dd26 hashes the full presented key (prefix_secret) like the Java data
        # plane. Rotate with no overlap, publish, and prove the authoritative
        # record rejects the previous secret immediately.
        created_secret = fixture.secret
        version_before_rotate = gateway_ready_version(config)
        rotate_key(config, fixture, overlap_seconds=0)
        rotated_secret = fixture.secret
        assert rotated_secret != created_secret, "rotate returned the same secret"
        publish_and_wait(config, version_before_rotate)
        with _new_client(config, rotated_secret) as client:
            _assert_stub_completion(client.chat_completions(model=STUB_MODEL, messages=MESSAGES))
        _expect_rejected(config, created_secret, code="key_expired")

        print("[smoke] 11. rotate overlap>0: prev valid in window, rejected after expiry", flush=True)
        version_before_overlap = gateway_ready_version(config)
        rotate_key(config, fixture, overlap_seconds=ROTATE_OVERLAP_SECONDS)
        current_secret = fixture.secret
        publish_and_wait(config, version_before_overlap)
        with _new_client(config, current_secret) as client:
            _assert_stub_completion(client.chat_completions(model=STUB_MODEL, messages=MESSAGES))
        with _new_client(config, rotated_secret) as client:
            _assert_stub_completion(client.chat_completions(model=STUB_MODEL, messages=MESSAGES))
        # The create-time secret is now two generations old: never accepted.
        _expect_rejected(config, created_secret, code="invalid_api_key")
        _wait_rejected(config, rotated_secret, code="key_expired",
                       timeout=ROTATE_OVERLAP_SECONDS + 60)

        print("[smoke] 12. revoke: current, previous and historical rejected", flush=True)
        version_before_revoke = gateway_ready_version(config)
        revoked = revoke_key(config, fixture)
        assert revoked["revokedAt"], "control plane did not record the revocation"
        publish_and_wait(config, version_before_revoke)
        _expect_rejected(config, current_secret, code="key_revoked")
        _expect_rejected(config, rotated_secret, code="key_revoked")
        _expect_rejected(config, created_secret, code="invalid_api_key")
    finally:
        if fixture is not None:
            report = cleanup(config, fixture)
            if report.retained or report.errors:
                print(f"[smoke] cleanup: {report.summary()}", flush=True)


def _run_langchain(config: SmokeConfig, secret: str) -> None:
    try:
        from ai_gateway.langchain import ChatGateway
    except ImportError as exc:
        if config.require_langchain:
            raise AssertionError("smoke job requires the langchain extra") from exc
        print("[smoke] langchain extra not installed; skipping adapter checks", flush=True)
        return

    with ChatGateway(api_key=secret, base_url=config.gateway_url,
                     model=STUB_MODEL, max_tokens=16) as model:
        message = model.invoke("smoke ping")
        assert message.content == STUB_CONTENT
        assert message.response_metadata["_gateway"]["requestId"].startswith("req_")
        assert message.response_metadata["gateway_headers"]["x-gateway-provider"] == "openai"

    async def ainvoke() -> None:
        async with ChatGateway(api_key=secret, base_url=config.gateway_url,
                               model=STUB_MODEL) as model:
            message = await model.ainvoke("smoke ping")
            assert message.content == STUB_CONTENT
            assert message.response_metadata["_gateway"]["requestId"].startswith("req_")

    asyncio.run(ainvoke())
