"""Offline unit checks for the smoke harness helpers (no live services)."""

from __future__ import annotations

import subprocess
from typing import Any, Callable

import pytest

from .smoke_lib import (CleanupReport, Fixture, SmokeConfig, SmokeSetupError,
                        cleanup, create_fixture, parse_publish_command,
                        rotate_key, strip_secrets, trigger_publisher)


class _Response:
    def __init__(self, status: int, payload: Any = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self) -> Any:
        if self._payload is None:
            raise ValueError("no JSON body")
        return self._payload


class _FakeCP:
    """Scripted stand-in for ControlPlaneHTTP; records calls for assertions."""

    def __init__(self, routes: dict[tuple[str, str], Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, str]] = []
        self.closed = False

    def request(self, method: str, path: str, *, json: Any = None,
                idempotency_key: str | None = None) -> _Response:
        self.calls.append((method, path))
        handler = self.routes.get((method, path))
        if handler is None:
            raise AssertionError(f"unexpected request {method} {path}")
        if isinstance(handler, BaseException):
            raise handler
        return handler() if callable(handler) else handler

    def expect(self, method: str, path: str, *, status=(200,), json: Any = None,
               idempotency_key: str | None = None) -> _Response:
        response = self.request(method, path, json=json, idempotency_key=idempotency_key)
        if response.status_code not in status:
            raise AssertionError(f"{method} {path} -> {response.status_code}")
        return response

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> "_FakeCP":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _factory(routes: dict[tuple[str, str], Any]) -> Callable[[str, str], _FakeCP]:
    return lambda sub, email: _FakeCP(routes)


def _fixture(**overrides) -> Fixture:
    base = dict(sub_a="s-a", email_a="a@example.test", sub_b="s-b", email_b="b@example.test",
                org_id="org_1", project_id="prj_1", key_id="key_1",
                secret="gw_live_key_1_X", prefix="gw_live_key_1")
    base.update(overrides)
    return Fixture(**base)


def _config(**overrides) -> SmokeConfig:
    base = dict(control_plane_url="https://cp.example/api/v1",
                gateway_url="https://gw.example/v1",
                pepper="dev-pepper", internal_token="tok",
                ready_timeout=1.0, publish_timeout=1.0, publish_command=None,
                suffix="abc123")
    base.update(overrides)
    return SmokeConfig(**base)


def test_strip_secrets_redacts_gateway_keys():
    text = "key gw_live_key_abc_SECRETVALUE failed, other gw_live_key_def_X"
    redacted = strip_secrets(text)
    assert "SECRETVALUE" not in redacted
    assert redacted.count("gw_live_<redacted>") == 2


def test_parse_publish_command():
    assert parse_publish_command(None) is None
    assert parse_publish_command("") is None
    assert parse_publish_command("aws lambda invoke --function-name f out.json") == [
        "aws", "lambda", "invoke", "--function-name", "f", "out.json"]


def test_gateway_root_strips_v1_only():
    assert _config(gateway_url="http://localhost:8080/v1").gateway_root == "http://localhost:8080"
    assert _config(gateway_url="http://localhost:8080/v1/").gateway_root == "http://localhost:8080"
    assert _config(gateway_url="http://localhost:8080/").gateway_root == "http://localhost:8080"


def test_principals_are_unique_and_email_present():
    sub, email = _config().principal("a")
    assert sub == "smoke-a-abc123"
    assert email.endswith("@example.test")


def test_create_fixture_tracks_resources_on_success():
    routes = {
        ("GET", "/me"): _Response(200, {"sub": "s-a"}),
        ("POST", "/orgs"): _Response(201, {"id": "org_1"}),
        ("POST", "/orgs/org_1/projects"): _Response(201, {"id": "prj_1"}),
        ("POST", "/projects/prj_1/api-keys"): _Response(
            201, {"id": "key_1", "secret": "gw_live_key_1_X", "prefix": "gw_live_key_1"}),
    }
    fixture = create_fixture(_config(), client_factory=_factory(routes))
    assert fixture.created == ["org org_1", "project prj_1", "key key_1"]
    assert fixture.secret == "gw_live_key_1_X"


def test_create_fixture_partial_failure_cleans_up_and_preserves_error():
    # Key creation fails; the already-created project refuses deletion with 409
    # (revoked/absent key rows), which must be reported, not swallow the error.
    routes: dict[tuple[str, str], Any] = {
        ("GET", "/me"): _Response(200, {"sub": "s-a"}),
        ("POST", "/orgs"): _Response(201, {"id": "org_1"}),
        ("POST", "/orgs/org_1/projects"): _Response(201, {"id": "prj_1"}),
        ("POST", "/projects/prj_1/api-keys"): AssertionError("key create failed: 500"),
        ("DELETE", "/projects/prj_1"): _Response(409, text="project is not empty"),
    }

    with pytest.raises(AssertionError, match="key create failed") as caught:
        create_fixture(_config(), client_factory=_factory(routes))

    report: CleanupReport = caught.value.cleanup_report
    assert report.project_delete_status == 409
    assert any("project prj_1" in item for item in report.retained)
    assert any("org org_1" in item for item in report.retained)
    assert "gw_live" not in report.summary()
    assert isinstance(caught.value, AssertionError)


def test_cleanup_reports_revoked_key_and_nonempty_project():
    routes = {
        ("DELETE", "/api-keys/key_1"): _Response(200, {"id": "key_1"}),
        ("DELETE", "/projects/prj_1"): _Response(409, text="project is not empty"),
    }
    report = cleanup(_config(), _fixture(), client_factory=_factory(routes))
    assert report.key_delete_status == 200
    assert report.project_delete_status == 409
    assert any("revoked" in item for item in report.retained)
    assert any("409" in item for item in report.retained)
    assert any("org org_1" in item for item in report.retained)
    assert not report.errors


def test_cleanup_with_nothing_created_is_empty():
    report = cleanup(_config(), _fixture(org_id=None, project_id=None, key_id=None,
                                         secret=None, prefix=None))
    assert report.summary() == "nothing to clean up"


def test_rotate_key_replaces_fixture_secret():
    routes = {
        ("POST", "/api-keys/key_1/rotate"): _Response(
            200, {"id": "key_1", "secret": "gw_live_key_1_Y", "prefix": "gw_live_key_1"}),
    }
    fixture = _fixture()
    rotated = rotate_key(_config(), fixture, overlap_seconds=0,
                         client_factory=_factory(routes))
    assert rotated["secret"] == "gw_live_key_1_Y"
    assert fixture.secret == "gw_live_key_1_Y"


def _completed(returncode: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=["aws"], returncode=returncode,
                                       stdout=stdout, stderr=stderr)


def _publish_config() -> SmokeConfig:
    return _config(publish_command=["aws", "lambda", "invoke"])


def test_trigger_publisher_scheduler_mode_when_unconfigured():
    assert trigger_publisher(_config()) == "scheduler"


def test_trigger_publisher_accepts_synchronous_success(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(0, '{"StatusCode":200,"ExecutedVersion":"$LATEST"}'))
    assert trigger_publisher(_publish_config()) == "cli"


def test_trigger_publisher_rejects_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: _completed(1, stderr="AccessDenied: boom"))
    with pytest.raises(SmokeSetupError, match="failed"):
        trigger_publisher(_publish_config())


def test_trigger_publisher_rejects_function_error_with_exit_zero(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: _completed(0, '{"StatusCode":200,"FunctionError":"Unhandled"}'))
    with pytest.raises(SmokeSetupError, match="FunctionError"):
        trigger_publisher(_publish_config())


def test_trigger_publisher_rejects_empty_metadata(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(0, ""))
    with pytest.raises(SmokeSetupError, match="no JSON metadata"):
        trigger_publisher(_publish_config())


def test_trigger_publisher_rejects_malformed_metadata(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(0, "not-json"))
    with pytest.raises(SmokeSetupError, match="not valid JSON"):
        trigger_publisher(_publish_config())


def test_trigger_publisher_rejects_non_200_status(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _completed(0, '{"StatusCode":500}'))
    with pytest.raises(SmokeSetupError, match="StatusCode=500"):
        trigger_publisher(_publish_config())


def test_trigger_publisher_rejects_timeout(monkeypatch):
    def boom(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="aws", timeout=60)

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(SmokeSetupError, match="timed out"):
        trigger_publisher(_publish_config())
