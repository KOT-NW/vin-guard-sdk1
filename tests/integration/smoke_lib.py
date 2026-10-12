"""Helpers for the real gateway smoke test.

This module is intentionally not part of the installable SDK. It uses raw
HTTPX for control-plane mutations the SDK does not expose (create org/project/
key), and it never reads Valkey or the database directly.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import httpx

SECRET_RE = re.compile(r"gw_live_[A-Za-z0-9_]+")
ClientFactory = Callable[[str, str], "ControlPlaneHTTP"]


class SmokeSetupError(RuntimeError):
    """Raised when opt-in smoke prerequisites are missing or broken."""


def strip_secrets(text: str) -> str:
    return SECRET_RE.sub("gw_live_<redacted>", str(text))


@dataclass
class SmokeConfig:
    control_plane_url: str
    gateway_url: str
    pepper: str
    internal_token: str
    ready_timeout: float
    publish_timeout: float
    publish_command: list[str] | None
    require_langchain: bool = False
    suffix: str = field(default_factory=lambda: uuid.uuid4().hex[:10])

    @property
    def gateway_root(self) -> str:
        base = self.gateway_url.rstrip("/")
        return base[:-3] if base.endswith("/v1") else base

    def principal(self, side: str) -> tuple[str, str]:
        return (f"smoke-{side}-{self.suffix}", f"smoke-{side}-{self.suffix}@example.test")


class ControlPlaneHTTP:
    """Minimal dev-mode control-plane client for mutations and bootstrap.

    Bootstrap uses the real local mechanism (``x-dev-sub`` + ``x-dev-email``)
    so the JIT user is provisioned and membership checks stay enabled.
    """

    def __init__(self, base_url: str, sub: str, email: str, *,
                 transport: httpx.BaseTransport | None = None) -> None:
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0,
                                  transport=transport)
        self._headers = {"x-dev-sub": sub, "x-dev-email": email}

    def request(self, method: str, path: str, *, json: Any = None,
                idempotency_key: str | None = None) -> httpx.Response:
        headers = dict(self._headers)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return self._http.request(method, path, json=json, headers=headers)

    def expect(self, method: str, path: str, *, status: Iterable[int] = (200,),
               json: Any = None, idempotency_key: str | None = None) -> httpx.Response:
        response = self.request(method, path, json=json, idempotency_key=idempotency_key)
        if response.status_code not in status:
            raise AssertionError(
                f"{method} {path} -> {response.status_code}: "
                f"{strip_secrets(response.text)[:300]}"
            )
        return response

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "ControlPlaneHTTP":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class Fixture:
    """Resources created by one smoke run, tracked as soon as each one exists.

    Fields are populated incrementally by :func:`create_fixture` so a failure
    part-way through a run still knows what was already created. Only IDs are
    retained; ``secret`` is the one-time gateway key and never logged.
    """

    sub_a: str
    email_a: str
    sub_b: str
    email_b: str
    org_id: str | None = None
    project_id: str | None = None
    key_id: str | None = None
    secret: str | None = None
    prefix: str | None = None

    @property
    def created(self) -> list[str]:
        labels: list[str] = []
        if self.org_id:
            labels.append(f"org {self.org_id}")
        if self.project_id:
            labels.append(f"project {self.project_id}")
        if self.key_id:
            labels.append(f"key {self.key_id}")
        return labels


@dataclass
class CleanupReport:
    """Outcome of a best-effort cleanup; identifies anything left behind."""

    key_delete_status: int | None = None
    project_delete_status: int | None = None
    retained: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = []
        if self.key_delete_status is not None:
            parts.append(f"key DELETE -> {self.key_delete_status}")
        if self.project_delete_status is not None:
            parts.append(f"project DELETE -> {self.project_delete_status}")
        if self.retained:
            parts.append("retained: " + ", ".join(self.retained))
        if self.errors:
            parts.append("errors: " + ", ".join(self.errors))
        return "; ".join(parts) if parts else "nothing to clean up"


def default_client_factory(config: SmokeConfig) -> ClientFactory:
    return lambda sub, email: ControlPlaneHTTP(config.control_plane_url, sub, email)


def bootstrap_principal(cp: ControlPlaneHTTP) -> dict[str, Any]:
    response = cp.expect("GET", "/me")
    return response.json()


def create_fixture(config: SmokeConfig, *,
                   client_factory: ClientFactory | None = None) -> Fixture:
    """Create an isolated org/project/key, tracking each resource as it lands.

    If creation fails part-way, the already-created resources are cleaned up
    best-effort and reported on the original exception (``cleanup_report``)
    before it is re-raised unchanged, so cleanup never masks the root cause.
    """
    factory = client_factory or default_client_factory(config)
    sub_a, email_a = config.principal("a")
    sub_b, email_b = config.principal("b")
    fixture = Fixture(sub_a=sub_a, email_a=email_a, sub_b=sub_b, email_b=email_b)
    with factory(sub_a, email_a) as cp:
        try:
            bootstrap_principal(cp)
            org = cp.expect("POST", "/orgs", status=(201,),
                            json={"name": f"smoke-org-{config.suffix}", "plan": "free"},
                            idempotency_key=f"smoke-org-{config.suffix}").json()
            fixture.org_id = org["id"]
            project = cp.expect("POST", f"/orgs/{fixture.org_id}/projects", status=(201,),
                                json={"name": f"smoke-project-{config.suffix}"},
                                idempotency_key=f"smoke-project-{config.suffix}").json()
            fixture.project_id = project["id"]
            key = cp.expect("POST", f"/projects/{fixture.project_id}/api-keys", status=(201,),
                            json={"name": f"smoke-key-{config.suffix}"},
                            idempotency_key=f"smoke-key-{config.suffix}").json()
            fixture.key_id = key["id"]
            fixture.secret = key["secret"]
            fixture.prefix = key["prefix"]
        except BaseException as exc:
            report = _cleanup(factory, fixture)
            _report_retention(report)
            setattr(exc, "cleanup_report", report)
            raise
    return fixture


def revoke_key(config: SmokeConfig, fixture: Fixture, *,
               client_factory: ClientFactory | None = None) -> dict[str, Any]:
    factory = client_factory or default_client_factory(config)
    with factory(fixture.sub_a, fixture.email_a) as cp:
        response = cp.expect("DELETE", f"/api-keys/{fixture.key_id}")
        return response.json()


def rotate_key(config: SmokeConfig, fixture: Fixture, *, overlap_seconds: int = 0,
               client_factory: ClientFactory | None = None) -> dict[str, Any]:
    """Rotate the fixture key through the raw control-plane API.

    The SDK intentionally has no key mutation; rotate is exercised with HTTPX.
    On success the fixture's one-time ``secret`` is replaced with the new one.
    """
    factory = client_factory or default_client_factory(config)
    with factory(fixture.sub_a, fixture.email_a) as cp:
        response = cp.expect("POST", f"/api-keys/{fixture.key_id}/rotate",
                             json={"overlapSeconds": overlap_seconds})
        rotated = response.json()
    fixture.secret = rotated["secret"]
    fixture.prefix = rotated.get("prefix", fixture.prefix)
    return rotated


def cleanup(config: SmokeConfig, fixture: Fixture, *,
            client_factory: ClientFactory | None = None) -> CleanupReport:
    """Best-effort cleanup of only the resources this run created."""
    factory = client_factory or default_client_factory(config)
    return _cleanup(factory, fixture)


def _cleanup(factory: ClientFactory, fixture: Fixture) -> CleanupReport:
    report = CleanupReport()
    if not fixture.created:
        return report
    with factory(fixture.sub_a, fixture.email_a) as cp:
        if fixture.key_id:
            report.key_delete_status = _delete(cp, f"/api-keys/{fixture.key_id}", report)
            if report.key_delete_status in (200, 204):
                # DELETE revokes, it does not remove the row; the key stays but is
                # unusable. Reported for transparency, no secret included.
                report.retained.append(f"key {fixture.key_id} (revoked, row retained)")
            elif report.key_delete_status not in (404,):
                report.retained.append(
                    f"key {fixture.key_id} (DELETE -> {report.key_delete_status})")
        if fixture.project_id:
            report.project_delete_status = _delete(cp, f"/projects/{fixture.project_id}", report)
            if report.project_delete_status in (200, 204):
                pass
            elif report.project_delete_status == 409:
                # `project is not empty`: revoked key rows still exist, so the
                # control plane refuses. Documented retention, not an error.
                report.retained.append(
                    f"project {fixture.project_id} (DELETE -> 409, keys retained)")
            else:
                report.retained.append(
                    f"project {fixture.project_id} (DELETE -> {report.project_delete_status})")
    if fixture.org_id:
        # There is no DELETE /orgs route; the org is always retained.
        report.retained.append(f"org {fixture.org_id} (no delete endpoint)")
    return report


def _delete(cp: ControlPlaneHTTP, path: str, report: CleanupReport) -> int | None:
    try:
        return cp.request("DELETE", path).status_code
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort by design
        report.errors.append(f"DELETE {path}: {strip_secrets(str(exc))[:120]}")
        return None


def _report_retention(report: CleanupReport) -> None:
    if report.retained or report.errors:
        print(f"[smoke] cleanup after failure: {report.summary()}", file=sys.stderr)


def wait_until(predicate: Callable[[], Any], *, timeout: float, describe: str,
               interval: float = 1.0) -> Any:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except (httpx.HTTPError, OSError, AssertionError) as exc:
            last_error = exc
        time.sleep(interval)
    detail = f" (last: {strip_secrets(str(last_error))[:200]})" if last_error else ""
    raise SmokeSetupError(f"Timed out after {timeout:.0f}s waiting for {describe}{detail}")


def wait_for_control_plane(config: SmokeConfig) -> None:
    sub, email = config.principal("a")

    def probe() -> bool:
        with ControlPlaneHTTP(config.control_plane_url, sub, email) as cp:
            return cp.request("GET", "/me").status_code == 200

    wait_until(probe, timeout=config.ready_timeout, describe="control plane /me")


def gateway_ready_version(config: SmokeConfig) -> int:
    response = httpx.get(f"{config.gateway_root}/readyz", timeout=10.0)
    if response.status_code != 200:
        return 0
    try:
        return int(response.json().get("snapshotVersion") or 0)
    except (ValueError, AttributeError):
        return 0


def wait_for_gateway(config: SmokeConfig) -> int:
    """Wait for /healthz, then return the current snapshot version (may be 0)."""

    def probe() -> bool:
        response = httpx.get(f"{config.gateway_root}/healthz", timeout=10.0)
        if response.status_code != 200 or response.json().get("status") != "ok":
            raise AssertionError(f"healthz {response.status_code}")
        return True

    wait_until(probe, timeout=config.ready_timeout, describe="gateway /healthz")
    return gateway_ready_version(config)


def trigger_publisher(config: SmokeConfig) -> str:
    """Invoke the real publisher Lambda; fall back to the real scheduler.

    With a configured command the AWS CLI exit code alone is not proof of
    success: a synchronous invoke can exit 0 and still report
    ``FunctionError=Unhandled``. The invocation metadata JSON is validated.
    """
    if config.publish_command is None:
        return "scheduler"
    try:
        result = subprocess.run(config.publish_command, capture_output=True, text=True,
                                timeout=60, env=_aws_env())
    except subprocess.TimeoutExpired as exc:
        raise SmokeSetupError("publisher invoke timed out after 60s") from exc
    if result.returncode != 0:
        raise SmokeSetupError(
            "publisher invoke failed: " + strip_secrets(result.stderr or result.stdout)[:400]
        )
    assert_invoke_succeeded(result.stdout)
    return "cli"


def assert_invoke_succeeded(stdout: str) -> dict[str, Any]:
    """Validate ``aws lambda invoke`` metadata; exit 0 alone is not success."""
    text = (stdout or "").strip()
    if not text:
        raise SmokeSetupError(
            "publisher invoke emitted no JSON metadata; exit 0 alone is not proof of success")
    try:
        metadata = json.loads(text)
    except ValueError as exc:
        raise SmokeSetupError(
            "publisher invoke metadata is not valid JSON: " + strip_secrets(text)[:200]) from exc
    if not isinstance(metadata, dict):
        raise SmokeSetupError("publisher invoke metadata is not a JSON object")
    function_error = metadata.get("FunctionError")
    if function_error:
        raise SmokeSetupError(
            f"publisher invoke reported FunctionError={function_error}: "
            + strip_secrets(text)[:400])
    status = metadata.get("StatusCode")
    if status != 200:
        raise SmokeSetupError(
            f"publisher invoke StatusCode={status!r} (expected 200): "
            + strip_secrets(text)[:200])
    return metadata


def _aws_env() -> dict[str, str]:
    import os

    env = dict(os.environ)
    env.setdefault("AWS_ACCESS_KEY_ID", "test")
    env.setdefault("AWS_SECRET_ACCESS_KEY", "test")
    env.setdefault("AWS_REGION", "us-east-1")
    # Keep the invoke output to JSON metadata on stdout (no pager prompts).
    env.setdefault("AWS_PAGER", "")
    return env


def publish_and_wait(config: SmokeConfig, before_version: int) -> int:
    """Publish, then wait for the gateway snapshot to advance."""
    mode = trigger_publisher(config)
    if mode == "scheduler":
        print(f"[smoke] waiting for the scheduled publisher (baseline version {before_version})")

    def advanced() -> int | bool:
        version = gateway_ready_version(config)
        return version if version > before_version else False

    return wait_until(advanced, timeout=config.publish_timeout,
                      describe=f"snapshot version > {before_version} after publish")


def parse_publish_command(value: str | None) -> list[str] | None:
    if not value:
        return None
    return shlex.split(value)
