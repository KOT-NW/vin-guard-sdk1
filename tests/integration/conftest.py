"""Opt-in gate and configuration for the real end-to-end smoke test.

Default runs (`pytest sdk-python/tests`) skip these tests. The dedicated smoke
job sets ``AIGW_SMOKE=1``; from then on a missing config, unreachable service
or bad prerequisite fails the run instead of skipping or xpassing.
"""

from __future__ import annotations

import os

import pytest

from .smoke_lib import SmokeConfig, parse_publish_command


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: real control-plane + gateway smoke (opt-in)")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("AIGW_SMOKE") == "1":
        return
    skip = pytest.mark.skip(
        reason="integration smoke: set AIGW_SMOKE=1 and the AIGW_* config (see README)")
    for item in items:
        if item.get_closest_marker("integration"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def smoke_config() -> SmokeConfig:
    if os.environ.get("AIGW_SMOKE") != "1":
        pytest.skip("AIGW_SMOKE is not set")
    required = [key for key in ("AIGW_CONTROL_PLANE_URL", "AIGW_GATEWAY_URL")
                if not os.environ.get(key)]
    if required:
        pytest.fail("AIGW_SMOKE=1 but missing required environment: " + ", ".join(required))
    return SmokeConfig(
        control_plane_url=os.environ["AIGW_CONTROL_PLANE_URL"],
        gateway_url=os.environ["AIGW_GATEWAY_URL"],
        pepper=os.environ.get("AIGW_PEPPER", "dev-pepper"),
        internal_token=os.environ.get("AIGW_INTERNAL_TOKEN", "dev-internal-token"),
        ready_timeout=float(os.environ.get("AIGW_READY_TIMEOUT", "180")),
        publish_timeout=float(os.environ.get("AIGW_PUBLISH_TIMEOUT", "180")),
        publish_command=parse_publish_command(os.environ.get("AIGW_PUBLISH_COMMAND")),
        require_langchain=os.environ.get("AIGW_REQUIRE_LANGCHAIN") == "1",
    )


@pytest.fixture(scope="session")
def smoke_service(smoke_config: SmokeConfig):
    """Wait for both services before the single ordered smoke test runs."""
    from .smoke_lib import wait_for_control_plane, wait_for_gateway

    wait_for_control_plane(smoke_config)
    wait_for_gateway(smoke_config)
    return smoke_config
