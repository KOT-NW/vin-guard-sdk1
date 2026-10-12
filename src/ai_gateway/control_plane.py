"""Read-only dashboard API client with JWT auth and cursor pagination.

These routes belong to the serverless control plane, not the Java data plane.
Dict payloads deliberately preserve unknown fields and nullable wire values.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from ._shared import raise_for_status, translate_transport_errors
from .errors import GatewayError


@dataclass(frozen=True)
class Page:
    data: list[dict[str, Any]]
    next_cursor: str | None
    has_more: bool

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Page:
        if not isinstance(payload, dict):
            raise ValueError("Expected {data, pagination} control-plane envelope")
        pagination = payload.get("pagination")
        data = payload.get("data")
        if not isinstance(data, list) or not isinstance(pagination, dict):
            raise ValueError("Expected {data, pagination} control-plane envelope")
        more = pagination.get("hasMore")
        cursor = pagination.get("nextCursor")
        if not isinstance(more, bool) or (cursor is not None and not isinstance(cursor, str)):
            raise ValueError("Invalid pagination fields")
        if more and not cursor:
            raise ValueError("hasMore=true requires a non-empty nextCursor")
        return cls(data=data, next_cursor=cursor, has_more=more)


def _id(value: str) -> str:
    if not value or value in (".", ".."):
        raise ValueError("A resource ID is required")
    return quote(value, safe="")


class ControlPlaneClient:
    """Use an OIDC access token, or explicitly opt into a dev principal.

    Token callables are resolved on every request so applications can refresh
    tokens themselves. Gateway API keys are not dashboard credentials.
    """

    def __init__(self, *, base_url: str,
                 access_token: str | Callable[[], str] | None = None,
                 dev_subject: str | None = None,
                 timeout: float | httpx.Timeout = 30.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        if bool(access_token) == bool(dev_subject):
            raise ValueError("Provide exactly one of access_token or dev_subject")
        self._access_token = access_token
        self._dev_subject = dev_subject
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout,
                                  transport=transport)

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        if self._dev_subject:
            headers = {"x-dev-sub": self._dev_subject}
        else:
            token = self._access_token() if callable(self._access_token) else self._access_token
            if not token:
                raise ValueError("Access token provider returned an empty token")
            headers = {"Authorization": f"Bearer {token}"}
        if "limit" in params and params["limit"] is not None:
            limit = params["limit"]
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
                raise ValueError("limit must be an integer between 1 and 100")
        with translate_transport_errors():
            response = self._http.get(path, headers=headers,
                                      params={k: v for k, v in params.items() if v is not None})
        raise_for_status(response)
        try:
            return response.json()
        except ValueError as exc:
            raise GatewayError("Control plane returned an unexpected response body",
                               code="invalid_response", status=response.status_code,
                               request_id=response.headers.get("x-gateway-request-id"),
                               payload=None, headers=response.headers) from exc

    def _page(self, path: str, **params: Any) -> Page:
        return Page.from_payload(self._get(path, **params))

    def me(self) -> dict[str, Any]:
        return self._get("/me")

    def list_orgs(self, **params: Any) -> Page:
        return self._page("/orgs", **params)

    def list_projects(self, org_id: str, **params: Any) -> Page:
        return self._page(f"/orgs/{_id(org_id)}/projects", **params)

    def list_api_keys(self, project_id: str, **params: Any) -> Page:
        return self._page(f"/projects/{_id(project_id)}/api-keys", **params)

    def list_providers(self, org_id: str, **params: Any) -> Page:
        return self._page(f"/orgs/{_id(org_id)}/providers", **params)

    def list_models(self, org_id: str, **params: Any) -> Page:
        return self._page(f"/orgs/{_id(org_id)}/models", **params)

    def list_routing_policies(self, org_id: str, **params: Any) -> Page:
        return self._page(f"/orgs/{_id(org_id)}/routing-policies", **params)

    def list_policy_versions(self, policy_id: str, **params: Any) -> Page:
        return self._page(f"/routing-policies/{_id(policy_id)}/versions", **params)

    def get_usage(self, org_id: str, *, from_: str | None = None,
                  to: str | None = None, group_by: str = "model") -> Page:
        if group_by not in ("model", "provider", "project", "key", "tag"):
            raise ValueError("Unsupported usage group_by")
        # The current control plane only supports daily rollups (audit M4).
        return self._page(f"/orgs/{_id(org_id)}/usage",
                          **{"from": from_, "to": to, "groupBy": group_by})

    def list_requests(self, org_id: str, **params: Any) -> Page:
        return self._page(f"/orgs/{_id(org_id)}/requests", **params)

    def get_request(self, request_id: str) -> dict[str, Any]:
        return self._get(f"/requests/{_id(request_id)}")

    def get_routing_insights(self, org_id: str, *, from_: str | None = None,
                             to: str | None = None) -> dict[str, Any]:
        return self._get(f"/orgs/{_id(org_id)}/routing/insights", **{"from": from_, "to": to})

    def list_audit_logs(self, org_id: str, **params: Any) -> Page:
        return self._page(f"/orgs/{_id(org_id)}/audit-logs", **params)

    def iter_items(self, list_method: Callable[..., Page], *args: Any,
                   **params: Any) -> Iterator[dict[str, Any]]:
        """Traverse pages lazily; fail if the server repeats a cursor."""
        seen = {params["cursor"]} if params.get("cursor") else set()
        while True:
            page = list_method(*args, **params)
            yield from page.data
            if not page.has_more:
                return
            if not page.next_cursor or page.next_cursor in seen:
                raise ValueError("Pagination cursor did not advance")
            seen.add(page.next_cursor)
            params["cursor"] = page.next_cursor

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> ControlPlaneClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
