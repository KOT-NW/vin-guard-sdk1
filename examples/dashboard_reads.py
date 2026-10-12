"""Dashboard requests require a separate OIDC access token."""

import os

from ai_gateway import ControlPlaneClient


def main():
    with ControlPlaneClient(
        base_url=os.environ["CONTROL_PLANE_BASE_URL"],
        access_token=os.environ["CONTROL_PLANE_ACCESS_TOKEN"],
    ) as client:
        org_id = os.environ["GATEWAY_ORG_ID"]
        print(client.get_usage(org_id, group_by="project").data)
        for request in client.iter_items(client.list_requests, org_id, limit=100):
            print(request["requestId"], request["route"])


if __name__ == "__main__":
    main()
