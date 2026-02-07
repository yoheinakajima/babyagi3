import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["GODADDY_API_KEY", "GODADDY_API_SECRET"], packages=["httpx"])
def godaddy_domain_available(domain: str, check_type: str = "FAST") -> dict:
    """Check if a domain is available via GoDaddy domains API."""
    api_key = os.getenv("GODADDY_API_KEY")
    api_secret = os.getenv("GODADDY_API_SECRET")
    if not api_key or not api_secret:
        return tool_error("GODADDY_API_KEY / GODADDY_API_SECRET are not set")

    return request_json(
        "GET",
        "https://api.godaddy.com/v1/domains/available",
        headers={"Authorization": f"sso-key {api_key}:{api_secret}"},
        params={"domain": domain, "checkType": check_type},
    )
