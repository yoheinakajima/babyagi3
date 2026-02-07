import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["HUNTER_API_KEY"], packages=["httpx"])
def hunter_domain_search(domain: str, limit: int = 10) -> dict:
    """Search company email patterns and contacts with Hunter.io domain search."""
    api_key = os.getenv("HUNTER_API_KEY")
    if not api_key:
        return tool_error("HUNTER_API_KEY is not set")

    return request_json(
        "GET",
        "https://api.hunter.io/v2/domain-search",
        params={"domain": domain, "limit": limit, "api_key": api_key},
    )
