import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["X_API_BEARER_TOKEN"], packages=["httpx"])
def x_recent_search(query: str, max_results: int = 10) -> dict:
    """Search recent posts on X (Twitter) via v2 recent search."""
    token = os.getenv("X_API_BEARER_TOKEN")
    if not token:
        return tool_error("X_API_BEARER_TOKEN is not set")

    return request_json(
        "GET",
        "https://api.x.com/2/tweets/search/recent",
        headers={"Authorization": f"Bearer {token}"},
        params={"query": query, "max_results": max_results},
    )
