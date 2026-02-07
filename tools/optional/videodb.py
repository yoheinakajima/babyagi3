import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["VIDEODB_API_KEY"], packages=["httpx"])
def videodb_search_videos(query: str, limit: int = 10) -> dict:
    """Search indexed videos with VideoDB."""
    api_key = os.getenv("VIDEODB_API_KEY")
    if not api_key:
        return tool_error("VIDEODB_API_KEY is not set")

    base_url = os.getenv("VIDEODB_API_BASE", "https://api.videodb.io/v1")
    return request_json(
        "POST",
        f"{base_url}/videos/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={"query": query, "limit": limit},
    )
