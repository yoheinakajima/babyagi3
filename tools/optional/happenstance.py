import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["HAPPENSTANCE_API_KEY"], packages=["httpx"])
def happenstance_search(query: str, limit: int = 5) -> dict:
    """Search Happenstance records using its REST search endpoint."""
    api_key = os.getenv("HAPPENSTANCE_API_KEY")
    if not api_key:
        return tool_error("HAPPENSTANCE_API_KEY is not set")

    base_url = os.getenv("HAPPENSTANCE_API_BASE", "https://api.happenstance.ai/v1")
    return request_json(
        "POST",
        f"{base_url}/search",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={"query": query, "limit": limit},
    )
