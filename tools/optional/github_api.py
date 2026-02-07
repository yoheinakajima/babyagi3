import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["GITHUB_TOKEN"], packages=["httpx"])
def github_search_repositories(query: str, sort: str = "stars", per_page: int = 10) -> dict:
    """Search GitHub repositories with authenticated API access."""
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        return tool_error("GITHUB_TOKEN is not set")

    return request_json(
        "GET",
        "https://api.github.com/search/repositories",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        params={"q": query, "sort": sort, "per_page": per_page},
    )
