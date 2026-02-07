import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["EXA_API_KEY"], packages=["httpx"])
def exa_research_search(query: str, num_results: int = 5, use_websets: bool = False) -> dict:
    """Run semantic web research with Exa search (supports websets mode)."""
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        return tool_error("EXA_API_KEY is not set")

    payload = {
        "query": query,
        "numResults": num_results,
        "type": "webset" if use_websets else "neural",
    }
    return request_json(
        "POST",
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key, "Content-Type": "application/json"},
        json_body=payload,
    )
