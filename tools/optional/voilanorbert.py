import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["VOILANORBERT_API_KEY"], packages=["httpx"])
def voilanorbert_search_name(first_name: str, last_name: str, domain: str) -> dict:
    """Find an email address using VoilaNorbert's name search endpoint."""
    api_key = os.getenv("VOILANORBERT_API_KEY")
    if not api_key:
        return tool_error("VOILANORBERT_API_KEY is not set")

    return request_json(
        "POST",
        "https://api.voilanorbert.com/2018-01-08/search/name",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={"name": f"{first_name} {last_name}", "domain": domain},
    )
