import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["PEOPLEDATALABS_API_KEY"], packages=["httpx"])
def pdl_person_enrich(linkedin_url: str = None, email: str = None, full_name: str = None) -> dict:
    """Enrich a person profile via People Data Labs Person Enrichment API."""
    api_key = os.getenv("PEOPLEDATALABS_API_KEY")
    if not api_key:
        return tool_error("PEOPLEDATALABS_API_KEY is not set")
    if not any([linkedin_url, email, full_name]):
        return tool_error("Provide at least one lookup field (linkedin_url, email, or full_name)")

    payload = {"profile": linkedin_url, "email": email, "name": full_name}
    payload = {k: v for k, v in payload.items() if v}
    return request_json(
        "POST",
        "https://api.peopledatalabs.com/v5/person/enrich",
        headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
        json_body=payload,
    )
