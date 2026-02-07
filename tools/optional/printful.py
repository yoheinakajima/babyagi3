import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["PRINTFUL_API_KEY"], packages=["httpx"])
def printful_list_products(limit: int = 20, offset: int = 0) -> dict:
    """List products from your Printful store."""
    api_key = os.getenv("PRINTFUL_API_KEY")
    if not api_key:
        return tool_error("PRINTFUL_API_KEY is not set")

    return request_json(
        "GET",
        "https://api.printful.com/store/products",
        headers={"Authorization": f"Bearer {api_key}"},
        params={"limit": limit, "offset": offset},
    )
