import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["SHOPIFY_ACCESS_TOKEN", "SHOPIFY_STORE_DOMAIN"], packages=["httpx"])
def shopify_graphql(query: str, variables: dict = None, api_version: str = "2024-10") -> dict:
    """Run a Shopify Admin GraphQL query against your store."""
    token = os.getenv("SHOPIFY_ACCESS_TOKEN")
    store = os.getenv("SHOPIFY_STORE_DOMAIN")
    if not token or not store:
        return tool_error("SHOPIFY_ACCESS_TOKEN / SHOPIFY_STORE_DOMAIN are not set")

    return request_json(
        "POST",
        f"https://{store}/admin/api/{api_version}/graphql.json",
        headers={"X-Shopify-Access-Token": token, "Content-Type": "application/json"},
        json_body={"query": query, "variables": variables or {}},
    )
