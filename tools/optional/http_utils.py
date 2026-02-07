"""Shared HTTP helpers for optional API tools."""

from __future__ import annotations

from typing import Any

from tools import tool_error


def request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict:
    """Send an HTTP request and normalize response payloads as dicts."""
    try:
        import httpx
    except ImportError:
        return tool_error("httpx not installed", fix="pip install httpx")

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                json=json_body,
            )
    except Exception as exc:
        return tool_error(f"Request failed: {exc}")

    content_type = response.headers.get("content-type", "")
    parsed: Any
    if "application/json" in content_type:
        try:
            parsed = response.json()
        except Exception:
            parsed = {"raw": response.text}
    else:
        parsed = {"raw": response.text}

    if response.status_code >= 400:
        return tool_error(
            f"Request failed with status {response.status_code}",
            details=parsed,
            endpoint=url,
        )

    return {
        "status_code": response.status_code,
        "data": parsed,
    }
