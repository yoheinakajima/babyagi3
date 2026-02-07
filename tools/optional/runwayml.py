import os

from tools import tool, tool_error
from tools.optional.http_utils import request_json


@tool(env=["RUNWAY_API_KEY"], packages=["httpx"])
def runway_create_task(task_type: str, prompt_text: str, model: str = "gen4_turbo") -> dict:
    """Create a Runway task (image/video generation) using prompt text."""
    api_key = os.getenv("RUNWAY_API_KEY")
    if not api_key:
        return tool_error("RUNWAY_API_KEY is not set")

    return request_json(
        "POST",
        "https://api.dev.runwayml.com/v1/tasks",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": "2024-11-06",
        },
        json_body={
            "taskType": task_type,
            "internal": False,
            "options": {"promptText": prompt_text, "model": model},
        },
    )
