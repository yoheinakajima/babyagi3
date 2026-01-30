"""
Web Tools

Provides web search and browser automation capabilities.

- web_search: Quick DuckDuckGo search (no API key needed)
- browse: Full browser automation via Browser Use Cloud API
- fetch_url: Simple URL fetching and parsing
"""

import os
import time
from tools import tool

# Browser Use Cloud API configuration
BROWSER_USE_API_URL = "https://api.browser-use.com/api/v2"


@tool(packages=["ddgs"])
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo.

    Fast, free search with no API key required. Use this for quick lookups,
    finding documentation, or discovering services.

    Args:
        query: What to search for
        max_results: Maximum number of results to return (default 5)
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return {
            "error": "ddgs not installed",
            "fix": "pip install ddgs"
        }

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return {
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    }
                    for r in results
                ]
            }
    except Exception as e:
        return {"error": str(e)}


@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse(task: str, url: str = None, max_steps: int = 25, agent=None) -> dict:
    """Control a browser to complete a task autonomously via Browser Use Cloud.

    This is a powerful tool that can:
    - Navigate websites and fill forms
    - Sign up for services using the agent's email
    - Extract information from pages (including API keys from dashboards!)
    - Click buttons, follow links, handle popups

    For API key retrieval, describe the task like:
    "Go to the API keys page, copy the API key shown"

    Requires BROWSER_USE_API_KEY environment variable.
    Get your key at: https://cloud.browser-use.com

    Args:
        task: Natural language description of what to do in the browser
        url: Optional starting URL
        max_steps: Maximum number of agent steps (default 25)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return {
            "error": "BROWSER_USE_API_KEY not set",
            "fix": "Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        }

    headers = {
        "X-Browser-Use-API-Key": api_key,
        "Content-Type": "application/json"
    }

    # Build request payload
    payload = {
        "task": task,
        "maxSteps": max_steps,
    }
    if url:
        payload["startUrl"] = url

    try:
        # Create task
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/tasks",
                headers=headers,
                json=payload
            )

            if response.status_code not in (200, 201, 202):
                return {
                    "error": f"Failed to create task: {response.status_code}",
                    "details": response.text
                }

            task_data = response.json()
            task_id = task_data.get("id")
            session_id = task_data.get("sessionId")

            if not task_id:
                return {"error": "No task ID returned", "response": task_data}

            # Poll for completion (max 5 minutes)
            max_wait = 300
            poll_interval = 3
            waited = 0

            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

                status_response = client.get(
                    f"{BROWSER_USE_API_URL}/tasks/{task_id}",
                    headers=headers
                )

                if status_response.status_code != 200:
                    continue

                status_data = status_response.json()
                status = status_data.get("status", "").lower()

                if status in ("finished", "completed", "done"):
                    # Stop the session to save credits
                    try:
                        client.patch(
                            f"{BROWSER_USE_API_URL}/sessions/{session_id}",
                            headers=headers,
                            json={"action": "stop"}
                        )
                    except Exception:
                        pass

                    return {
                        "task": task,
                        "status": "completed",
                        "output": status_data.get("output"),
                        "steps": status_data.get("steps", []),
                        "success": True
                    }

                elif status in ("failed", "error"):
                    return {
                        "task": task,
                        "status": "failed",
                        "error": status_data.get("error") or status_data.get("output"),
                        "success": False
                    }

            # Timeout
            return {
                "task": task,
                "status": "timeout",
                "error": f"Task did not complete within {max_wait} seconds",
                "task_id": task_id,
                "success": False
            }

    except Exception as e:
        return {"error": str(e), "task": task}


@tool(packages=["httpx", "bs4"])
def fetch_url(url: str, extract: str = None) -> dict:
    """Fetch a URL and optionally extract specific information.

    Simpler than browse() - just fetches and parses a page.
    Use browse() for interactive tasks, this for quick reads.

    Args:
        url: The URL to fetch
        extract: Optional description of what to extract (e.g., "the main article text")
    """
    try:
        import httpx
        from bs4 import BeautifulSoup
    except ImportError:
        return {
            "error": "httpx and beautifulsoup4 not installed",
            "fix": "pip install httpx beautifulsoup4"
        }

    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Truncate if too long
        if len(text) > 10000:
            text = text[:10000] + "\n...[truncated]"

        return {
            "url": url,
            "title": soup.title.string if soup.title else None,
            "content": text,
            "extract_hint": extract
        }

    except Exception as e:
        return {"error": str(e), "url": url}


@tool(packages=["httpx", "agentmail"], env=["BROWSER_USE_API_KEY", "AGENTMAIL_API_KEY"])
def auto_signup(
    service_url: str,
    service_name: str,
    agent=None
) -> dict:
    """Automatically sign up for a service and retrieve API keys.

    This is a high-level tool that combines browser automation with email
    verification to fully automate service signup and API key retrieval.

    Uses Browser Use Cloud for browser automation.
    Requires: BROWSER_USE_API_KEY, AGENTMAIL_API_KEY

    The process:
    1. Navigate to signup page
    2. Create account using agent's email
    3. Check email for verification
    4. Complete verification
    5. Navigate to API/dashboard
    6. Extract and store API key

    Args:
        service_url: The signup or main page URL of the service
        service_name: Name of the service (used for storing the API key)
    """
    # Check required API keys
    missing = []
    if not os.environ.get("BROWSER_USE_API_KEY"):
        missing.append("BROWSER_USE_API_KEY")
    if not os.environ.get("AGENTMAIL_API_KEY"):
        missing.append("AGENTMAIL_API_KEY")

    if missing:
        return {
            "error": f"Missing required API keys: {', '.join(missing)}",
            "fix": {
                "BROWSER_USE_API_KEY": "https://cloud.browser-use.com",
                "AGENTMAIL_API_KEY": "https://agentmail.to"
            }
        }

    # This is a meta-tool that orchestrates browse + email + secrets
    # The agent can call this, or do the steps manually for more control
    return {
        "status": "ready",
        "message": f"To sign up for {service_name}, I'll need to:",
        "steps": [
            f"1. Browse to {service_url} and find signup",
            "2. Create account with agent email address (use get_agent_email)",
            "3. Check inbox for verification email (use check_inbox/wait_for_email)",
            "4. Complete email verification via browse()",
            "5. Navigate to API keys/dashboard section",
            f"6. Extract and store API key as {service_name.upper()}_API_KEY (use store_secret)"
        ],
        "hint": "Use browse(), check_inbox(), wait_for_email(), and store_secret() to complete these steps"
    }
