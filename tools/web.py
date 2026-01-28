"""
Web Tools

Provides web search and browser automation capabilities.

- web_search: Quick DuckDuckGo search (no API key needed)
- browse: Full browser automation via BrowserUse (can grab API keys!)

Install dependencies:
    pip install duckduckgo-search browser-use
"""

from tools import tool


@tool
def web_search(query: str, max_results: int = 5) -> dict:
    """Search the web using DuckDuckGo.

    Fast, free search with no API key required. Use this for quick lookups,
    finding documentation, or discovering services.

    Args:
        query: What to search for
        max_results: Maximum number of results to return (default 5)
    """
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return {
            "error": "duckduckgo-search not installed",
            "fix": "pip install duckduckgo-search"
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


@tool
def browse(task: str, url: str = None, agent=None) -> dict:
    """Control a browser to complete a task autonomously.

    This is a powerful tool that can:
    - Navigate websites and fill forms
    - Sign up for services using the agent's email
    - Extract information from pages (including API keys from dashboards!)
    - Click buttons, follow links, handle popups

    For API key retrieval, describe the task like:
    "Go to the API keys page, copy the API key shown"

    Args:
        task: Natural language description of what to do in the browser
        url: Optional starting URL (can also be part of the task description)
    """
    try:
        from browser_use import Agent as BrowserAgent, Browser, BrowserConfig
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return {
            "error": "browser-use not installed",
            "fix": "pip install browser-use langchain-anthropic"
        }

    import asyncio
    import os

    # Check for Anthropic API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"error": "ANTHROPIC_API_KEY environment variable required for browser automation"}

    try:
        # Build the full task
        full_task = task
        if url:
            full_task = f"Go to {url} and {task}"

        # Configure browser
        browser_config = BrowserConfig(
            headless=True,  # Run without GUI
        )

        # Use Claude for browser automation
        llm = ChatAnthropic(model="claude-sonnet-4-20250514")

        async def run_browser_task():
            browser = Browser(config=browser_config)
            browser_agent = BrowserAgent(
                task=full_task,
                llm=llm,
                browser=browser,
            )
            result = await browser_agent.run()
            await browser.close()
            return result

        # Run the async task
        result = asyncio.run(run_browser_task())

        return {
            "task": full_task,
            "result": str(result),
            "success": True
        }

    except Exception as e:
        return {
            "error": str(e),
            "task": task
        }


@tool
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


@tool
def auto_signup(
    service_url: str,
    service_name: str,
    agent=None
) -> dict:
    """Automatically sign up for a service and retrieve API keys.

    This is a high-level tool that combines browser automation with email
    verification to fully automate service signup and API key retrieval.

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
    # This is a meta-tool that orchestrates browse + email + secrets
    # The agent can call this, or do the steps manually for more control

    return {
        "status": "ready",
        "message": f"To sign up for {service_name}, I'll need to:",
        "steps": [
            f"1. Browse to {service_url} and find signup",
            "2. Create account with agent email address",
            "3. Check inbox for verification email",
            "4. Complete email verification",
            "5. Navigate to API keys/dashboard section",
            f"6. Extract and store API key as {service_name.upper()}_API_KEY"
        ],
        "hint": "Use browse(), check_inbox(), and store_secret() to complete these steps"
    }
