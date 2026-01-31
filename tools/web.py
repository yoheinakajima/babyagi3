"""
Web Tools

Provides web search and browser automation capabilities.

- web_search: Quick DuckDuckGo search (no API key needed)
- browse: Full browser automation via Browser Use Cloud API
- browse_with_credentials: Secure credential injection for browser tasks
- browse_checkout: Secure payment form filling (credit card never exposed to AI)
- browse_session: Create reusable browser sessions
- browse_profile: Manage persistent browser profiles
- browse_control: Pause/resume/stop running tasks
- fetch_url: Simple URL fetching and parsing

SECURITY: Payment credentials are NEVER exposed to the AI. They flow directly
from the system keyring to the Browser Use API via encrypted channels.
"""

import os
import re
import sys
import time
from tools import tool, tool_error

# Browser Use Cloud API configuration
BROWSER_USE_API_URL = "https://api.browser-use.com/api/v1"

# Available LLM models for browser automation
BROWSER_USE_MODELS = {
    "default": None,  # Use API default
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "gemini-2.5-flash": "gemini-2.5-flash",
}


def _get_headers() -> dict:
    """Get API headers with authentication."""
    api_key = os.environ.get("BROWSER_USE_API_KEY")
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }


def _show_live_url(url: str, task: str):
    """Display live browser URL if running interactively."""
    if sys.stderr.isatty():
        # Truncate task for display
        short_task = task[:50] + "..." if len(task) > 50 else task
        print(f"\033[36m🌐 Browser session: {url}\033[0m", file=sys.stderr, flush=True)
        print(f"\033[36m   Task: {short_task}\033[0m", file=sys.stderr, flush=True)


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
        return tool_error("ddgs not installed", fix="pip install ddgs")

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
def browse(
    task: str,
    url: str = None,
    max_steps: int = 25,
    model: str = None,
    output_schema: dict = None,
    session_id: str = None,
    save_session: bool = False,
    agent=None
) -> dict:
    """Control a browser to complete a task autonomously via Browser Use Cloud.

    This is a powerful tool that can:
    - Navigate websites and fill forms
    - Sign up for services using the agent's email
    - Extract information from pages (including API keys from dashboards!)
    - Click buttons, follow links, handle popups
    - Return structured data matching a schema

    For API key retrieval, describe the task like:
    "Go to the API keys page, copy the API key shown"

    Requires BROWSER_USE_API_KEY environment variable.
    Get your key at: https://cloud.browser-use.com

    Args:
        task: Natural language description of what to do in the browser
        url: Optional starting URL
        max_steps: Maximum number of agent steps (default 25)
        model: LLM model to use (gpt-4.1, gpt-4.1-mini, claude-sonnet, gemini-2.5-flash)
        output_schema: JSON schema for structured output extraction
        session_id: Reuse an existing browser session (preserves cookies/state)
        save_session: Keep session alive after task for reuse (returns session_id)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    headers = _get_headers()

    # Build request payload
    payload = {
        "task": task,
        "maxSteps": max_steps,
    }
    if url:
        payload["startUrl"] = url

    # Add model selection
    if model:
        model_id = BROWSER_USE_MODELS.get(model, model)
        if model_id:
            payload["model"] = model_id

    # Add structured output schema
    if output_schema:
        payload["outputSchema"] = output_schema

    # Add session reuse
    if session_id:
        payload["sessionId"] = session_id

    # Configure session persistence
    if save_session:
        payload["keepSessionAlive"] = True

    try:
        # Create task
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/run-task",
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
            live_url = task_data.get("liveUrl")
            returned_session_id = task_data.get("sessionId") or session_id

            if not task_id:
                return {"error": "No task ID returned", "response": task_data}

            # Show live URL for interactive debugging
            if live_url:
                _show_live_url(live_url, task)

            # Poll for completion (max 5 minutes)
            max_wait = 300
            poll_interval = 3
            waited = 0

            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

                status_response = client.get(
                    f"{BROWSER_USE_API_URL}/task/{task_id}",
                    headers=headers
                )

                if status_response.status_code != 200:
                    continue

                status_data = status_response.json()
                status = status_data.get("status", "").lower()

                if status in ("finished", "completed", "done"):
                    # Stop the session to save credits (unless saving for reuse)
                    if not save_session and returned_session_id:
                        try:
                            client.post(
                                f"{BROWSER_USE_API_URL}/session/{returned_session_id}/stop",
                                headers=headers
                            )
                        except Exception:
                            pass

                    result = {
                        "task": task,
                        "status": "completed",
                        "output": status_data.get("output"),
                        "steps": status_data.get("steps", []),
                        "task_id": task_id,
                        "success": True
                    }

                    # Include session ID for reuse if saved
                    if save_session and returned_session_id:
                        result["session_id"] = returned_session_id

                    return result

                elif status in ("failed", "error"):
                    return {
                        "task": task,
                        "status": "failed",
                        "error": status_data.get("error") or status_data.get("output"),
                        "task_id": task_id,
                        "success": False
                    }

            # Timeout
            return {
                "task": task,
                "status": "timeout",
                "error": f"Task did not complete within {max_wait} seconds",
                "task_id": task_id,
                "session_id": returned_session_id,
                "success": False
            }

    except Exception as e:
        return {"error": str(e), "task": task}


# =============================================================================
# Secure Credential Injection for Browser Tasks
# =============================================================================

def _get_credential_secrets(service: str, credential_type: str = None, agent=None) -> tuple[dict, dict]:
    """
    Retrieve credentials from keyring and format for Browser Use secrets API.

    SECURITY: This function retrieves the actual credential values but they are
    NEVER returned to the AI. They go directly to Browser Use's encrypted secrets.

    Returns:
        (secrets_dict, metadata_dict) where:
        - secrets_dict: Actual values for Browser Use API (NEVER shown to AI)
        - metadata_dict: Safe metadata for AI (masked values, types, etc.)
    """
    try:
        import keyring
        from tools.secrets import KEYRING_SERVICE
        from tools.credentials import _generate_secret_ref
    except ImportError:
        return {}, {"error": "keyring not available"}

    secrets = {}
    metadata = {"service": service, "credential_type": credential_type}

    # Try to get from memory store first
    cred = None
    if agent and hasattr(agent, 'memory') and agent.memory is not None:
        try:
            cred = agent.memory.store.get_credential(service, credential_type)
        except Exception:
            pass

    if cred:
        metadata["found"] = True
        metadata["credential_type"] = cred.credential_type

        if cred.credential_type == "account":
            # Account credentials
            if cred.username:
                secrets["x_username"] = cred.username
                metadata["username"] = cred.username
            if cred.email:
                secrets["x_email"] = cred.email
                metadata["email"] = cred.email
            if cred.password_ref:
                password = keyring.get_password(KEYRING_SERVICE, cred.password_ref)
                if password:
                    secrets["x_password"] = password
                    metadata["has_password"] = True

        elif cred.credential_type == "credit_card":
            # Credit card credentials - NEVER expose full number
            if cred.card_ref:
                card_number = keyring.get_password(KEYRING_SERVICE, cred.card_ref)
                if card_number:
                    secrets["x_card_number"] = card_number
                    # Only show masked version to AI
                    metadata["card_masked"] = f"****{cred.card_last_four}"
                    metadata["card_type"] = cred.card_type

            if cred.card_expiry:
                secrets["x_card_expiry"] = cred.card_expiry
                metadata["card_expiry"] = cred.card_expiry

            # Get CVV from keyring
            cvv_ref = _generate_secret_ref(service, "credit_card", "cvv")
            cvv = keyring.get_password(KEYRING_SERVICE, cvv_ref)
            if cvv:
                secrets["x_card_cvv"] = cvv
                metadata["has_cvv"] = True

            if cred.billing_name:
                secrets["x_billing_name"] = cred.billing_name
                metadata["billing_name"] = cred.billing_name

            if cred.billing_address:
                secrets["x_billing_address"] = cred.billing_address
                metadata["has_billing_address"] = True

        # Update last used
        if agent and hasattr(agent, 'memory') and agent.memory is not None:
            try:
                agent.memory.store.update_credential_last_used(cred.id)
            except Exception:
                pass
    else:
        # Fallback: try direct keyring lookup
        password_ref = _generate_secret_ref(service, credential_type or "account", "password")
        password = keyring.get_password(KEYRING_SERVICE, password_ref)
        if password:
            secrets["x_password"] = password
            metadata["found"] = True
            metadata["has_password"] = True

        card_ref = _generate_secret_ref(service, "credit_card", "card")
        card = keyring.get_password(KEYRING_SERVICE, card_ref)
        if card:
            secrets["x_card_number"] = card
            metadata["found"] = True
            clean_num = re.sub(r'[\s-]', '', card)
            metadata["card_masked"] = f"****{clean_num[-4:]}"

    if not secrets:
        metadata["found"] = False
        metadata["error"] = f"No credentials found for '{service}'"

    return secrets, metadata


@tool(packages=["httpx", "keyring"], env=["BROWSER_USE_API_KEY"])
def browse_with_credentials(
    task: str,
    credential_service: str,
    credential_type: str = None,
    url: str = None,
    allowed_domains: list = None,
    max_steps: int = 25,
    model: str = None,
    session_id: str = None,
    save_session: bool = False,
    agent=None
) -> dict:
    """
    Browse with secure credential injection - credentials never visible to AI.

    Use this when you need to log into a site or fill forms with stored credentials.
    The actual passwords/card numbers are NEVER shown in messages - they go directly
    to the browser via encrypted channels.

    In your task description, use these placeholders that will be replaced:
    - x_username: The stored username
    - x_email: The stored email
    - x_password: The stored password
    - x_card_number: The full card number (you'll never see this)
    - x_card_expiry: Card expiry date
    - x_card_cvv: Card CVV (you'll never see this)
    - x_billing_name: Name on card

    Example task: "Login with username x_username and password x_password"

    SECURITY: For payment forms, use browse_checkout() instead which has
    additional safety measures.

    Args:
        task: Browser task with placeholders (x_username, x_password, etc.)
        credential_service: Service name to look up credentials for
        credential_type: "account" or "credit_card" (auto-detected if not specified)
        url: Starting URL
        allowed_domains: Lock browser to these domains only (REQUIRED for credit cards)
        max_steps: Maximum browser steps
        model: LLM model for browser automation
        session_id: Reuse existing session
        save_session: Keep session for reuse
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com"
        )

    # Get credentials securely (actual values never returned to AI)
    secrets, metadata = _get_credential_secrets(credential_service, credential_type, agent)

    if not secrets:
        return {
            "error": f"No credentials found for '{credential_service}'",
            "hint": f"First store credentials with: store_credential(service='{credential_service}', ...)"
        }

    # SECURITY: Require allowed_domains for credit card credentials
    if metadata.get("card_masked") and not allowed_domains:
        return tool_error(
            "Credit card credentials require allowed_domains for security",
            fix="Specify allowed_domains=['checkout.example.com'] to restrict where card data can be used"
        )

    headers = _get_headers()

    # Build request payload with secrets
    payload = {
        "task": task,
        "maxSteps": max_steps,
        "secrets": secrets,  # Encrypted by Browser Use API
    }

    if url:
        payload["startUrl"] = url

    if allowed_domains:
        payload["allowedDomains"] = allowed_domains

    if model:
        model_id = BROWSER_USE_MODELS.get(model, model)
        if model_id:
            payload["model"] = model_id

    if session_id:
        payload["sessionId"] = session_id

    if save_session:
        payload["keepSessionAlive"] = True

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/run-task",
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
            live_url = task_data.get("liveUrl")
            returned_session_id = task_data.get("sessionId") or session_id

            if not task_id:
                return {"error": "No task ID returned", "response": task_data}

            if live_url:
                _show_live_url(live_url, f"[SECURE] {task[:30]}...")

            # Poll for completion
            max_wait = 300
            poll_interval = 3
            waited = 0

            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

                status_response = client.get(
                    f"{BROWSER_USE_API_URL}/task/{task_id}",
                    headers=headers
                )

                if status_response.status_code != 200:
                    continue

                status_data = status_response.json()
                status = status_data.get("status", "").lower()

                if status in ("finished", "completed", "done"):
                    if not save_session and returned_session_id:
                        try:
                            client.post(
                                f"{BROWSER_USE_API_URL}/session/{returned_session_id}/stop",
                                headers=headers
                            )
                        except Exception:
                            pass

                    # Return result with SAFE metadata only (no actual credentials)
                    result = {
                        "task": task,
                        "status": "completed",
                        "output": status_data.get("output"),
                        "steps": status_data.get("steps", []),
                        "task_id": task_id,
                        "success": True,
                        "credential_used": metadata,  # Safe metadata only
                    }

                    if save_session and returned_session_id:
                        result["session_id"] = returned_session_id

                    return result

                elif status in ("failed", "error"):
                    return {
                        "task": task,
                        "status": "failed",
                        "error": status_data.get("error") or status_data.get("output"),
                        "task_id": task_id,
                        "success": False,
                        "credential_used": metadata,
                    }

            return {
                "task": task,
                "status": "timeout",
                "error": f"Task did not complete within {max_wait} seconds",
                "task_id": task_id,
                "session_id": returned_session_id,
                "success": False
            }

    except Exception as e:
        return {"error": str(e), "task": task}


@tool(packages=["httpx", "keyring"], env=["BROWSER_USE_API_KEY"])
def browse_checkout(
    checkout_url: str,
    card_service: str = "payment",
    task: str = None,
    allowed_domains: list = None,
    max_steps: int = 20,
    model: str = None,
    session_id: str = None,
    agent=None
) -> dict:
    """
    Complete a checkout with stored credit card - card numbers NEVER visible to AI.

    This is the SECURE way to make payments. The credit card number and CVV
    are retrieved from the system keyring and sent directly to the browser
    via Browser Use's encrypted secrets API. The AI never sees the full card number.

    SECURITY FEATURES:
    - Full card number and CVV are NEVER in AI context or logs
    - Browser is locked to allowed_domains only
    - Card data goes directly from keyring to Browser Use encryption
    - Only masked card info (****1234) is returned to AI

    Before using, store your card:
        store_credential(
            service="payment",
            credential_type="credit_card",
            card_number="4111111111111111",
            card_expiry="12/25",
            card_cvv="123",
            billing_name="John Doe"
        )

    Args:
        checkout_url: The checkout page URL
        card_service: Service name where card is stored (default: "payment")
        task: Optional custom task (default: fill payment form and complete checkout)
        allowed_domains: Domains where card can be used (REQUIRED - extracted from URL if not provided)
        max_steps: Maximum steps for checkout flow
        model: LLM model to use
        session_id: Existing session to continue

    Returns:
        Checkout result with order confirmation (card details NEVER included)
    """
    import httpx
    from urllib.parse import urlparse

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com"
        )

    # Get card credentials securely
    secrets, metadata = _get_credential_secrets(card_service, "credit_card", agent)

    if not secrets or "x_card_number" not in secrets:
        return tool_error(
            f"No credit card found for service '{card_service}'",
            fix=f"Store card first: store_credential(service='{card_service}', credential_type='credit_card', card_number='...', card_expiry='MM/YY', card_cvv='...')"
        )

    # Extract domain from checkout URL for security
    parsed_url = urlparse(checkout_url)
    checkout_domain = parsed_url.netloc

    if not allowed_domains:
        # Auto-restrict to checkout domain
        allowed_domains = [checkout_domain]

    # Verify checkout URL is in allowed domains
    domain_allowed = False
    for domain in allowed_domains:
        if domain == checkout_domain or checkout_domain.endswith(f".{domain}"):
            domain_allowed = True
            break

    if not domain_allowed:
        return tool_error(
            f"Checkout URL domain '{checkout_domain}' not in allowed_domains",
            fix=f"Add '{checkout_domain}' to allowed_domains or check the URL"
        )

    # Build the checkout task
    if not task:
        task = """
        Complete the checkout payment form:
        1. Look for credit card / payment form fields
        2. Enter card number: x_card_number
        3. Enter expiry date: x_card_expiry (format may need adjustment for MM/YY or MM/YYYY)
        4. Enter CVV/security code: x_card_cvv
        5. Enter cardholder name: x_billing_name (if there's a name field)
        6. If there's a billing address field, enter: x_billing_address
        7. Click the pay/submit/complete button
        8. Wait for confirmation and report the order number or confirmation message
        """

    headers = _get_headers()

    payload = {
        "task": task,
        "startUrl": checkout_url,
        "maxSteps": max_steps,
        "secrets": secrets,
        "allowedDomains": allowed_domains,
        "useVision": False,  # Disable vision to prevent card number in screenshots
    }

    if model:
        model_id = BROWSER_USE_MODELS.get(model, model)
        if model_id:
            payload["model"] = model_id

    if session_id:
        payload["sessionId"] = session_id

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/run-task",
                headers=headers,
                json=payload
            )

            if response.status_code not in (200, 201, 202):
                return {
                    "error": f"Failed to start checkout: {response.status_code}",
                    "details": response.text
                }

            task_data = response.json()
            task_id = task_data.get("id")
            live_url = task_data.get("liveUrl")
            returned_session_id = task_data.get("sessionId") or session_id

            if not task_id:
                return {"error": "No task ID returned"}

            if live_url:
                _show_live_url(live_url, f"[CHECKOUT] {checkout_domain}")

            # Poll for completion (checkout may need more time)
            max_wait = 300
            poll_interval = 3
            waited = 0

            while waited < max_wait:
                time.sleep(poll_interval)
                waited += poll_interval

                status_response = client.get(
                    f"{BROWSER_USE_API_URL}/task/{task_id}",
                    headers=headers
                )

                if status_response.status_code != 200:
                    continue

                status_data = status_response.json()
                status = status_data.get("status", "").lower()

                if status in ("finished", "completed", "done"):
                    # Stop session to ensure card data isn't persisted
                    if returned_session_id:
                        try:
                            client.post(
                                f"{BROWSER_USE_API_URL}/session/{returned_session_id}/stop",
                                headers=headers
                            )
                        except Exception:
                            pass

                    # Return SAFE result - no card details
                    return {
                        "checkout_url": checkout_url,
                        "status": "completed",
                        "output": status_data.get("output"),
                        "card_used": metadata.get("card_masked", "****"),  # Only masked
                        "card_type": metadata.get("card_type"),
                        "task_id": task_id,
                        "success": True,
                        "message": "Checkout completed. Card details were never exposed."
                    }

                elif status in ("failed", "error"):
                    # Stop session on failure too
                    if returned_session_id:
                        try:
                            client.post(
                                f"{BROWSER_USE_API_URL}/session/{returned_session_id}/stop",
                                headers=headers
                            )
                        except Exception:
                            pass

                    return {
                        "checkout_url": checkout_url,
                        "status": "failed",
                        "error": status_data.get("error") or status_data.get("output"),
                        "card_used": metadata.get("card_masked"),
                        "task_id": task_id,
                        "success": False
                    }

            return {
                "checkout_url": checkout_url,
                "status": "timeout",
                "error": f"Checkout did not complete within {max_wait} seconds",
                "task_id": task_id,
                "success": False
            }

    except Exception as e:
        return {"error": str(e), "checkout_url": checkout_url}


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
        return tool_error("httpx and beautifulsoup4 not installed", fix="pip install httpx beautifulsoup4")

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


# =============================================================================
# Browser Session & Control Tools
# =============================================================================

@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse_control(task_id: str, action: str) -> dict:
    """Control a running browser task - pause, resume, or stop.

    Use this to intervene in long-running tasks. Useful when:
    - You need to manually handle a CAPTCHA (pause, solve, resume)
    - A task is stuck and needs to be stopped
    - You want to take over control temporarily

    Args:
        task_id: The task ID returned from browse()
        action: Control action - "pause", "resume", or "stop"
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    if action not in ("pause", "resume", "stop"):
        return tool_error(
            f"Invalid action: {action}",
            fix="Use one of: pause, resume, stop"
        )

    headers = _get_headers()

    try:
        with httpx.Client(timeout=30) as client:
            response = client.post(
                f"{BROWSER_USE_API_URL}/task/{task_id}/{action}",
                headers=headers
            )

            if response.status_code not in (200, 201, 202):
                return {
                    "error": f"Failed to {action} task: {response.status_code}",
                    "details": response.text
                }

            return {
                "task_id": task_id,
                "action": action,
                "status": "success",
                "message": f"Task {action}d successfully"
            }

    except Exception as e:
        return {"error": str(e), "task_id": task_id}


@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse_session(action: str, session_id: str = None, profile_id: str = None) -> dict:
    """Manage browser sessions for multi-task workflows.

    Sessions preserve browser state (cookies, localStorage, auth) between tasks.
    Create a session, run multiple browse() tasks with that session_id, then stop.

    Workflow example:
    1. create session -> get session_id
    2. browse(task="login", session_id=session_id, save_session=True)
    3. browse(task="do something", session_id=session_id, save_session=True)
    4. browse_session(action="stop", session_id=session_id)

    Args:
        action: Session action - "create" or "stop"
        session_id: Required for "stop" action
        profile_id: Optional profile ID to use with "create" (for persistent auth)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    if action not in ("create", "stop"):
        return tool_error(
            f"Invalid action: {action}",
            fix="Use one of: create, stop"
        )

    headers = _get_headers()

    try:
        with httpx.Client(timeout=30) as client:
            if action == "create":
                payload = {}
                if profile_id:
                    payload["profileId"] = profile_id

                response = client.post(
                    f"{BROWSER_USE_API_URL}/sessions",
                    headers=headers,
                    json=payload if payload else None
                )

                if response.status_code not in (200, 201, 202):
                    return {
                        "error": f"Failed to create session: {response.status_code}",
                        "details": response.text
                    }

                data = response.json()
                live_url = data.get("liveUrl")

                # Show live URL for interactive debugging
                if live_url:
                    _show_live_url(live_url, "New browser session")

                return {
                    "action": "create",
                    "session_id": data.get("id"),
                    "live_url": live_url,
                    "status": "created",
                    "message": "Session created. Use session_id in browse() calls."
                }

            elif action == "stop":
                if not session_id:
                    return tool_error(
                        "session_id required for stop action",
                        fix="Provide the session_id to stop"
                    )

                response = client.post(
                    f"{BROWSER_USE_API_URL}/session/{session_id}/stop",
                    headers=headers
                )

                if response.status_code not in (200, 201, 202, 204):
                    return {
                        "error": f"Failed to stop session: {response.status_code}",
                        "details": response.text
                    }

                return {
                    "action": "stop",
                    "session_id": session_id,
                    "status": "stopped",
                    "message": "Session stopped. Browser resources released."
                }

    except Exception as e:
        return {"error": str(e)}


@tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
def browse_profile(action: str, name: str = None, profile_id: str = None) -> dict:
    """Manage browser profiles for persistent authentication.

    Profiles save browser fingerprints, cookies, and auth state permanently.
    Unlike sessions (temporary), profiles persist across restarts.

    Use profiles when:
    - You need to stay logged into services long-term
    - You want consistent browser identity (for bot detection)
    - You're automating workflows that require authentication

    Workflow:
    1. Create profile: browse_profile(action="create", name="my-service")
    2. Login once: browse(task="login to service", save_session=True)
    3. Future tasks: browse_session(action="create", profile_id=profile_id)
       then browse() with that session_id - already logged in!

    Args:
        action: Profile action - "create", "list", or "delete"
        name: Profile name (required for create)
        profile_id: Profile ID (required for delete)
    """
    import httpx

    api_key = os.environ.get("BROWSER_USE_API_KEY")
    if not api_key:
        return tool_error(
            "BROWSER_USE_API_KEY not set",
            fix="Get your API key at https://cloud.browser-use.com and set BROWSER_USE_API_KEY"
        )

    if action not in ("create", "list", "delete"):
        return tool_error(
            f"Invalid action: {action}",
            fix="Use one of: create, list, delete"
        )

    headers = _get_headers()

    try:
        with httpx.Client(timeout=30) as client:
            if action == "create":
                if not name:
                    return tool_error(
                        "name required for create action",
                        fix="Provide a name for the profile"
                    )

                response = client.post(
                    f"{BROWSER_USE_API_URL}/profiles",
                    headers=headers,
                    json={"name": name}
                )

                if response.status_code not in (200, 201, 202):
                    return {
                        "error": f"Failed to create profile: {response.status_code}",
                        "details": response.text
                    }

                data = response.json()
                return {
                    "action": "create",
                    "profile_id": data.get("id"),
                    "name": name,
                    "status": "created",
                    "message": "Profile created. Use profile_id with browse_session(action='create')."
                }

            elif action == "list":
                response = client.get(
                    f"{BROWSER_USE_API_URL}/profiles",
                    headers=headers
                )

                if response.status_code != 200:
                    return {
                        "error": f"Failed to list profiles: {response.status_code}",
                        "details": response.text
                    }

                data = response.json()
                profiles = data.get("profiles", data) if isinstance(data, dict) else data
                return {
                    "action": "list",
                    "profiles": [
                        {"id": p.get("id"), "name": p.get("name")}
                        for p in (profiles if isinstance(profiles, list) else [])
                    ],
                    "count": len(profiles) if isinstance(profiles, list) else 0
                }

            elif action == "delete":
                if not profile_id:
                    return tool_error(
                        "profile_id required for delete action",
                        fix="Provide the profile_id to delete"
                    )

                response = client.delete(
                    f"{BROWSER_USE_API_URL}/profiles/{profile_id}",
                    headers=headers
                )

                if response.status_code not in (200, 201, 202, 204):
                    return {
                        "error": f"Failed to delete profile: {response.status_code}",
                        "details": response.text
                    }

                return {
                    "action": "delete",
                    "profile_id": profile_id,
                    "status": "deleted",
                    "message": "Profile deleted."
                }

    except Exception as e:
        return {"error": str(e)}
