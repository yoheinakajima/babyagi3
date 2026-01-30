"""
Email Tools

Provides email capabilities via AgentMail - email infrastructure built for AI agents.

Features:
- Create dedicated inboxes for the agent
- Send emails
- Receive and read emails (for verification, notifications, etc.)
- Wait for specific emails (e.g., verification codes)

Install:
    pip install agentmail

Setup:
    Get API key from https://agentmail.to and set AGENTMAIL_API_KEY
"""

import os
import time
from tools import tool

# Cache the client and inbox
_client = None
_default_inbox = None
_default_inbox_id = None


def _get_client():
    """Get or create AgentMail client."""
    global _client

    if _client is not None:
        return _client

    try:
        from agentmail import AgentMail
    except ImportError:
        return None

    api_key = os.environ.get("AGENTMAIL_API_KEY")
    if not api_key:
        return None

    _client = AgentMail(api_key=api_key)
    return _client


def _get_inbox_id():
    """Get the inbox ID (email address) to use.

    Priority:
    1. AGENTMAIL_INBOX_ID environment variable (if set)
    2. First inbox from account
    3. Create a new inbox

    Returns the inbox_id (email address format like 'name@agentmail.to')
    """
    global _default_inbox_id, _default_inbox

    # Check if we already have a cached inbox ID
    if _default_inbox_id is not None:
        return _default_inbox_id

    # Check for configured inbox ID first
    configured_inbox = os.environ.get("AGENTMAIL_INBOX_ID")
    if configured_inbox:
        _default_inbox_id = configured_inbox
        return _default_inbox_id

    client = _get_client()
    if not client:
        return None

    try:
        # Try to get existing inboxes first
        inboxes_response = client.inboxes.list()
        # Handle both list response object and direct list
        inboxes = getattr(inboxes_response, 'inboxes', None) or inboxes_response

        if inboxes and len(inboxes) > 0:
            inbox = inboxes[0]
            _default_inbox = inbox
            # inbox_id is the email address (e.g., 'name@agentmail.to')
            _default_inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
            return _default_inbox_id

        # Create a new inbox if none exist
        inbox = client.inboxes.create()
        _default_inbox = inbox
        _default_inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
        return _default_inbox_id

    except Exception:
        return None


def _get_default_inbox():
    """Get or create the agent's default inbox object (for backwards compatibility)."""
    global _default_inbox

    if _default_inbox is not None:
        return _default_inbox

    # This will populate _default_inbox as a side effect
    _get_inbox_id()
    return _default_inbox


@tool(packages=["agentmail"], env=["AGENTMAIL_API_KEY"])
def get_agent_email() -> dict:
    """Get the agent's email address.

    Returns the agent's AgentMail email address that can be used for:
    - Service signups
    - Receiving verification emails
    - Newsletter subscriptions
    - Any email-based communication

    No arguments needed - returns the agent's primary email.
    """
    try:
        from agentmail import AgentMail
    except ImportError:
        return {
            "error": "agentmail not installed",
            "fix": "pip install agentmail"
        }

    if not os.environ.get("AGENTMAIL_API_KEY"):
        return {
            "error": "AGENTMAIL_API_KEY not set",
            "fix": "Get API key from https://agentmail.to and set AGENTMAIL_API_KEY environment variable"
        }

    inbox_id = _get_inbox_id()
    if not inbox_id:
        return {"error": "Failed to get or create inbox. Set AGENTMAIL_INBOX_ID environment variable with your inbox email address."}

    # inbox_id is the email address (e.g., 'name@agentmail.to')
    return {
        "email": inbox_id,
        "inbox_id": inbox_id
    }


@tool(packages=["agentmail"], env=["AGENTMAIL_API_KEY"])
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email from the agent's email address.

    Args:
        to: Recipient email address
        subject: Email subject line
        body: Email body content (plain text)
    """
    try:
        from agentmail import AgentMail
    except ImportError:
        return {
            "error": "agentmail not installed",
            "fix": "pip install agentmail"
        }

    client = _get_client()
    if not client:
        return {
            "error": "AGENTMAIL_API_KEY not set",
            "fix": "Get API key from https://agentmail.to"
        }

    inbox_id = _get_inbox_id()
    if not inbox_id:
        return {"error": "Failed to get inbox. Set AGENTMAIL_INBOX_ID environment variable with your inbox email address."}

    try:
        # Use client.inboxes.messages.send() with text parameter (not body)
        message = client.inboxes.messages.send(
            inbox_id=inbox_id,
            to=to,
            subject=subject,
            text=body
        )
        return {
            "sent": True,
            "to": to,
            "subject": subject,
            "message_id": getattr(message, 'message_id', None) or getattr(message, 'id', None)
        }
    except Exception as e:
        return {"error": str(e)}


@tool(packages=["agentmail"], env=["AGENTMAIL_API_KEY"])
def check_inbox(limit: int = 10, unread_only: bool = True) -> dict:
    """Check the agent's email inbox for new messages.

    Use this to:
    - Check for verification emails after signup
    - Read incoming messages
    - Find specific emails

    Args:
        limit: Maximum number of messages to return
        unread_only: Only return unread messages (default True)
    """
    try:
        from agentmail import AgentMail
    except ImportError:
        return {
            "error": "agentmail not installed",
            "fix": "pip install agentmail"
        }

    client = _get_client()
    if not client:
        return {
            "error": "AGENTMAIL_API_KEY not set",
            "fix": "Get API key from https://agentmail.to"
        }

    inbox_id = _get_inbox_id()
    if not inbox_id:
        return {"error": "Failed to get inbox. Set AGENTMAIL_INBOX_ID environment variable with your inbox email address."}

    try:
        # Use client.inboxes.messages.list()
        messages_response = client.inboxes.messages.list(inbox_id=inbox_id)
        # Handle both list response object and direct list
        messages = getattr(messages_response, 'messages', None) or messages_response or []

        result = []
        for msg in messages:
            msg_data = {
                "id": getattr(msg, 'message_id', None) or getattr(msg, 'id', None),
                "from": getattr(msg, 'from_', None) or getattr(msg, 'sender', None),
                "subject": getattr(msg, 'subject', None),
                "snippet": getattr(msg, 'snippet', None) or getattr(msg, 'preview', None),
                "received_at": getattr(msg, 'received_at', None) or getattr(msg, 'created_at', None),
                "read": getattr(msg, 'read', None)
            }

            # Filter unread if requested
            if unread_only and msg_data.get("read"):
                continue

            result.append(msg_data)

            if len(result) >= limit:
                break

        return {
            "count": len(result),
            "messages": result
        }
    except Exception as e:
        return {"error": str(e)}


@tool(packages=["agentmail"], env=["AGENTMAIL_API_KEY"])
def read_email(message_id: str) -> dict:
    """Read the full content of a specific email.

    Args:
        message_id: The ID of the message to read (from check_inbox)
    """
    try:
        from agentmail import AgentMail
    except ImportError:
        return {
            "error": "agentmail not installed",
            "fix": "pip install agentmail"
        }

    client = _get_client()
    if not client:
        return {
            "error": "AGENTMAIL_API_KEY not set"
        }

    inbox_id = _get_inbox_id()
    if not inbox_id:
        return {"error": "Failed to get inbox. Set AGENTMAIL_INBOX_ID environment variable with your inbox email address."}

    try:
        # Use client.inboxes.messages.get()
        message = client.inboxes.messages.get(inbox_id=inbox_id, message_id=message_id)

        return {
            "id": getattr(message, 'message_id', None) or getattr(message, 'id', None),
            "from": getattr(message, 'from_', None) or getattr(message, 'sender', None),
            "to": getattr(message, 'to', None),
            "subject": getattr(message, 'subject', None),
            "body": getattr(message, 'body', None) or getattr(message, 'text', None),
            "html": getattr(message, 'html', None),
            "received_at": getattr(message, 'received_at', None)
        }
    except Exception as e:
        return {"error": str(e)}


@tool(packages=["agentmail"], env=["AGENTMAIL_API_KEY"])
def wait_for_email(
    from_contains: str = None,
    subject_contains: str = None,
    timeout_seconds: int = 60
) -> dict:
    """Wait for a specific email to arrive.

    Useful for waiting for verification emails after signup.
    Polls the inbox until a matching email is found or timeout.

    Args:
        from_contains: Text that should appear in sender address
        subject_contains: Text that should appear in subject line
        timeout_seconds: How long to wait before giving up (default 60)
    """
    try:
        from agentmail import AgentMail
    except ImportError:
        return {
            "error": "agentmail not installed",
            "fix": "pip install agentmail"
        }

    client = _get_client()
    if not client:
        return {"error": "AGENTMAIL_API_KEY not set"}

    inbox_id = _get_inbox_id()
    if not inbox_id:
        return {"error": "Failed to get inbox. Set AGENTMAIL_INBOX_ID environment variable with your inbox email address."}

    start_time = time.time()
    poll_interval = 3  # seconds

    try:
        while time.time() - start_time < timeout_seconds:
            # Use client.inboxes.messages.list()
            messages_response = client.inboxes.messages.list(inbox_id=inbox_id)
            # Handle both list response object and direct list
            messages = getattr(messages_response, 'messages', None) or messages_response or []

            for msg in messages:
                sender = getattr(msg, 'from_', '') or getattr(msg, 'sender', '') or ''
                subject = getattr(msg, 'subject', '') or ''

                # Check filters
                if from_contains and from_contains.lower() not in sender.lower():
                    continue
                if subject_contains and subject_contains.lower() not in subject.lower():
                    continue

                # Found a match!
                return {
                    "found": True,
                    "message": {
                        "id": getattr(msg, 'message_id', None) or getattr(msg, 'id', None),
                        "from": sender,
                        "subject": subject,
                        "snippet": getattr(msg, 'snippet', None)
                    }
                }

            time.sleep(poll_interval)

        return {
            "found": False,
            "error": f"No matching email found within {timeout_seconds} seconds",
            "filters": {
                "from_contains": from_contains,
                "subject_contains": subject_contains
            }
        }

    except Exception as e:
        return {"error": str(e)}
