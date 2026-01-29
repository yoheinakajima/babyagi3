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


def _get_default_inbox():
    """Get or create the agent's default inbox."""
    global _default_inbox

    if _default_inbox is not None:
        return _default_inbox

    client = _get_client()
    if not client:
        return None

    try:
        # Try to get existing inboxes first
        inboxes = client.inboxes.list()
        if inboxes and len(inboxes) > 0:
            _default_inbox = inboxes[0]
            return _default_inbox

        # Create a new inbox
        _default_inbox = client.inboxes.create()
        return _default_inbox

    except Exception:
        return None


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

    inbox = _get_default_inbox()
    if not inbox:
        return {"error": "Failed to get or create inbox"}

    email_address = getattr(inbox, 'email', None) or getattr(inbox, 'address', None)
    if not email_address:
        # Try to construct from username and domain
        username = getattr(inbox, 'username', None)
        domain = getattr(inbox, 'domain', 'agentmail.to')
        if username:
            email_address = f"{username}@{domain}"

    return {
        "email": email_address,
        "inbox_id": getattr(inbox, 'id', None)
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

    inbox = _get_default_inbox()
    if not inbox:
        return {"error": "Failed to get inbox"}

    try:
        inbox_id = getattr(inbox, 'id', inbox)
        message = client.messages.send(
            inbox_id=inbox_id,
            to=to,
            subject=subject,
            body=body
        )
        return {
            "sent": True,
            "to": to,
            "subject": subject,
            "message_id": getattr(message, 'id', None)
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

    inbox = _get_default_inbox()
    if not inbox:
        return {"error": "Failed to get inbox"}

    try:
        inbox_id = getattr(inbox, 'id', inbox)
        messages = client.messages.list(inbox_id=inbox_id, limit=limit)

        result = []
        for msg in messages:
            msg_data = {
                "id": getattr(msg, 'id', None),
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

        return {
            "count": len(result),
            "messages": result[:limit]
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

    inbox = _get_default_inbox()
    if not inbox:
        return {"error": "Failed to get inbox"}

    try:
        inbox_id = getattr(inbox, 'id', inbox)
        message = client.messages.get(inbox_id=inbox_id, message_id=message_id)

        return {
            "id": getattr(message, 'id', None),
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

    inbox = _get_default_inbox()
    if not inbox:
        return {"error": "Failed to get inbox"}

    start_time = time.time()
    poll_interval = 3  # seconds

    try:
        inbox_id = getattr(inbox, 'id', inbox)

        while time.time() - start_time < timeout_seconds:
            messages = client.messages.list(inbox_id=inbox_id, limit=20)

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
                        "id": getattr(msg, 'id', None),
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
