"""
Shared AgentMail Client

Provides a singleton pattern for AgentMail client and inbox management.
Used by both tools/email.py and listeners/email.py to avoid duplication.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Cache the client and inbox
_client = None
_default_inbox = None
_default_inbox_id = None


def get_client():
    """Get or create AgentMail client singleton.

    Returns:
        AgentMail client instance, or None if unavailable.
    """
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


def get_inbox_id() -> str | None:
    """Get the inbox ID (email address) to use.

    Priority:
    1. AGENTMAIL_INBOX_ID environment variable (if set)
    2. First inbox from account
    3. Create a new inbox

    Returns:
        The inbox_id (email address format like 'name@agentmail.to'),
        or None if unavailable.
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

    client = get_client()
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

    except Exception as e:
        logger.debug("Could not get or create default email inbox: %s", e)
        return None


def get_default_inbox() -> Any:
    """Get or create the agent's default inbox object.

    Returns:
        Inbox object, or None if unavailable.
    """
    global _default_inbox

    if _default_inbox is not None:
        return _default_inbox

    # This will populate _default_inbox as a side effect
    get_inbox_id()
    return _default_inbox


def reset_cache():
    """Reset all cached values. Useful for testing."""
    global _client, _default_inbox, _default_inbox_id
    _client = None
    _default_inbox = None
    _default_inbox_id = None
