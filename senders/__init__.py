"""
Senders - Output channels for the agent.

Senders handle outbound communication. Each sender implements a simple protocol:
- name: str - Channel identifier
- capabilities: list[str] - What this sender supports (attachments, images, etc.)
- send(to, content, **kwargs) - Send a message

Usage:
    from senders.email import EmailSender
    agent.register_sender("email", EmailSender(config))
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Sender(Protocol):
    """Protocol for channel senders.

    Implement this to add a new output channel.
    """

    name: str
    capabilities: list[str]

    async def send(self, to: str, content: str, **kwargs) -> dict:
        """Send a message.

        Args:
            to: Recipient identifier (email, phone, user ID, or "owner")
            content: Message content
            **kwargs: Channel-specific options (subject, attachments, image, etc.)

        Returns:
            {"sent": True, ...} on success
            {"error": "..."} on failure
        """
        ...
