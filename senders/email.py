"""
Email Sender - Send emails via AgentMail.

Leverages the existing email tool infrastructure.
"""

import os


class EmailSender:
    """Sender that sends emails via AgentMail."""

    name = "email"
    capabilities = ["attachments", "subject", "html"]

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._client = None
        self._inbox_id = None

    def _get_client(self):
        """Get or create AgentMail client."""
        if self._client is not None:
            return self._client

        try:
            from agentmail import AgentMail
        except ImportError:
            return None

        api_key = self.config.get("api_key") or os.environ.get("AGENTMAIL_API_KEY")
        if not api_key:
            return None

        self._client = AgentMail(api_key=api_key)
        return self._client

    def _get_inbox_id(self):
        """Get inbox ID to send from."""
        if self._inbox_id:
            return self._inbox_id

        # Check config first
        self._inbox_id = self.config.get("inbox_id") or os.environ.get("AGENTMAIL_INBOX_ID")
        if self._inbox_id:
            return self._inbox_id

        # Try to get from client
        client = self._get_client()
        if not client:
            return None

        try:
            inboxes_response = client.inboxes.list()
            inboxes = getattr(inboxes_response, 'inboxes', None) or inboxes_response

            if inboxes and len(inboxes) > 0:
                inbox = inboxes[0]
                self._inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
                return self._inbox_id

            # Create new inbox
            inbox = client.inboxes.create()
            self._inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
            return self._inbox_id
        except Exception:
            return None

    async def send(self, to: str, content: str, **kwargs) -> dict:
        """Send an email.

        Args:
            to: Recipient email address
            content: Email body
            subject: Email subject (default: "Message from Assistant")
            reply_to: Message ID to reply to
        """
        client = self._get_client()
        if not client:
            return {"error": "AgentMail not configured. Set AGENTMAIL_API_KEY."}

        inbox_id = self._get_inbox_id()
        if not inbox_id:
            return {"error": "No inbox configured. Set AGENTMAIL_INBOX_ID."}

        subject = kwargs.get("subject", "Message from Assistant")

        try:
            message = client.inboxes.messages.send(
                inbox_id=inbox_id,
                to=to,
                subject=subject,
                text=content
            )
            return {
                "sent": True,
                "channel": "email",
                "to": to,
                "subject": subject,
                "message_id": getattr(message, 'message_id', None) or getattr(message, 'id', None)
            }
        except Exception as e:
            return {"error": str(e)}
