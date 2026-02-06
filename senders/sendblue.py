"""
SendBlue Sender - Send SMS/iMessage via SendBlue API.

Enables the agent to send text messages via iMessage or SMS.
Requires SENDBLUE_API_KEY and SENDBLUE_API_SECRET environment variables.
"""

import os
import logging

import httpx

logger = logging.getLogger(__name__)

# SendBlue API base URL
SENDBLUE_API_BASE = "https://api.sendblue.co"


class SendBlueSender:
    """Sender that sends SMS/iMessage via SendBlue API."""

    name = "sendblue"
    capabilities = ["sms", "imessage", "media"]

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._api_key = None
        self._api_secret = None
        self._from_number = None

    def _get_credentials(self):
        """Get SendBlue API credentials."""
        if self._api_key is None:
            self._api_key = self.config.get("api_key") or os.environ.get("SENDBLUE_API_KEY")
        if self._api_secret is None:
            self._api_secret = self.config.get("api_secret") or os.environ.get("SENDBLUE_API_SECRET")
        if self._from_number is None:
            self._from_number = self.config.get("from_number") or os.environ.get("SENDBLUE_PHONE_NUMBER")
        return self._api_key, self._api_secret, self._from_number

    def _get_headers(self):
        """Build authentication headers."""
        api_key, api_secret, _ = self._get_credentials()
        return {
            "sb-api-key-id": api_key,
            "sb-api-secret-key": api_secret,
            "Content-Type": "application/json"
        }

    async def send(self, to: str, content: str, **kwargs) -> dict:
        """Send an SMS/iMessage via SendBlue.

        Args:
            to: Recipient phone number (E.164 format, e.g., +19998887777)
            content: Message content
            media_url: Optional URL to media to attach
            send_style: Optional send style (e.g., "invisible", "gentle", "loud")

        Returns:
            dict with sent status, message_id, etc.
        """
        api_key, api_secret, from_number = self._get_credentials()

        if not api_key or not api_secret:
            return {"error": "SendBlue not configured. Set SENDBLUE_API_KEY and SENDBLUE_API_SECRET."}

        if not from_number:
            return {"error": "SendBlue from_number not configured. Set SENDBLUE_PHONE_NUMBER."}

        # Normalize phone number - ensure E.164 format
        to_number = self._normalize_phone(to)

        # Build request payload
        payload = {
            "number": to_number,
            "content": content,
            "from_number": from_number
        }

        # Optional parameters
        if kwargs.get("media_url"):
            payload["media_url"] = kwargs["media_url"]
        if kwargs.get("send_style"):
            payload["send_style"] = kwargs["send_style"]
        if kwargs.get("status_callback"):
            payload["statusCallback"] = kwargs["status_callback"]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{SENDBLUE_API_BASE}/api/send-message",
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30.0
                )

                if response.status_code in (200, 202):
                    data = response.json()
                    return {
                        "sent": True,
                        "channel": "sendblue",
                        "to": to_number,
                        "message_id": data.get("message_handle"),
                        "status": data.get("status", "queued" if response.status_code == 202 else "sent")
                    }
                else:
                    error_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
                    error_msg = error_data.get("error_message", response.text)
                    logger.error(f"SendBlue send failed: {response.status_code} - {error_msg}")
                    return {"error": f"SendBlue API error: {error_msg}"}

        except httpx.TimeoutException:
            return {"error": "SendBlue API timeout"}
        except Exception as e:
            logger.error(f"SendBlue send error: {e}")
            return {"error": str(e)}

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to E.164 format.

        Args:
            phone: Phone number in various formats

        Returns:
            Phone number in E.164 format (+1XXXXXXXXXX)
        """
        # Remove common formatting characters
        cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")

        # If it doesn't start with +, assume US number
        if not cleaned.startswith("+"):
            # Remove leading 1 if present to normalize
            if cleaned.startswith("1") and len(cleaned) == 11:
                cleaned = "+" + cleaned
            else:
                # Assume US number, prepend +1
                cleaned = "+1" + cleaned

        return cleaned
