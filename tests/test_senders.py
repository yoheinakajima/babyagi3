"""
Unit tests for senders/ modules

Tests cover:
- Sender protocol (duck typing check)
- CLISender creation and send method
"""

import asyncio
from unittest.mock import patch, MagicMock

import pytest

from senders import Sender
from senders.cli import CLISender


# =============================================================================
# Sender Protocol Tests
# =============================================================================


class TestSenderProtocol:
    """Test that the Sender protocol works for duck typing."""

    def test_clisender_matches_protocol(self):
        sender = CLISender()
        assert isinstance(sender, Sender)

    def test_custom_sender_matches_protocol(self):
        class CustomSender:
            name = "custom"
            capabilities = ["text_only"]

            async def send(self, to: str, content: str, **kwargs) -> dict:
                return {"sent": True}

        assert isinstance(CustomSender(), Sender)

    def test_incomplete_sender_fails_protocol(self):
        class BadSender:
            name = "bad"
            # Missing capabilities and send

        assert not isinstance(BadSender(), Sender)


# =============================================================================
# CLISender Tests
# =============================================================================


class TestCLISender:
    """Test CLISender implementation."""

    def test_creation_defaults(self):
        sender = CLISender()
        assert sender.name == "cli"
        assert sender.capabilities == ["text_only"]
        assert sender.prefix == "Assistant"

    def test_creation_custom_prefix(self):
        sender = CLISender(prefix="Agent")
        assert sender.prefix == "Agent"

    @pytest.mark.asyncio
    async def test_send(self):
        sender = CLISender()
        with patch("senders.cli.console") as mock_console:
            result = await sender.send("anyone", "Hello world")
            assert result == {"sent": True, "channel": "cli"}
            mock_console.agent.assert_called_once_with(
                "Hello world", prefix="Assistant"
            )

    @pytest.mark.asyncio
    async def test_send_ignores_to(self):
        """CLI sender always prints to stdout regardless of 'to'."""
        sender = CLISender()
        with patch("senders.cli.console") as mock_console:
            await sender.send("user@example.com", "Test message")
            mock_console.agent.assert_called_once()
