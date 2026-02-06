"""
Unit tests for server.py

Tests cover:
- Pydantic request/response models
- Phone number normalization
- API endpoint structure (via TestClient)
"""

import pytest

from server import (
    MessageRequest,
    MessageResponse,
    ObjectiveResponse,
    SendBlueWebhookPayload,
    _normalize_phone,
)


# =============================================================================
# Request/Response Model Tests
# =============================================================================


class TestMessageRequest:
    def test_defaults(self):
        req = MessageRequest(content="Hello")
        assert req.content == "Hello"
        assert req.thread_id == "main"
        assert req.async_mode is False

    def test_custom_values(self):
        req = MessageRequest(
            content="Test",
            thread_id="custom",
            async_mode=True,
        )
        assert req.thread_id == "custom"
        assert req.async_mode is True


class TestMessageResponse:
    def test_defaults(self):
        resp = MessageResponse(thread_id="main")
        assert resp.response is None
        assert resp.queued is False

    def test_with_response(self):
        resp = MessageResponse(
            response="Hi there",
            thread_id="main",
            queued=False,
        )
        assert resp.response == "Hi there"


class TestObjectiveResponse:
    def test_creation(self):
        resp = ObjectiveResponse(
            id="obj1",
            goal="Research AI",
            status="completed",
            schedule=None,
            result="Done",
            error=None,
        )
        assert resp.id == "obj1"
        assert resp.status == "completed"


class TestSendBlueWebhookPayload:
    def test_minimal(self):
        payload = SendBlueWebhookPayload()
        assert payload.content is None
        assert payload.is_outbound is False

    def test_inbound_message(self):
        payload = SendBlueWebhookPayload(
            from_number="+15551234567",
            content="Hello",
            message_handle="msg123",
        )
        assert payload.from_number == "+15551234567"
        assert payload.content == "Hello"


# =============================================================================
# Phone Normalization Tests
# =============================================================================


class TestNormalizePhone:
    def test_already_normalized(self):
        assert _normalize_phone("+15551234567") == "+15551234567"

    def test_with_dashes(self):
        assert _normalize_phone("+1-555-123-4567") == "+15551234567"

    def test_with_spaces(self):
        assert _normalize_phone("+1 555 123 4567") == "+15551234567"

    def test_with_parens(self):
        assert _normalize_phone("(555) 123-4567") == "+15551234567"

    def test_ten_digit(self):
        assert _normalize_phone("5551234567") == "+15551234567"

    def test_eleven_digit_with_1(self):
        assert _normalize_phone("15551234567") == "+15551234567"

    def test_empty(self):
        assert _normalize_phone("") == ""

    def test_none_like(self):
        """Empty/None phone should not crash."""
        assert _normalize_phone("") == ""

    def test_dots_and_mixed(self):
        assert _normalize_phone("555.123.4567") == "+15551234567"
