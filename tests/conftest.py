"""
Shared fixtures for the BabyAGI test suite.

Provides common setup like temp directories, mock configs,
and environment variable management.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure the project root is on sys.path so tests can import project modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory that is cleaned up after the test."""
    return tmp_path


@pytest.fixture
def scheduler_dir(tmp_path):
    """Provide a temporary directory for scheduler persistence."""
    d = tmp_path / "scheduler"
    d.mkdir()
    return str(d)


@pytest.fixture
def config_dir(tmp_path):
    """Provide a temporary directory for config files."""
    d = tmp_path / "config"
    d.mkdir()
    return d


@pytest.fixture
def clean_env():
    """Temporarily clear LLM-related env vars to avoid side effects."""
    keys = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "AGENTMAIL_API_KEY", "E2B_API_KEY",
        "SENDBLUE_API_KEY", "SENDBLUE_API_SECRET",
        "BABYAGI_CONFIG",
        "OWNER_ID", "OWNER_NAME", "OWNER_EMAIL", "OWNER_PHONE", "OWNER_TIMEZONE",
        "AGENT_NAME", "AGENT_DESCRIPTION", "AGENT_OBJECTIVE",
    ]
    saved = {}
    for key in keys:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    yield
    # Restore
    for key, val in saved.items():
        os.environ[key] = val
    for key in keys:
        if key not in saved and key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_anthropic_key():
    """Set a fake ANTHROPIC_API_KEY for tests that need provider detection."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test-key-123"}):
        yield


@pytest.fixture
def mock_openai_key():
    """Set a fake OPENAI_API_KEY for tests that need provider detection."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key-123"}):
        yield


@pytest.fixture
def sample_config():
    """Return a minimal config dict for testing."""
    return {
        "owner": {
            "id": "test_owner",
            "name": "Test User",
            "email": "test@example.com",
            "phone": "+15551234567",
            "timezone": "UTC",
        },
        "channels": {
            "cli": {"enabled": True},
            "email": {"enabled": False},
            "voice": {"enabled": False},
        },
        "agent": {
            "model": "claude-sonnet-4-20250514",
            "name": "TestAgent",
            "description": "A test agent",
        },
    }
