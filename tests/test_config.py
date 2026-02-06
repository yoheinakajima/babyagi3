"""
Unit tests for config.py

Tests cover:
- Environment variable substitution (${VAR} and ${VAR:default})
- Default config generation
- Config loading from YAML files
- Config merging with defaults
- Channel config helpers (get_channel_config, is_channel_enabled)
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from config import (
    load_config,
    _substitute_env_vars,
    _default_config,
    _merge_with_defaults,
    get_channel_config,
    is_channel_enabled,
)


# =============================================================================
# Environment Variable Substitution Tests
# =============================================================================


class TestSubstituteEnvVars:
    """Test _substitute_env_vars for ${VAR} and ${VAR:default} patterns."""

    def test_simple_var(self):
        with patch.dict(os.environ, {"MY_VAR": "hello"}):
            result = _substitute_env_vars("value: ${MY_VAR}")
            assert result == "value: hello"

    def test_var_with_default(self):
        # Ensure MY_VAR is not set
        env = os.environ.copy()
        env.pop("MY_MISSING_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            result = _substitute_env_vars("value: ${MY_MISSING_VAR:fallback}")
            assert result == "value: fallback"

    def test_var_set_overrides_default(self):
        with patch.dict(os.environ, {"MY_VAR": "actual"}):
            result = _substitute_env_vars("value: ${MY_VAR:fallback}")
            assert result == "value: actual"

    def test_missing_var_empty_default(self):
        env = os.environ.copy()
        env.pop("NOPE", None)
        with patch.dict(os.environ, env, clear=True):
            result = _substitute_env_vars("value: ${NOPE}")
            assert result == "value: "

    def test_multiple_vars(self):
        with patch.dict(os.environ, {"A": "1", "B": "2"}):
            result = _substitute_env_vars("${A} and ${B}")
            assert result == "1 and 2"

    def test_no_substitution_needed(self):
        result = _substitute_env_vars("plain text with no vars")
        assert result == "plain text with no vars"

    def test_nested_colon_in_default(self):
        env = os.environ.copy()
        env.pop("URL_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            result = _substitute_env_vars("url: ${URL_VAR:http://localhost:8080}")
            assert result == "url: http://localhost:8080"


# =============================================================================
# Default Config Tests
# =============================================================================


class TestDefaultConfig:
    """Test _default_config generation."""

    def test_has_required_sections(self, clean_env):
        config = _default_config()
        assert "owner" in config
        assert "channels" in config
        assert "agent" in config

    def test_cli_enabled_by_default(self, clean_env):
        config = _default_config()
        assert config["channels"]["cli"]["enabled"] is True

    def test_voice_disabled_by_default(self, clean_env):
        config = _default_config()
        assert config["channels"]["voice"]["enabled"] is False

    def test_owner_defaults(self, clean_env):
        config = _default_config()
        assert config["owner"]["id"] == "owner"
        assert config["owner"]["name"] == ""

    def test_env_overrides_defaults(self):
        with patch.dict(os.environ, {
            "OWNER_ID": "custom_owner",
            "OWNER_NAME": "John",
            "AGENT_NAME": "MyBot",
        }):
            config = _default_config()
            assert config["owner"]["id"] == "custom_owner"
            assert config["owner"]["name"] == "John"
            assert config["agent"]["name"] == "MyBot"


# =============================================================================
# Config Merge Tests
# =============================================================================


class TestMergeWithDefaults:
    """Test deep merge of user config with defaults."""

    def test_empty_override(self, clean_env):
        result = _merge_with_defaults({})
        # Should have all default sections
        assert "owner" in result
        assert "channels" in result
        assert "agent" in result

    def test_partial_override(self, clean_env):
        result = _merge_with_defaults({"agent": {"name": "Custom"}})
        assert result["agent"]["name"] == "Custom"
        # Other agent keys should still be from defaults
        assert "description" in result["agent"]

    def test_nested_deep_merge(self, clean_env):
        result = _merge_with_defaults({
            "channels": {
                "cli": {"enabled": False}
            }
        })
        assert result["channels"]["cli"]["enabled"] is False
        # Other channels should still exist
        assert "email" in result["channels"]

    def test_new_keys_added(self, clean_env):
        result = _merge_with_defaults({"custom_key": "custom_value"})
        assert result["custom_key"] == "custom_value"


# =============================================================================
# Config Loading Tests
# =============================================================================


class TestLoadConfig:
    """Test load_config from YAML file."""

    def test_missing_file_returns_defaults(self, clean_env):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BABYAGI_CONFIG", None)
            config = load_config("/nonexistent/path/config.yaml")
            assert "owner" in config
            assert "channels" in config

    def test_load_from_yaml(self, config_dir, clean_env):
        yaml_content = """
agent:
  name: YAMLBot
  description: A bot from YAML
channels:
  cli:
    enabled: true
"""
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml_content)
        config = load_config(str(config_file))
        assert config["agent"]["name"] == "YAMLBot"
        assert config["channels"]["cli"]["enabled"] is True

    def test_yaml_with_env_substitution(self, config_dir, clean_env):
        yaml_content = """
agent:
  name: ${TEST_BOT_NAME:DefaultBot}
"""
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml_content)

        with patch.dict(os.environ, {"TEST_BOT_NAME": "EnvBot"}):
            config = load_config(str(config_file))
            assert config["agent"]["name"] == "EnvBot"

    def test_env_var_for_config_path(self, config_dir, clean_env):
        yaml_content = """
agent:
  name: EnvPathBot
"""
        config_file = config_dir / "config.yaml"
        config_file.write_text(yaml_content)
        with patch.dict(os.environ, {"BABYAGI_CONFIG": str(config_file)}):
            config = load_config()
            assert config["agent"]["name"] == "EnvPathBot"


# =============================================================================
# Channel Config Helpers Tests
# =============================================================================


class TestChannelHelpers:
    """Test get_channel_config and is_channel_enabled."""

    def test_get_channel_config(self):
        config = {
            "channels": {
                "cli": {"enabled": True, "theme": "dark"},
                "email": {"enabled": False},
            }
        }
        cli_config = get_channel_config(config, "cli")
        assert cli_config["enabled"] is True
        assert cli_config["theme"] == "dark"

    def test_get_nonexistent_channel(self):
        config = {"channels": {"cli": {"enabled": True}}}
        result = get_channel_config(config, "voice")
        assert result == {}

    def test_get_channel_no_channels_section(self):
        config = {}
        result = get_channel_config(config, "cli")
        assert result == {}

    def test_is_channel_enabled_true(self):
        config = {"channels": {"cli": {"enabled": True}}}
        assert is_channel_enabled(config, "cli") is True

    def test_is_channel_enabled_false(self):
        config = {"channels": {"email": {"enabled": False}}}
        assert is_channel_enabled(config, "email") is False

    def test_is_channel_enabled_missing(self):
        config = {"channels": {}}
        assert is_channel_enabled(config, "voice") is False

    def test_is_channel_enabled_no_enabled_key(self):
        config = {"channels": {"cli": {"theme": "dark"}}}
        assert is_channel_enabled(config, "cli") is False
