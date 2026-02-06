"""
Unit tests for llm_config.py

Tests cover:
- ModelConfig creation and get_litellm_model
- Provider detection (Anthropic, OpenAI, none)
- is_llm_configured check
- NoLLMProviderError
- Default model configurations for each provider
- LLMConfig creation from config dict
- Global config management (get/set/init)
- create_default_config
"""

import os
from unittest.mock import patch

import pytest

from llm_config import (
    ModelConfig,
    LLMConfig,
    NoLLMProviderError,
    get_available_provider,
    is_llm_configured,
    get_missing_config_message,
    check_llm_configuration,
    get_default_models_for_provider,
    create_default_config,
    get_llm_config,
    set_llm_config,
    init_llm_config,
    ANTHROPIC_DEFAULTS,
    OPENAI_DEFAULTS,
)


# =============================================================================
# ModelConfig Tests
# =============================================================================


class TestModelConfig:
    """Test ModelConfig dataclass."""

    def test_defaults(self):
        mc = ModelConfig(model_id="claude-sonnet-4-20250514")
        assert mc.max_tokens == 4096
        assert mc.temperature == 0.0
        assert mc.provider == "auto"

    def test_custom_values(self):
        mc = ModelConfig(
            model_id="gpt-4o",
            max_tokens=8096,
            temperature=0.7,
            provider="openai",
        )
        assert mc.model_id == "gpt-4o"
        assert mc.max_tokens == 8096

    def test_get_litellm_model_claude(self):
        mc = ModelConfig(model_id="claude-sonnet-4-20250514")
        assert mc.get_litellm_model() == "claude-sonnet-4-20250514"

    def test_get_litellm_model_gpt(self):
        mc = ModelConfig(model_id="gpt-4o")
        assert mc.get_litellm_model() == "gpt-4o"

    def test_get_litellm_model_o1(self):
        mc = ModelConfig(model_id="o1")
        assert mc.get_litellm_model() == "o1"

    def test_get_litellm_model_other(self):
        mc = ModelConfig(model_id="custom-model")
        assert mc.get_litellm_model() == "custom-model"


# =============================================================================
# Provider Detection Tests
# =============================================================================


class TestProviderDetection:
    """Test provider detection based on environment variables."""

    def test_anthropic_detected(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            assert get_available_provider() == "anthropic"

    def test_openai_detected(self, clean_env):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            assert get_available_provider() == "openai"

    def test_anthropic_preferred_when_both(self, clean_env):
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "sk-ant-test",
            "OPENAI_API_KEY": "sk-test",
        }):
            assert get_available_provider() == "anthropic"

    def test_none_when_no_keys(self, clean_env):
        assert get_available_provider() == "none"


class TestIsLLMConfigured:
    """Test is_llm_configured check."""

    def test_configured_with_anthropic(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            assert is_llm_configured() is True

    def test_configured_with_openai(self, clean_env):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            assert is_llm_configured() is True

    def test_not_configured(self, clean_env):
        assert is_llm_configured() is False


class TestCheckLLMConfiguration:
    """Test check_llm_configuration raises error when needed."""

    def test_raises_when_not_configured(self, clean_env):
        with pytest.raises(NoLLMProviderError):
            check_llm_configuration()

    def test_no_raise_when_configured(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            check_llm_configuration()  # Should not raise


class TestMissingConfigMessage:
    """Test helpful error message."""

    def test_message_contains_instructions(self):
        msg = get_missing_config_message()
        assert "ANTHROPIC_API_KEY" in msg
        assert "OPENAI_API_KEY" in msg
        assert "config.yaml" in msg


# =============================================================================
# Default Models Tests
# =============================================================================


class TestDefaultModels:
    """Test get_default_models_for_provider."""

    def test_anthropic_defaults(self):
        models = get_default_models_for_provider("anthropic")
        assert "claude" in models["agent"]
        assert "claude" in models["fast"]
        assert models == ANTHROPIC_DEFAULTS

    def test_openai_defaults(self):
        models = get_default_models_for_provider("openai")
        assert "gpt" in models["agent"]
        assert "gpt" in models["fast"]
        assert models == OPENAI_DEFAULTS

    def test_unknown_provider_falls_back(self):
        models = get_default_models_for_provider("none")
        # Falls back to Anthropic defaults
        assert models == ANTHROPIC_DEFAULTS

    def test_defaults_are_copies(self):
        models = get_default_models_for_provider("anthropic")
        models["agent"] = "modified"
        assert ANTHROPIC_DEFAULTS["agent"] != "modified"


# =============================================================================
# LLMConfig Tests
# =============================================================================


class TestLLMConfig:
    """Test LLMConfig dataclass and factory methods."""

    def test_default_creation(self):
        config = LLMConfig()
        assert config.agent_model.model_id is not None
        assert config.fast_model.model_id is not None
        assert config.embedding_model is not None

    def test_from_config_empty(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            config = LLMConfig.from_config({})
            assert "claude" in config.agent_model.model_id

    def test_from_config_with_overrides(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            config = LLMConfig.from_config({
                "llm": {
                    "agent_model": {"model": "custom-model", "max_tokens": 2048},
                    "embedding_model": "text-embedding-3-large",
                }
            })
            assert config.agent_model.model_id == "custom-model"
            assert config.agent_model.max_tokens == 2048
            assert config.embedding_model == "text-embedding-3-large"

    def test_from_config_string_model(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            config = LLMConfig.from_config({
                "llm": {
                    "fast_model": "gpt-4o-mini",
                }
            })
            assert config.fast_model.model_id == "gpt-4o-mini"

    def test_parse_model_config_string(self):
        mc = LLMConfig._parse_model_config("gpt-4o")
        assert mc.model_id == "gpt-4o"
        assert mc.max_tokens == 4096

    def test_parse_model_config_dict(self):
        mc = LLMConfig._parse_model_config({
            "model": "gpt-4o",
            "max_tokens": 8096,
            "temperature": 0.5,
        })
        assert mc.model_id == "gpt-4o"
        assert mc.max_tokens == 8096
        assert mc.temperature == 0.5


# =============================================================================
# Global Config Management Tests
# =============================================================================


class TestGlobalConfig:
    """Test global config get/set/init."""

    def test_create_default_config_anthropic(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            config = create_default_config()
            assert "claude" in config.agent_model.model_id
            assert config.embedding_model == "local"

    def test_create_default_config_openai(self, clean_env):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            config = create_default_config()
            assert "gpt" in config.agent_model.model_id
            assert config.embedding_model == "text-embedding-3-small"

    def test_set_and_get_llm_config(self):
        custom = LLMConfig(
            agent_model=ModelConfig(model_id="test-model"),
        )
        set_llm_config(custom)
        retrieved = get_llm_config()
        assert retrieved.agent_model.model_id == "test-model"

    def test_init_llm_config_with_dict(self, clean_env):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-test"}):
            init_llm_config({"llm": {"embedding_model": "custom-embed"}})
            config = get_llm_config()
            assert config.embedding_model == "custom-embed"

    def test_init_llm_config_without_dict(self, clean_env):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            init_llm_config(None)
            config = get_llm_config()
            assert config is not None
