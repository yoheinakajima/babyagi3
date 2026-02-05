"""
LLM Configuration for multi-provider support using LiteLLM.

This module provides:
- Configurable models for different use cases (coding, research, agent, memory)
- Multi-provider support (OpenAI, Anthropic) via LiteLLM
- Automatic provider detection based on available API keys
- Instrumented clients for metrics tracking

Use cases:
- skill_building_model: Most powerful model for complex skill/tool creation
- coding_model: Powerful model for code generation and analysis
- research_model: Model for research and information synthesis
- agent_model: General agent operations
- memory_model: Memory operations (extraction, summaries, learning)
- fast_model: Quick, cheap operations (classification, routing)

Provider Detection:
- If ANTHROPIC_API_KEY is set: Uses Claude models by default
- If OPENAI_API_KEY is set: Uses GPT models by default
- If both are set: Prefers Anthropic (can be overridden in config)
- If neither is set: Raises helpful error at startup
"""

import os
from dataclasses import dataclass, field
from typing import Any, Literal


def _get_api_key(name: str) -> str | None:
    """Get API key from environment, refreshing each time."""
    return os.environ.get(name)


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.0
    provider: str = "auto"  # auto, openai, anthropic

    def get_litellm_model(self) -> str:
        """Get the model ID formatted for LiteLLM."""
        # LiteLLM uses prefixes for some providers
        # anthropic/ prefix for Anthropic models
        # openai/ or no prefix for OpenAI models
        if self.model_id.startswith("claude"):
            return self.model_id  # LiteLLM auto-detects claude models
        elif self.model_id.startswith("gpt") or self.model_id.startswith("o1"):
            return self.model_id  # LiteLLM auto-detects OpenAI models
        return self.model_id


# ═══════════════════════════════════════════════════════════
# PROVIDER DETECTION AND VALIDATION
# ═══════════════════════════════════════════════════════════


class NoLLMProviderError(Exception):
    """Raised when no LLM provider API key is configured."""
    pass


def get_available_provider() -> str:
    """
    Detect which LLM provider is available based on API keys.

    Returns:
        'anthropic' if ANTHROPIC_API_KEY is set
        'openai' if OPENAI_API_KEY is set
        'none' if neither is set
    """
    if _get_api_key("ANTHROPIC_API_KEY"):
        return "anthropic"
    elif _get_api_key("OPENAI_API_KEY"):
        return "openai"
    return "none"


def is_llm_configured() -> bool:
    """
    Check if at least one LLM provider is configured.

    Returns:
        True if either OPENAI_API_KEY or ANTHROPIC_API_KEY is set
    """
    return bool(_get_api_key("OPENAI_API_KEY") or _get_api_key("ANTHROPIC_API_KEY"))


def get_missing_config_message() -> str:
    """
    Get a helpful message explaining how to configure an LLM provider.

    Returns:
        Multi-line string with setup instructions
    """
    return """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         LLM CONFIGURATION REQUIRED                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  BabyAGI requires an LLM provider to function. Please set one of these       ║
║  environment variables:                                                      ║
║                                                                              ║
║  Option 1: Anthropic (Claude) - Recommended                                  ║
║  ─────────────────────────────────────────────                               ║
║    export ANTHROPIC_API_KEY="sk-ant-..."                                     ║
║                                                                              ║
║    Get your API key at: https://console.anthropic.com/                       ║
║    Models used: claude-sonnet-4, claude-3-5-haiku                            ║
║                                                                              ║
║  Option 2: OpenAI (GPT)                                                      ║
║  ──────────────────────                                                      ║
║    export OPENAI_API_KEY="sk-..."                                            ║
║                                                                              ║
║    Get your API key at: https://platform.openai.com/api-keys                 ║
║    Models used: gpt-4o, gpt-4o-mini                                          ║
║                                                                              ║
║  You can also set these in a .env file in the project root.                  ║
║                                                                              ║
║  For model customization, edit config.yaml:                                  ║
║    llm:                                                                      ║
║      agent_model:                                                            ║
║        model: "gpt-4o"  # or "claude-sonnet-4-20250514"                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def check_llm_configuration() -> None:
    """
    Check if LLM is configured and raise helpful error if not.

    Raises:
        NoLLMProviderError: If no LLM provider is configured
    """
    if not is_llm_configured():
        raise NoLLMProviderError(get_missing_config_message())


# ═══════════════════════════════════════════════════════════
# DEFAULT MODEL CONFIGURATIONS BY PROVIDER
# ═══════════════════════════════════════════════════════════


# Default models for Anthropic (when ANTHROPIC_API_KEY is set)
ANTHROPIC_DEFAULTS = {
    "skill_building": "claude-opus-4-20250514",  # Most powerful for complex skill creation
    "coding": "claude-sonnet-4-20250514",        # Best for code generation
    "research": "claude-sonnet-4-20250514",      # Good for research/analysis
    "agent": "claude-sonnet-4-20250514",         # Main agent operations
    "memory": "claude-sonnet-4-20250514",        # Memory extraction/summaries
    "fast": "claude-3-5-haiku-20241022",         # Quick/cheap operations
    "embedding": "text-embedding-3-small",       # OpenAI embeddings (or local fallback)
}

# Default models for OpenAI (when OPENAI_API_KEY is set)
OPENAI_DEFAULTS = {
    "skill_building": "o1",       # Most powerful for complex skill creation (reasoning model)
    "coding": "gpt-4o",           # Best for code generation
    "research": "gpt-4o",         # Good for research/analysis
    "agent": "gpt-4o",            # Main agent operations
    "memory": "gpt-4o-mini",      # Memory extraction/summaries (cheaper)
    "fast": "gpt-4o-mini",        # Quick/cheap operations
    "embedding": "text-embedding-3-small",  # OpenAI embeddings
}


def get_default_models_for_provider(provider: str) -> dict[str, str]:
    """
    Get default model IDs for each use case based on provider.

    Args:
        provider: 'anthropic' or 'openai'

    Returns:
        Dictionary mapping use case to model ID
    """
    if provider == "anthropic":
        return ANTHROPIC_DEFAULTS.copy()
    elif provider == "openai":
        return OPENAI_DEFAULTS.copy()
    else:
        # No provider - return Anthropic defaults but they won't work
        return ANTHROPIC_DEFAULTS.copy()


# ═══════════════════════════════════════════════════════════
# LLM CONFIGURATION CLASS
# ═══════════════════════════════════════════════════════════


@dataclass
class LLMConfig:
    """
    Configuration for all LLM models used in the system.

    Different use cases can use different models:
    - skill_building: Most powerful model for complex skill/tool creation (e.g., Claude Opus 4, o1)
    - coding: Powerful model for code generation (e.g., Claude Sonnet, GPT-4o)
    - research: Model for research tasks (e.g., Claude Sonnet, GPT-4o)
    - agent: Main agent operations (e.g., Claude Sonnet, GPT-4o)
    - memory: Memory operations like extraction, summaries (e.g., Claude Sonnet, GPT-4o-mini)
    - fast: Quick operations like classification (e.g., Claude Haiku, GPT-4o-mini)
    """

    # Model configurations for different use cases
    skill_building_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id=ANTHROPIC_DEFAULTS["skill_building"],
        max_tokens=16384,  # Higher token limit for complex skill generation
        temperature=0.0,
    ))

    coding_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id=ANTHROPIC_DEFAULTS["coding"],
        max_tokens=8096,
        temperature=0.0,
    ))

    research_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id=ANTHROPIC_DEFAULTS["research"],
        max_tokens=4096,
        temperature=0.0,
    ))

    agent_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id=ANTHROPIC_DEFAULTS["agent"],
        max_tokens=8096,
        temperature=0.0,
    ))

    memory_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id=ANTHROPIC_DEFAULTS["memory"],
        max_tokens=2048,
        temperature=0.0,
    ))

    fast_model: ModelConfig = field(default_factory=lambda: ModelConfig(
        model_id=ANTHROPIC_DEFAULTS["fast"],
        max_tokens=1024,
        temperature=0.0,
    ))

    # Embedding configuration
    embedding_model: str = "text-embedding-3-small"
    embedding_provider: str = "auto"  # auto, openai, voyage, local

    @classmethod
    def from_config(cls, config: dict) -> "LLMConfig":
        """Create LLMConfig from a configuration dictionary."""
        llm_config = config.get("llm", {})

        # Start with provider-appropriate defaults
        provider = get_available_provider()
        defaults = get_default_models_for_provider(provider)

        instance = cls(
            skill_building_model=ModelConfig(model_id=defaults["skill_building"], max_tokens=16384),
            coding_model=ModelConfig(model_id=defaults["coding"], max_tokens=8096),
            research_model=ModelConfig(model_id=defaults["research"], max_tokens=4096),
            agent_model=ModelConfig(model_id=defaults["agent"], max_tokens=8096),
            memory_model=ModelConfig(model_id=defaults["memory"], max_tokens=2048),
            fast_model=ModelConfig(model_id=defaults["fast"], max_tokens=1024),
            embedding_model=defaults.get("embedding", "text-embedding-3-small"),
            embedding_provider=provider if provider != "none" else "local",
        )

        # Override with config values if provided
        if "skill_building_model" in llm_config:
            instance.skill_building_model = cls._parse_model_config(llm_config["skill_building_model"])
        if "coding_model" in llm_config:
            instance.coding_model = cls._parse_model_config(llm_config["coding_model"])
        if "research_model" in llm_config:
            instance.research_model = cls._parse_model_config(llm_config["research_model"])
        if "agent_model" in llm_config:
            instance.agent_model = cls._parse_model_config(llm_config["agent_model"])
        if "memory_model" in llm_config:
            instance.memory_model = cls._parse_model_config(llm_config["memory_model"])
        if "fast_model" in llm_config:
            instance.fast_model = cls._parse_model_config(llm_config["fast_model"])

        # Embedding configuration
        if "embedding_model" in llm_config:
            instance.embedding_model = llm_config["embedding_model"]
        if "embedding_provider" in llm_config:
            instance.embedding_provider = llm_config["embedding_provider"]

        return instance

    @staticmethod
    def _parse_model_config(config: dict | str) -> ModelConfig:
        """Parse a model configuration from dict or string."""
        if isinstance(config, str):
            return ModelConfig(model_id=config)
        return ModelConfig(
            model_id=config.get("model", config.get("model_id", ANTHROPIC_DEFAULTS["agent"])),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.0),
            provider=config.get("provider", "auto"),
        )


def create_default_config() -> LLMConfig:
    """
    Create a default LLMConfig based on available API keys.

    Automatically selects appropriate models based on which provider is available.
    """
    provider = get_available_provider()
    models = get_default_models_for_provider(provider)

    return LLMConfig(
        skill_building_model=ModelConfig(model_id=models["skill_building"], max_tokens=16384),
        coding_model=ModelConfig(model_id=models["coding"], max_tokens=8096),
        research_model=ModelConfig(model_id=models["research"], max_tokens=4096),
        agent_model=ModelConfig(model_id=models["agent"], max_tokens=8096),
        memory_model=ModelConfig(model_id=models["memory"], max_tokens=2048),
        fast_model=ModelConfig(model_id=models["fast"], max_tokens=1024),
        embedding_model="text-embedding-3-small" if provider == "openai" else "text-embedding-3-small",
        embedding_provider=provider if provider != "none" else "local",
    )


# Global configuration instance
_llm_config: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """Get the global LLM configuration."""
    global _llm_config
    if _llm_config is None:
        _llm_config = create_default_config()
    return _llm_config


def set_llm_config(config: LLMConfig):
    """Set the global LLM configuration."""
    global _llm_config
    _llm_config = config


def init_llm_config(config: dict | None = None):
    """
    Initialize the LLM configuration from a config dictionary.

    Args:
        config: Configuration dictionary (typically from config.yaml)
    """
    global _llm_config
    if config:
        _llm_config = LLMConfig.from_config(config)
    else:
        _llm_config = create_default_config()


# ═══════════════════════════════════════════════════════════
# LITELLM COMPLETION WRAPPER
# ═══════════════════════════════════════════════════════════


class LiteLLMClient:
    """
    Wrapper around LiteLLM for unified LLM access.

    Provides a consistent interface regardless of the underlying provider.
    Handles both sync and async calls.
    """

    def __init__(self, model_config: ModelConfig | None = None):
        self.model_config = model_config or get_llm_config().agent_model
        self._check_litellm()

    def _check_litellm(self):
        """Check if litellm is available."""
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm is required for multi-provider LLM support. "
                "Install it with: pip install litellm"
            )

    def completion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> Any:
        """
        Make a synchronous completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            max_tokens: Override max tokens
            temperature: Override temperature
            tools: Optional list of tool definitions
            **kwargs: Additional arguments passed to litellm

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        model = model or self.model_config.model_id
        max_tokens = max_tokens or self.model_config.max_tokens
        temperature = temperature if temperature is not None else self.model_config.temperature

        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        if tools:
            call_kwargs["tools"] = tools

        return self._litellm.completion(**call_kwargs)

    async def acompletion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> Any:
        """
        Make an asynchronous completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Override the default model
            max_tokens: Override max tokens
            temperature: Override temperature
            tools: Optional list of tool definitions
            **kwargs: Additional arguments passed to litellm

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        model = model or self.model_config.model_id
        max_tokens = max_tokens or self.model_config.max_tokens
        temperature = temperature if temperature is not None else self.model_config.temperature

        call_kwargs = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        if tools:
            call_kwargs["tools"] = tools

        return await self._litellm.acompletion(**call_kwargs)


def get_client_for_use_case(
    use_case: Literal["skill_building", "coding", "research", "agent", "memory", "fast"]
) -> LiteLLMClient:
    """
    Get a LiteLLM client configured for a specific use case.

    Args:
        use_case: One of 'skill_building', 'coding', 'research', 'agent', 'memory', 'fast'

    Returns:
        Configured LiteLLMClient
    """
    config = get_llm_config()

    model_map = {
        "skill_building": config.skill_building_model,
        "coding": config.coding_model,
        "research": config.research_model,
        "agent": config.agent_model,
        "memory": config.memory_model,
        "fast": config.fast_model,
    }

    return LiteLLMClient(model_map.get(use_case, config.agent_model))
