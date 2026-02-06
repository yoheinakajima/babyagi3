"""
Instrumented API clients for transparent metrics collection.

These are drop-in replacements for the standard Anthropic and OpenAI clients.
They automatically track all API calls, emitting events that the MetricsCollector
can subscribe to.

Supports multiple providers via LiteLLM:
- Anthropic (Claude models)
- OpenAI (GPT models)
- Any LiteLLM-supported provider

Usage:
    # Instead of:
    from anthropic import Anthropic
    client = Anthropic()

    # Use:
    from metrics import InstrumentedAnthropic, set_event_emitter
    client = InstrumentedAnthropic()
    set_event_emitter(agent)  # Agent is an EventEmitter

    # Tag calls by source:
    from metrics import track_source
    with track_source("extraction"):
        response = client.messages.create(...)

    # Or use the LiteLLM-based client for multi-provider support:
    from metrics import InstrumentedLiteLLM
    client = InstrumentedLiteLLM()
    response = client.completion(messages=[...], model="claude-sonnet-4-20250514")
    response = client.completion(messages=[...], model="gpt-4o")  # Same interface!
"""

import asyncio
import logging
import time
from contextlib import contextmanager
from typing import Any, Literal

logger = logging.getLogger(__name__)

from .costs import calculate_cost, calculate_embedding_cost, estimate_tokens


# ═══════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════

# Event emitter reference (set during initialization)
_event_emitter = None

# Current source context for categorizing calls
_current_source = "agent"


def set_event_emitter(emitter):
    """
    Set the global event emitter for metrics.

    Call this once during agent initialization:
        set_event_emitter(agent)  # Agent inherits from EventEmitter

    Args:
        emitter: Object with an emit(event_name, data) method
    """
    global _event_emitter
    _event_emitter = emitter


def get_event_emitter():
    """Get the current event emitter."""
    return _event_emitter


@contextmanager
def track_source(source: str):
    """
    Context manager to set the source of API calls.

    All LLM/embedding calls made within this context will be tagged
    with the specified source for cost breakdown analysis.

    Args:
        source: Source identifier ("agent", "extraction", "summary", "retrieval")

    Usage:
        with track_source("extraction"):
            # All LLM calls here are tagged as "extraction"
            response = client.messages.create(...)
    """
    global _current_source
    previous = _current_source
    _current_source = source
    try:
        yield
    finally:
        _current_source = previous


def get_current_source() -> str:
    """Get the current source context."""
    return _current_source


def _is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient LLM API error worth retrying.

    Covers 503 Service Unavailable, 429 Rate Limit, and connection-level
    failures that are likely to resolve on their own.
    """
    exc_type = type(exc).__name__

    # litellm wraps HTTP 503 as ServiceUnavailableError
    if "ServiceUnavailable" in exc_type:
        return True
    # litellm wraps HTTP 429 as RateLimitError
    if "RateLimit" in exc_type:
        return True
    # litellm internal timeout
    if "Timeout" in exc_type:
        return True
    # Generic connection errors (connection refused, reset, etc.)
    if "APIConnectionError" in exc_type or "ConnectionError" in exc_type:
        return True

    # Fall back to inspecting the string representation for status codes
    exc_str = str(exc).lower()
    if "503" in exc_str or "service unavailable" in exc_str:
        return True
    if "429" in exc_str or "rate limit" in exc_str:
        return True
    if "connection refused" in exc_str or "connection reset" in exc_str:
        return True

    return False


# ═══════════════════════════════════════════════════════════
# INSTRUMENTED ANTHROPIC CLIENT
# ═══════════════════════════════════════════════════════════


class InstrumentedMessages:
    """
    Wrapper for anthropic.Anthropic().messages that tracks calls.

    Transparently intercepts create() calls to emit metrics events.
    """

    def __init__(self, messages):
        self._messages = messages

    def create(self, **kwargs) -> Any:
        """Instrumented messages.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = self._messages.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        model = kwargs.get("model", "unknown")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": response.stop_reason,
            })

        return response


class InstrumentedAsyncMessages:
    """
    Async wrapper for anthropic.AsyncAnthropic().messages.

    Same as InstrumentedMessages but for async clients.
    """

    def __init__(self, messages):
        self._messages = messages

    async def create(self, **kwargs) -> Any:
        """Instrumented async messages.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = await self._messages.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        model = kwargs.get("model", "unknown")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": response.stop_reason,
            })

        return response


class InstrumentedAnthropic:
    """
    Drop-in replacement for anthropic.Anthropic with automatic metrics.

    Usage:
        # Instead of:
        client = anthropic.Anthropic()

        # Use:
        client = InstrumentedAnthropic()

        # Everything else works the same
        response = client.messages.create(...)
    """

    def __init__(self, **kwargs):
        import anthropic
        self._client = anthropic.Anthropic(**kwargs)
        self._messages = InstrumentedMessages(self._client.messages)

    @property
    def messages(self):
        return self._messages


class InstrumentedAsyncAnthropic:
    """
    Drop-in replacement for anthropic.AsyncAnthropic with automatic metrics.

    Usage:
        client = InstrumentedAsyncAnthropic()
        response = await client.messages.create(...)
    """

    def __init__(self, **kwargs):
        import anthropic
        self._client = anthropic.AsyncAnthropic(**kwargs)
        self._messages = InstrumentedAsyncMessages(self._client.messages)

    @property
    def messages(self):
        return self._messages


# ═══════════════════════════════════════════════════════════
# INSTRUMENTED OPENAI CLIENT
# ═══════════════════════════════════════════════════════════


class InstrumentedChatCompletions:
    """
    Wrapper for openai.OpenAI().chat.completions that tracks calls.

    Transparently intercepts create() calls to emit metrics events.
    """

    def __init__(self, completions):
        self._completions = completions

    def create(self, **kwargs) -> Any:
        """Instrumented chat.completions.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = self._completions.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        model = kwargs.get("model", "gpt-4o")
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Determine stop reason
        stop_reason = "unknown"
        if response.choices:
            stop_reason = response.choices[0].finish_reason or "unknown"

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "provider": "openai",
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": stop_reason,
            })

        return response


class InstrumentedAsyncChatCompletions:
    """
    Async wrapper for openai.AsyncOpenAI().chat.completions.

    Same as InstrumentedChatCompletions but for async clients.
    """

    def __init__(self, completions):
        self._completions = completions

    async def create(self, **kwargs) -> Any:
        """Instrumented async chat.completions.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = await self._completions.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        model = kwargs.get("model", "gpt-4o")
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Determine stop reason
        stop_reason = "unknown"
        if response.choices:
            stop_reason = response.choices[0].finish_reason or "unknown"

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "provider": "openai",
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": stop_reason,
            })

        return response


class InstrumentedChat:
    """Wrapper for openai.OpenAI().chat namespace."""

    def __init__(self, chat):
        self._chat = chat
        self._completions = InstrumentedChatCompletions(chat.completions)

    @property
    def completions(self):
        return self._completions


class InstrumentedAsyncChat:
    """Wrapper for openai.AsyncOpenAI().chat namespace."""

    def __init__(self, chat):
        self._chat = chat
        self._completions = InstrumentedAsyncChatCompletions(chat.completions)

    @property
    def completions(self):
        return self._completions


class InstrumentedEmbeddings:
    """
    Wrapper for openai.OpenAI().embeddings that tracks calls.

    Transparently intercepts create() calls to emit metrics events.
    """

    def __init__(self, embeddings):
        self._embeddings = embeddings

    def create(self, **kwargs) -> Any:
        """Instrumented embeddings.create() call."""
        start_time = time.time()

        # Make the actual API call
        response = self._embeddings.create(**kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Calculate metrics
        model = kwargs.get("model", "text-embedding-3-small")
        input_data = kwargs.get("input", "")

        if isinstance(input_data, list):
            text_count = len(input_data)
            token_estimate = sum(estimate_tokens(t) for t in input_data)
        else:
            text_count = 1
            token_estimate = estimate_tokens(input_data)

        cost = calculate_embedding_cost(model, token_estimate)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("embedding_call_end", {
                "provider": "openai",
                "model": model,
                "text_count": text_count,
                "token_estimate": token_estimate,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "cached": False,
            })

        return response


class InstrumentedOpenAI:
    """
    Drop-in replacement for openai.OpenAI with automatic metrics.

    Instruments both chat completions and embeddings APIs.

    Usage:
        # Instead of:
        client = OpenAI()

        # Use:
        client = InstrumentedOpenAI()

        # Everything else works the same
        response = client.chat.completions.create(...)
        embeddings = client.embeddings.create(...)
    """

    def __init__(self, **kwargs):
        from openai import OpenAI
        self._client = OpenAI(**kwargs)
        self._chat = InstrumentedChat(self._client.chat)
        self._embeddings = InstrumentedEmbeddings(self._client.embeddings)

    @property
    def chat(self):
        return self._chat

    @property
    def embeddings(self):
        return self._embeddings


class InstrumentedAsyncOpenAI:
    """
    Drop-in replacement for openai.AsyncOpenAI with automatic metrics.

    Instruments both chat completions and embeddings APIs.

    Usage:
        client = InstrumentedAsyncOpenAI()
        response = await client.chat.completions.create(...)
    """

    def __init__(self, **kwargs):
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(**kwargs)
        self._chat = InstrumentedAsyncChat(self._client.chat)

    @property
    def chat(self):
        return self._chat


# ═══════════════════════════════════════════════════════════
# CACHE-AWARE EMBEDDING TRACKING
# ═══════════════════════════════════════════════════════════


def emit_embedding_cache_hit(model: str, text_count: int, token_estimate: int):
    """
    Emit an event for cached embedding lookups.

    Call this when embeddings are retrieved from cache instead of API.
    This helps track cache hit rates for cost analysis.

    Args:
        model: The model that would have been used
        text_count: Number of texts that were cached
        token_estimate: Estimated tokens saved
    """
    if _event_emitter:
        _event_emitter.emit("embedding_call_end", {
            "provider": "cache",
            "model": model,
            "text_count": text_count,
            "token_estimate": token_estimate,
            "cost_usd": 0.0,
            "duration_ms": 0,
            "cached": True,
        })


# ═══════════════════════════════════════════════════════════
# LITELLM INSTRUMENTED CLIENTS (Multi-Provider Support)
# ═══════════════════════════════════════════════════════════


class InstrumentedLiteLLM:
    """
    Instrumented LiteLLM client for multi-provider LLM support.

    Supports any LiteLLM-compatible model (Anthropic, OpenAI, etc.)
    with automatic metrics tracking.

    Usage:
        client = InstrumentedLiteLLM()

        # Works with any provider
        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet-4-20250514"
        )

        # Or OpenAI
        response = client.completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-4o"
        )
    """

    def __init__(self, default_model: str | None = None):
        """
        Initialize the instrumented LiteLLM client.

        Args:
            default_model: Default model to use if not specified in calls
        """
        self._litellm = None
        self._default_model = default_model
        self._ensure_litellm()

    def _ensure_litellm(self):
        """Lazily import litellm."""
        if self._litellm is None:
            try:
                import litellm
                # Disable litellm's own logging to avoid noise
                litellm.suppress_debug_info = True
                self._litellm = litellm
            except ImportError:
                raise ImportError(
                    "litellm is required for multi-provider LLM support. "
                    "Install it with: pip install litellm"
                )

    def _detect_provider(self, model: str) -> str:
        """Detect the provider from the model name."""
        if model.startswith("claude"):
            return "anthropic"
        elif model.startswith("gpt") or model.startswith("o1"):
            return "openai"
        elif model.startswith("gemini"):
            return "google"
        elif "/" in model:
            # LiteLLM format like "anthropic/claude-3"
            return model.split("/")[0]
        return "unknown"

    def _extract_usage(self, response: Any, model: str) -> tuple[int, int]:
        """Extract token usage from LiteLLM response."""
        try:
            usage = response.usage
            if usage:
                # LiteLLM uses OpenAI-style naming
                input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(usage, "completion_tokens", 0) or 0
                return input_tokens, output_tokens
        except Exception as e:
            logger.debug("Could not extract token usage from response: %s", e)
        return 0, 0

    def _extract_stop_reason(self, response: Any) -> str:
        """Extract stop reason from LiteLLM response."""
        try:
            if response.choices:
                return response.choices[0].finish_reason or "unknown"
        except Exception as e:
            logger.debug("Could not extract stop reason from response: %s", e)
        return "unknown"

    def completion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        system: str | None = None,
        **kwargs
    ) -> Any:
        """
        Make a synchronous completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (e.g., "claude-sonnet-4-20250514", "gpt-4o")
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            tools: Optional list of tool definitions
            system: Optional system prompt (will be prepended to messages)
            **kwargs: Additional arguments passed to litellm

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        model = model or self._default_model
        if not model:
            raise ValueError("Model must be specified either in call or as default")

        start_time = time.time()

        # Handle system prompt - prepend to messages if provided
        final_messages = messages
        if system:
            final_messages = [{"role": "system", "content": system}] + messages

        call_kwargs = {
            "model": model,
            "messages": final_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        if tools:
            call_kwargs["tools"] = tools

        # Make the API call
        response = self._litellm.completion(**call_kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        input_tokens, output_tokens = self._extract_usage(response, model)
        cost = calculate_cost(model, input_tokens, output_tokens)
        stop_reason = self._extract_stop_reason(response)
        provider = self._detect_provider(model)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "provider": provider,
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": stop_reason,
            })

        return response

    async def acompletion(
        self,
        messages: list[dict],
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        tools: list[dict] | None = None,
        system: str | None = None,
        **kwargs
    ) -> Any:
        """
        Make an asynchronous completion call.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (e.g., "claude-sonnet-4-20250514", "gpt-4o")
            max_tokens: Maximum tokens to generate
            temperature: Temperature for sampling
            tools: Optional list of tool definitions
            system: Optional system prompt (will be prepended to messages)
            **kwargs: Additional arguments passed to litellm

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        model = model or self._default_model
        if not model:
            raise ValueError("Model must be specified either in call or as default")

        start_time = time.time()

        # Handle system prompt - prepend to messages if provided
        final_messages = messages
        if system:
            final_messages = [{"role": "system", "content": system}] + messages

        call_kwargs = {
            "model": model,
            "messages": final_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs
        }

        if tools:
            call_kwargs["tools"] = tools

        # Make the API call
        response = await self._litellm.acompletion(**call_kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        input_tokens, output_tokens = self._extract_usage(response, model)
        cost = calculate_cost(model, input_tokens, output_tokens)
        stop_reason = self._extract_stop_reason(response)
        provider = self._detect_provider(model)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "provider": provider,
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": stop_reason,
            })

        return response


class InstrumentedLiteLLMEmbeddings:
    """
    Instrumented LiteLLM client for embeddings.

    Supports OpenAI and other embedding providers via LiteLLM.
    """

    def __init__(self, default_model: str = "text-embedding-3-small"):
        """
        Initialize the instrumented embedding client.

        Args:
            default_model: Default embedding model to use
        """
        self._litellm = None
        self._default_model = default_model
        self._ensure_litellm()

    def _ensure_litellm(self):
        """Lazily import litellm."""
        if self._litellm is None:
            try:
                import litellm
                litellm.suppress_debug_info = True
                self._litellm = litellm
            except ImportError:
                raise ImportError(
                    "litellm is required for embeddings. "
                    "Install it with: pip install litellm"
                )

    def create(
        self,
        input: str | list[str],
        model: str | None = None,
        **kwargs
    ) -> Any:
        """
        Create embeddings for the given input.

        Args:
            input: Text or list of texts to embed
            model: Embedding model to use
            **kwargs: Additional arguments

        Returns:
            LiteLLM embedding response
        """
        model = model or self._default_model
        start_time = time.time()

        # Normalize input to list
        texts = [input] if isinstance(input, str) else input

        # Make the API call
        response = self._litellm.embedding(model=model, input=texts, **kwargs)

        duration_ms = int((time.time() - start_time) * 1000)

        # Calculate metrics
        token_estimate = sum(estimate_tokens(t) for t in texts)
        cost = calculate_embedding_cost(model, token_estimate)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("embedding_call_end", {
                "provider": "litellm",
                "model": model,
                "text_count": len(texts),
                "token_estimate": token_estimate,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "cached": False,
            })

        return response


# ═══════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS FOR USE-CASE BASED CLIENTS
# ═══════════════════════════════════════════════════════════


def get_llm_client(
    use_case: Literal["skill_building", "coding", "research", "agent", "memory", "fast"] = "agent"
) -> InstrumentedLiteLLM:
    """
    Get an instrumented LLM client configured for a specific use case.

    Args:
        use_case: One of 'skill_building', 'coding', 'research', 'agent', 'memory', 'fast'

    Returns:
        Configured InstrumentedLiteLLM client
    """
    try:
        from llm_config import get_llm_config
        config = get_llm_config()

        model_map = {
            "skill_building": config.skill_building_model.model_id,
            "coding": config.coding_model.model_id,
            "research": config.research_model.model_id,
            "agent": config.agent_model.model_id,
            "memory": config.memory_model.model_id,
            "fast": config.fast_model.model_id,
        }

        return InstrumentedLiteLLM(default_model=model_map.get(use_case, config.agent_model.model_id))
    except ImportError:
        # Fallback if llm_config is not available
        return InstrumentedLiteLLM(default_model="claude-sonnet-4-20250514")


def get_model_for_use_case(
    use_case: Literal["skill_building", "coding", "research", "agent", "memory", "fast"] = "agent"
) -> str:
    """
    Get the model ID configured for a specific use case.

    Args:
        use_case: One of 'skill_building', 'coding', 'research', 'agent', 'memory', 'fast'

    Returns:
        Model ID string
    """
    try:
        from llm_config import get_llm_config
        config = get_llm_config()

        model_map = {
            "skill_building": config.skill_building_model.model_id,
            "coding": config.coding_model.model_id,
            "research": config.research_model.model_id,
            "agent": config.agent_model.model_id,
            "memory": config.memory_model.model_id,
            "fast": config.fast_model.model_id,
        }

        return model_map.get(use_case, config.agent_model.model_id)
    except ImportError:
        # Fallback defaults
        defaults = {
            "skill_building": "claude-opus-4-20250514",
            "coding": "claude-sonnet-4-20250514",
            "research": "claude-sonnet-4-20250514",
            "agent": "claude-sonnet-4-20250514",
            "memory": "claude-sonnet-4-20250514",
            "fast": "claude-haiku-4-5-20251001",
        }
        return defaults.get(use_case, "claude-sonnet-4-20250514")


# ═══════════════════════════════════════════════════════════
# ANTHROPIC-COMPATIBLE LITELLM ADAPTER
# ═══════════════════════════════════════════════════════════


class _AnthropicStyleResponse:
    """
    Adapts LiteLLM response to look like Anthropic response.

    This allows code written for the Anthropic API to work with LiteLLM.
    """

    def __init__(self, litellm_response):
        self._response = litellm_response
        self._content = self._convert_content()
        self._usage = self._convert_usage()

    def _convert_content(self):
        """Convert LiteLLM content to Anthropic-style content blocks."""
        content = []
        choice = self._response.choices[0] if self._response.choices else None

        if not choice:
            return content

        message = choice.message

        # Handle text content
        if message.content:
            content.append(_TextBlock(message.content))

        # Handle tool calls (LiteLLM format)
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                content.append(_ToolUseBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=_parse_json_safe(tool_call.function.arguments)
                ))

        return content

    def _convert_usage(self):
        """Convert LiteLLM usage to Anthropic-style usage."""
        usage = self._response.usage
        return _UsageInfo(
            input_tokens=getattr(usage, 'prompt_tokens', 0) or 0,
            output_tokens=getattr(usage, 'completion_tokens', 0) or 0
        )

    @property
    def content(self):
        return self._content

    @property
    def stop_reason(self) -> str:
        choice = self._response.choices[0] if self._response.choices else None
        if not choice:
            return "unknown"
        reason = choice.finish_reason
        # Map OpenAI finish reasons to Anthropic stop reasons
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "end_turn",
        }
        return mapping.get(reason, reason or "unknown")

    @property
    def usage(self):
        return self._usage


class _TextBlock:
    """Mimics Anthropic TextBlock."""

    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _ToolUseBlock:
    """Mimics Anthropic ToolUseBlock."""

    def __init__(self, id: str, name: str, input: dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input


class _UsageInfo:
    """Mimics Anthropic Usage info."""

    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def _parse_json_safe(json_str: str) -> dict:
    """Safely parse JSON string, returning empty dict on failure."""
    import json
    try:
        return json.loads(json_str) if json_str else {}
    except (json.JSONDecodeError, TypeError):
        return {}


class _AnthropicStyleMessages:
    """
    Adapter that provides Anthropic-style messages.create() interface
    on top of LiteLLM.
    """

    def __init__(self, client: "LiteLLMAnthropicAdapter", is_async: bool = False):
        self._client = client
        self._is_async = is_async

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert Anthropic-style tools to OpenAI/LiteLLM format."""
        if not tools:
            return None

        converted = []
        for tool in tools:
            converted.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {})
                }
            })
        return converted

    def _convert_messages(self, messages: list[dict], system: str = None) -> list[dict]:
        """Convert Anthropic-style messages to LiteLLM format."""
        import json
        result = []

        # Add system message if provided
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            # Handle simple string content
            if isinstance(content, str):
                result.append({"role": role, "content": content})
                continue

            # Handle list content (Anthropic style with blocks)
            if isinstance(content, list):
                # Check if it's tool_result blocks
                tool_results = []
                text_parts = []
                tool_calls = []

                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "tool_result":
                            tool_results.append({
                                "role": "tool",
                                "tool_call_id": block.get("tool_use_id", ""),
                                "content": str(block.get("content", ""))
                            })
                        elif block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    elif hasattr(block, "type"):
                        # Handle object-style blocks
                        if block.type == "tool_use":
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(block.input)
                                }
                            })
                        elif block.type == "text":
                            text_parts.append(block.text)

                # Handle assistant messages with tool_use blocks
                if role == "assistant" and (tool_calls or text_parts):
                    msg_dict = {"role": "assistant"}
                    if text_parts:
                        msg_dict["content"] = "\n".join(text_parts)
                    else:
                        msg_dict["content"] = ""
                    if tool_calls:
                        msg_dict["tool_calls"] = tool_calls
                    result.append(msg_dict)
                elif tool_results:
                    # Add tool results as separate messages
                    for tr in tool_results:
                        result.append(tr)
                elif text_parts:
                    result.append({"role": role, "content": "\n".join(text_parts)})

                continue

            # Fallback for other content types
            result.append({"role": role, "content": str(content) if content else ""})

        return result

    def create(self, **kwargs) -> _AnthropicStyleResponse:
        """Synchronous create call with Anthropic-style interface."""
        model = kwargs.get("model")
        max_tokens = kwargs.get("max_tokens", 4096)
        system = kwargs.get("system")
        tools = kwargs.get("tools")
        messages = kwargs.get("messages", [])

        start_time = time.time()

        # Convert to LiteLLM format
        converted_messages = self._convert_messages(messages, system)
        converted_tools = self._convert_tools(tools)

        call_kwargs = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }

        if converted_tools:
            call_kwargs["tools"] = converted_tools

        # Make the call with retry logic for transient errors (e.g. 503)
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = self._client._litellm.completion(**call_kwargs)
                break
            except Exception as e:
                if attempt < max_retries and _is_transient_error(e):
                    delay = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning(
                        f"Transient LLM error (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    raise

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "provider": self._client._detect_provider(model),
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": response.choices[0].finish_reason if response.choices else "unknown",
            })

        return _AnthropicStyleResponse(response)


class _AsyncAnthropicStyleMessages(_AnthropicStyleMessages):
    """Async version of the messages interface."""

    async def create(self, **kwargs) -> _AnthropicStyleResponse:
        """Async create call with Anthropic-style interface."""
        model = kwargs.get("model")
        max_tokens = kwargs.get("max_tokens", 4096)
        system = kwargs.get("system")
        tools = kwargs.get("tools")
        messages = kwargs.get("messages", [])

        start_time = time.time()

        # Convert to LiteLLM format
        converted_messages = self._convert_messages(messages, system)
        converted_tools = self._convert_tools(tools)

        call_kwargs = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
        }

        if converted_tools:
            call_kwargs["tools"] = converted_tools

        # Make the call with retry logic for transient errors (e.g. 503)
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                response = await self._client._litellm.acompletion(**call_kwargs)
                break
            except Exception as e:
                if attempt < max_retries and _is_transient_error(e):
                    delay = 2 ** (attempt + 1)  # 2s, 4s, 8s
                    logger.warning(
                        f"Transient LLM error (attempt {attempt + 1}/{max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

        duration_ms = int((time.time() - start_time) * 1000)

        # Extract metrics
        input_tokens = getattr(response.usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(response.usage, 'completion_tokens', 0) or 0
        cost = calculate_cost(model, input_tokens, output_tokens)

        # Emit event
        if _event_emitter:
            _event_emitter.emit("llm_call_end", {
                "provider": self._client._detect_provider(model),
                "source": _current_source,
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "duration_ms": duration_ms,
                "stop_reason": response.choices[0].finish_reason if response.choices else "unknown",
            })

        return _AnthropicStyleResponse(response)


class LiteLLMAnthropicAdapter:
    """
    Provides an Anthropic-compatible interface on top of LiteLLM.

    This allows existing code using InstrumentedAnthropic/InstrumentedAsyncAnthropic
    to work with any LiteLLM-supported provider without code changes.

    Usage:
        # Works like InstrumentedAnthropic but supports any provider
        client = LiteLLMAnthropicAdapter()

        # Use Claude
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            messages=[...],
            system="You are helpful",
            tools=[...]
        )

        # Same code works with OpenAI
        response = client.messages.create(
            model="gpt-4o",
            messages=[...],
            system="You are helpful",
            tools=[...]
        )
    """

    def __init__(self):
        self._litellm = None
        self._messages = None
        self._ensure_litellm()

    def _ensure_litellm(self):
        """Lazily import litellm."""
        if self._litellm is None:
            try:
                import litellm
                litellm.suppress_debug_info = True
                self._litellm = litellm
            except ImportError:
                raise ImportError(
                    "litellm is required for multi-provider LLM support. "
                    "Install it with: pip install litellm"
                )

    def _detect_provider(self, model: str) -> str:
        """Detect the provider from the model name."""
        if model.startswith("claude"):
            return "anthropic"
        elif model.startswith("gpt") or model.startswith("o1"):
            return "openai"
        elif model.startswith("gemini"):
            return "google"
        elif "/" in model:
            return model.split("/")[0]
        return "unknown"

    @property
    def messages(self) -> _AnthropicStyleMessages:
        """Get the messages interface."""
        if self._messages is None:
            self._messages = _AnthropicStyleMessages(self, is_async=False)
        return self._messages


class AsyncLiteLLMAnthropicAdapter(LiteLLMAnthropicAdapter):
    """
    Async version of LiteLLMAnthropicAdapter.

    Drop-in replacement for InstrumentedAsyncAnthropic that works with any provider.

    Usage:
        client = AsyncLiteLLMAnthropicAdapter()
        response = await client.messages.create(...)
    """

    @property
    def messages(self) -> _AsyncAnthropicStyleMessages:
        """Get the async messages interface."""
        if self._messages is None:
            self._messages = _AsyncAnthropicStyleMessages(self)
        return self._messages
