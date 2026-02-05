"""
Metrics system for BabyAGI.

Provides transparent instrumentation for all API calls (LLM, embeddings)
with cost tracking, latency measurement, and aggregation.

Supports multiple providers:
- Anthropic (Claude): InstrumentedAnthropic, InstrumentedAsyncAnthropic
- OpenAI (GPT, embeddings): InstrumentedOpenAI, InstrumentedAsyncOpenAI
- Multi-provider via LiteLLM: InstrumentedLiteLLM, InstrumentedLiteLLMEmbeddings

Usage:
    from metrics import (
        InstrumentedAsyncAnthropic,  # For Claude (direct)
        InstrumentedAsyncOpenAI,     # For GPT (direct)
        InstrumentedLiteLLM,         # Multi-provider via LiteLLM
        MetricsCollector,
        track_source,
        set_event_emitter,
        get_llm_client,              # Get client for use case
        get_model_for_use_case,      # Get model ID for use case
    )

    # Direct provider clients (legacy)
    client = InstrumentedAsyncAnthropic()  # or InstrumentedAsyncOpenAI()
    set_event_emitter(agent)

    # Multi-provider client (recommended)
    client = InstrumentedLiteLLM()
    response = client.completion(messages=[...], model="claude-sonnet-4-20250514")
    response = client.completion(messages=[...], model="gpt-4o")  # Same interface!

    # Use-case based client
    client = get_llm_client("coding")  # Gets client with coding model
    model = get_model_for_use_case("memory")  # Gets model ID for memory ops

    # Tag calls by source
    with track_source("extraction"):
        response = client.completion(messages=[...])
"""

from .clients import (
    InstrumentedAnthropic,
    InstrumentedAsyncAnthropic,
    InstrumentedOpenAI,
    InstrumentedAsyncOpenAI,
    InstrumentedLiteLLM,
    InstrumentedLiteLLMEmbeddings,
    LiteLLMAnthropicAdapter,
    AsyncLiteLLMAnthropicAdapter,
    set_event_emitter,
    track_source,
    get_llm_client,
    get_model_for_use_case,
)
from .collector import MetricsCollector
from .costs import calculate_cost, calculate_embedding_cost, format_cost
from .models import EmbeddingCallMetric, LLMCallMetric, SessionMetrics, MetricsSummary

__all__ = [
    # Instrumented clients - Anthropic (legacy/direct)
    "InstrumentedAnthropic",
    "InstrumentedAsyncAnthropic",
    # Instrumented clients - OpenAI (legacy/direct)
    "InstrumentedOpenAI",
    "InstrumentedAsyncOpenAI",
    # Instrumented clients - LiteLLM (multi-provider)
    "InstrumentedLiteLLM",
    "InstrumentedLiteLLMEmbeddings",
    # LiteLLM adapters with Anthropic-compatible interface
    "LiteLLMAnthropicAdapter",
    "AsyncLiteLLMAnthropicAdapter",
    # Configuration
    "set_event_emitter",
    "track_source",
    # Use-case based helpers
    "get_llm_client",
    "get_model_for_use_case",
    # Collector
    "MetricsCollector",
    # Cost calculation
    "calculate_cost",
    "calculate_embedding_cost",
    "format_cost",
    # Models
    "LLMCallMetric",
    "EmbeddingCallMetric",
    "SessionMetrics",
    "MetricsSummary",
]
