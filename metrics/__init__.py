"""
Metrics system for BabyAGI.

Provides transparent instrumentation for all API calls (LLM, embeddings)
with cost tracking, latency measurement, and aggregation.

Supports multiple providers:
- Anthropic (Claude): InstrumentedAnthropic, InstrumentedAsyncAnthropic
- OpenAI (GPT, embeddings): InstrumentedOpenAI, InstrumentedAsyncOpenAI

Usage:
    from metrics import (
        InstrumentedAsyncAnthropic,  # For Claude
        InstrumentedAsyncOpenAI,     # For GPT
        MetricsCollector,
        track_source,
        set_event_emitter,
    )

    # Swap clients at initialization
    client = InstrumentedAsyncAnthropic()  # or InstrumentedAsyncOpenAI()
    set_event_emitter(agent)

    # Tag calls by source
    with track_source("extraction"):
        response = client.messages.create(...)  # Anthropic
        # or
        response = client.chat.completions.create(...)  # OpenAI
"""

from .clients import (
    InstrumentedAnthropic,
    InstrumentedAsyncAnthropic,
    InstrumentedOpenAI,
    InstrumentedAsyncOpenAI,
    set_event_emitter,
    track_source,
)
from .collector import MetricsCollector
from .costs import calculate_cost, calculate_embedding_cost, format_cost
from .models import EmbeddingCallMetric, LLMCallMetric, SessionMetrics, MetricsSummary

__all__ = [
    # Instrumented clients - Anthropic
    "InstrumentedAnthropic",
    "InstrumentedAsyncAnthropic",
    # Instrumented clients - OpenAI
    "InstrumentedOpenAI",
    "InstrumentedAsyncOpenAI",
    # Configuration
    "set_event_emitter",
    "track_source",
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
