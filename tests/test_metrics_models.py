"""
Unit tests for metrics/models.py

Tests cover:
- LLMCallMetric computed properties (total_tokens, tokens_per_second)
- EmbeddingCallMetric fields
- ToolCallMetric fields
- SessionMetrics computed properties
- MetricsSummary computed properties
- SourceMetrics
"""

from datetime import datetime

import pytest

from metrics.models import (
    LLMCallMetric,
    EmbeddingCallMetric,
    ToolCallMetric,
    SessionMetrics,
    MetricsSummary,
    SourceMetrics,
)


# =============================================================================
# LLMCallMetric Tests
# =============================================================================


class TestLLMCallMetric:
    def test_creation(self):
        m = LLMCallMetric(
            id="m1",
            timestamp=datetime.now(),
            source="agent",
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
            duration_ms=500,
        )
        assert m.source == "agent"
        assert m.cost_usd == 0.001

    def test_total_tokens(self):
        m = LLMCallMetric(
            id="m1", timestamp=datetime.now(),
            source="agent", model="test",
            input_tokens=100, output_tokens=50,
        )
        assert m.total_tokens == 150

    def test_tokens_per_second(self):
        m = LLMCallMetric(
            id="m1", timestamp=datetime.now(),
            source="agent", model="test",
            input_tokens=100, output_tokens=100,
            duration_ms=1000,
        )
        assert m.tokens_per_second == 200.0

    def test_tokens_per_second_zero_duration(self):
        m = LLMCallMetric(
            id="m1", timestamp=datetime.now(),
            source="agent", model="test",
            input_tokens=100, output_tokens=100,
            duration_ms=0,
        )
        assert m.tokens_per_second == 0.0


# =============================================================================
# EmbeddingCallMetric Tests
# =============================================================================


class TestEmbeddingCallMetric:
    def test_creation(self):
        m = EmbeddingCallMetric(
            id="e1",
            timestamp=datetime.now(),
            provider="openai",
            model="text-embedding-3-small",
            text_count=10,
            token_estimate=500,
            cost_usd=0.00001,
            duration_ms=200,
            cached=True,
        )
        assert m.cached is True
        assert m.text_count == 10

    def test_defaults(self):
        m = EmbeddingCallMetric(
            id="e2", timestamp=datetime.now(),
            provider="openai", model="test",
        )
        assert m.text_count == 1
        assert m.cached is False


# =============================================================================
# ToolCallMetric Tests
# =============================================================================


class TestToolCallMetric:
    def test_success(self):
        m = ToolCallMetric(
            id="t1", timestamp=datetime.now(),
            tool_name="memory",
            duration_ms=100,
            success=True,
        )
        assert m.success is True
        assert m.error_message is None

    def test_failure(self):
        m = ToolCallMetric(
            id="t2", timestamp=datetime.now(),
            tool_name="web_search",
            duration_ms=5000,
            success=False,
            error_message="Timeout",
        )
        assert m.success is False
        assert m.error_message == "Timeout"


# =============================================================================
# SessionMetrics Tests
# =============================================================================


class TestSessionMetrics:
    def test_creation(self):
        sm = SessionMetrics(
            thread_id="thread_1",
            started_at=datetime.now(),
            last_activity=datetime.now(),
        )
        assert sm.llm_call_count == 0
        assert sm.total_cost_usd == 0.0
        assert sm.total_tokens == 0

    def test_total_cost(self):
        sm = SessionMetrics(
            thread_id="thread_1",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            total_llm_cost_usd=1.50,
            total_embedding_cost_usd=0.10,
        )
        assert sm.total_cost_usd == 1.60

    def test_total_tokens(self):
        sm = SessionMetrics(
            thread_id="thread_1",
            started_at=datetime.now(),
            last_activity=datetime.now(),
            total_input_tokens=1000,
            total_output_tokens=500,
        )
        assert sm.total_tokens == 1500


# =============================================================================
# MetricsSummary Tests
# =============================================================================


class TestMetricsSummary:
    def test_creation(self):
        ms = MetricsSummary(
            period_start=datetime.now(),
            period_end=datetime.now(),
        )
        assert ms.total_llm_calls == 0
        assert ms.tool_success_rate == 100.0

    def test_total_cost(self):
        ms = MetricsSummary(
            period_start=datetime.now(),
            period_end=datetime.now(),
            total_llm_cost_usd=5.00,
            total_embedding_cost_usd=0.50,
        )
        assert ms.total_cost_usd == 5.50


# =============================================================================
# SourceMetrics Tests
# =============================================================================


class TestSourceMetrics:
    def test_creation(self):
        sm = SourceMetrics(
            source="agent",
            period_start=datetime.now(),
            period_end=datetime.now(),
            call_count=10,
            cost_usd=0.50,
        )
        assert sm.source == "agent"
        assert sm.cost_pct == 0.0  # Placeholder
