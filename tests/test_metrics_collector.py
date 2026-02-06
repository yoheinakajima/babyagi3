"""
Unit tests for metrics/collector.py

Tests cover:
- MetricsCollector initialization
- Event handling (LLM calls, embedding calls, tool calls)
- Summary generation by period
- Metrics by source
- Session metrics
- Error tracking
- Slow operation detection
- Tool metrics
- Clear method
"""

from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from metrics.collector import MetricsCollector
from metrics.models import (
    LLMCallMetric,
    EmbeddingCallMetric,
    ToolCallMetric,
    SessionMetrics,
)


def _make_llm_call(source="agent", model="claude-sonnet-4-20250514",
                    input_tokens=100, output_tokens=50, cost=0.001,
                    duration_ms=500, thread_id=None, minutes_ago=0):
    """Helper to create LLM call metrics."""
    return {
        "source": source,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
        "duration_ms": duration_ms,
        "stop_reason": "end_turn",
    }


def _make_tool_call(name="memory", duration_ms=100, success=True,
                    error_msg=None, thread_id=None):
    """Helper for tool events."""
    result = {"status": "ok"} if success else {"error": error_msg or "Failed"}
    return {
        "name": name,
        "input": {"action": "search"},
        "result": result,
        "duration_ms": duration_ms,
    }


# =============================================================================
# Initialization Tests
# =============================================================================


class TestCollectorInit:
    def test_init_no_store(self):
        c = MetricsCollector()
        assert c.store is None
        assert c._llm_calls == []
        assert c._embedding_calls == []
        assert c._tool_calls == []

    def test_init_with_store(self):
        mock_store = object()
        c = MetricsCollector(store=mock_store)
        assert c.store is mock_store


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    def test_on_llm_call_end(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call())
        assert len(c._llm_calls) == 1
        assert c._llm_calls[0].source == "agent"
        assert c._llm_calls[0].input_tokens == 100

    def test_on_embedding_call_end(self):
        c = MetricsCollector()
        c._on_embedding_call_end({
            "provider": "openai",
            "model": "text-embedding-3-small",
            "text_count": 5,
            "token_estimate": 500,
            "cost_usd": 0.00001,
            "duration_ms": 200,
            "cached": False,
        })
        assert len(c._embedding_calls) == 1
        assert c._embedding_calls[0].provider == "openai"
        assert c._embedding_calls[0].text_count == 5

    def test_on_tool_end_success(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call())
        assert len(c._tool_calls) == 1
        assert c._tool_calls[0].success is True

    def test_on_tool_end_error(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(success=False, error_msg="Connection failed"))
        assert len(c._tool_calls) == 1
        assert c._tool_calls[0].success is False
        assert c._tool_calls[0].error_message == "Connection failed"

    def test_on_tool_end_error_via_status(self):
        c = MetricsCollector()
        c._on_tool_end({
            "name": "web_search",
            "input": {},
            "result": {"status": "error", "message": "Timeout"},
            "duration_ms": 5000,
        })
        assert c._tool_calls[0].success is False

    def test_set_thread_id(self):
        c = MetricsCollector()
        c.set_thread_id("thread_123")
        assert c._current_thread_id == "thread_123"

    def test_thread_id_applied_to_metrics(self):
        c = MetricsCollector()
        c.set_thread_id("thread_abc")
        c._on_llm_call_end(_make_llm_call())
        assert c._llm_calls[0].thread_id == "thread_abc"


# =============================================================================
# Summary Tests
# =============================================================================


class TestGetSummary:
    def test_empty_summary(self):
        c = MetricsCollector()
        summary = c.get_summary(period="all")
        assert summary.total_llm_calls == 0
        assert summary.total_tokens == 0
        assert summary.total_llm_cost_usd == 0.0
        assert summary.tool_success_rate == 100.0

    def test_summary_with_data(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call(cost=0.01))
        c._on_llm_call_end(_make_llm_call(cost=0.02, source="extraction"))
        c._on_tool_end(_make_tool_call())
        c._on_tool_end(_make_tool_call(name="web_search"))

        summary = c.get_summary(period="all")
        assert summary.total_llm_calls == 2
        assert summary.total_llm_cost_usd == pytest.approx(0.03, abs=0.001)
        assert summary.total_tool_calls == 2
        assert summary.tool_success_rate == 100.0

    def test_summary_cost_by_model(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call(model="claude-sonnet-4-20250514", cost=0.01))
        c._on_llm_call_end(_make_llm_call(model="gpt-4o", cost=0.005))

        summary = c.get_summary(period="all")
        assert "claude-sonnet-4-20250514" in summary.cost_by_model
        assert "gpt-4o" in summary.cost_by_model

    def test_summary_cost_by_source(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call(source="agent", cost=0.01))
        c._on_llm_call_end(_make_llm_call(source="extraction", cost=0.005))

        summary = c.get_summary(period="all")
        assert "agent" in summary.cost_by_source
        assert "extraction" in summary.cost_by_source

    def test_summary_calls_by_tool(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="memory"))
        c._on_tool_end(_make_tool_call(name="memory"))
        c._on_tool_end(_make_tool_call(name="web_search"))

        summary = c.get_summary(period="all")
        assert summary.calls_by_tool["memory"] == 2
        assert summary.calls_by_tool["web_search"] == 1

    def test_summary_errors_by_tool(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="web_search", success=False))
        c._on_tool_end(_make_tool_call(name="web_search", success=True))

        summary = c.get_summary(period="all")
        assert summary.errors_by_tool.get("web_search", 0) == 1


# =============================================================================
# Metrics By Source Tests
# =============================================================================


class TestGetMetricsBySource:
    def test_empty(self):
        c = MetricsCollector()
        sources = c.get_metrics_by_source(period="all")
        assert sources == {}

    def test_with_data(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call(source="agent", cost=0.01, duration_ms=500))
        c._on_llm_call_end(_make_llm_call(source="agent", cost=0.02, duration_ms=600))
        c._on_llm_call_end(_make_llm_call(source="extraction", cost=0.005, duration_ms=300))

        sources = c.get_metrics_by_source(period="all")
        assert "agent" in sources
        assert sources["agent"]["call_count"] == 2
        assert sources["agent"]["cost"] == pytest.approx(0.03, abs=0.001)


# =============================================================================
# Session Metrics Tests
# =============================================================================


class TestGetSessionMetrics:
    def test_empty_session(self):
        c = MetricsCollector()
        sm = c.get_session_metrics("thread_123")
        assert sm.thread_id == "thread_123"
        assert sm.llm_call_count == 0
        assert sm.tool_call_count == 0

    def test_session_with_data(self):
        c = MetricsCollector()
        c.set_thread_id("sess_1")
        c._on_llm_call_end(_make_llm_call(cost=0.01))
        c._on_tool_end(_make_tool_call())
        c._on_tool_end(_make_tool_call(success=False))

        sm = c.get_session_metrics("sess_1")
        assert sm.llm_call_count == 1
        assert sm.tool_call_count == 2
        assert sm.tool_success_count == 1
        assert sm.tool_error_count == 1


# =============================================================================
# Error Tracking Tests
# =============================================================================


class TestErrorTracking:
    def test_get_recent_errors(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="web", success=False, error_msg="Timeout"))
        c._on_tool_end(_make_tool_call(name="email", success=False, error_msg="Auth failed"))

        errors = c.get_recent_errors(limit=10)
        assert len(errors) == 2

    def test_get_error_counts(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="web", success=False))
        c._on_tool_end(_make_tool_call(name="web", success=False))
        c._on_tool_end(_make_tool_call(name="email", success=False))

        counts = c.get_error_counts()
        assert counts["web"] == 2
        assert counts["email"] == 1

    def test_get_errors_combined(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="web", success=False, error_msg="Failed"))

        errors = c.get_errors(limit=10)
        assert len(errors) == 1
        assert errors[0]["type"] == "tool"


# =============================================================================
# Slow Operations Tests
# =============================================================================


class TestSlowOperations:
    def test_no_slow_ops(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call(duration_ms=100))
        slow = c.get_slow_operations(threshold_ms=5000)
        assert slow == []

    def test_detect_slow_llm(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call(duration_ms=10000))
        slow = c.get_slow_operations(threshold_ms=5000)
        assert len(slow) == 1
        assert slow[0]["type"] == "llm_call"

    def test_detect_slow_tool(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(duration_ms=8000))
        slow = c.get_slow_operations(threshold_ms=5000)
        assert len(slow) == 1
        assert slow[0]["type"] == "tool"


# =============================================================================
# Tool Metrics Tests
# =============================================================================


class TestToolMetrics:
    def test_all_tools_summary(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="memory", duration_ms=100))
        c._on_tool_end(_make_tool_call(name="memory", duration_ms=200))
        c._on_tool_end(_make_tool_call(name="web", duration_ms=500))

        result = c.get_tool_metrics(period="all")
        assert "tools" in result
        assert len(result["tools"]) == 2

    def test_single_tool_metrics(self):
        c = MetricsCollector()
        c._on_tool_end(_make_tool_call(name="memory", duration_ms=100))
        c._on_tool_end(_make_tool_call(name="memory", duration_ms=200))
        c._on_tool_end(_make_tool_call(name="memory", success=False, duration_ms=50))

        result = c.get_tool_metrics(tool_name="memory", period="all")
        assert result["call_count"] == 3
        assert result["error_count"] == 1
        assert result["success_rate"] == pytest.approx(66.67, abs=1.0)

    def test_nonexistent_tool(self):
        c = MetricsCollector()
        result = c.get_tool_metrics(tool_name="nonexistent", period="all")
        assert "error" in result


# =============================================================================
# Clear Tests
# =============================================================================


class TestClear:
    def test_clear(self):
        c = MetricsCollector()
        c._on_llm_call_end(_make_llm_call())
        c._on_tool_end(_make_tool_call())
        c._on_embedding_call_end({
            "provider": "openai", "model": "test",
            "text_count": 1, "token_estimate": 100,
            "cost_usd": 0.0, "duration_ms": 50, "cached": False,
        })

        assert len(c._llm_calls) > 0
        c.clear()
        assert len(c._llm_calls) == 0
        assert len(c._embedding_calls) == 0
        assert len(c._tool_calls) == 0
