"""
Metrics collector - central hub for gathering and querying metrics.

Subscribes to events from instrumented clients and provides
aggregation/query methods for analysis.
"""

from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from .models import (
    LLMCallMetric,
    EmbeddingCallMetric,
    ToolCallMetric,
    SessionMetrics,
    MetricsSummary,
)
from .costs import format_cost


class MetricsCollector:
    """
    Collects and stores metrics from instrumented API clients.

    Subscribes to events emitted by InstrumentedAnthropic/OpenAI and
    provides query methods for aggregation and analysis.

    Usage:
        collector = MetricsCollector(memory_store)
        collector.attach(agent)  # Subscribe to events

        # Later...
        summary = collector.get_summary(period="day")
        by_source = collector.get_metrics_by_source()
    """

    def __init__(self, store=None):
        """
        Initialize the collector.

        Args:
            store: MemoryStore instance for persistence (optional).
                   If None, metrics are kept in memory only.
        """
        self.store = store

        # In-memory storage (always used, store is for persistence)
        self._llm_calls: list[LLMCallMetric] = []
        self._embedding_calls: list[EmbeddingCallMetric] = []
        self._tool_calls: list[ToolCallMetric] = []

        # Track active session
        self._current_thread_id: str | None = None

    def attach(self, agent):
        """
        Attach to an agent's event system.

        Subscribes to llm_call_end, embedding_call_end, tool_start, tool_end.

        Args:
            agent: Agent instance with EventEmitter (has .on() method)
        """
        agent.on("llm_call_end", self._on_llm_call_end)
        agent.on("embedding_call_end", self._on_embedding_call_end)
        agent.on("tool_start", self._on_tool_start)
        agent.on("tool_end", self._on_tool_end)

    # ═══════════════════════════════════════════════════════════
    # EVENT HANDLERS
    # ═══════════════════════════════════════════════════════════

    def _on_llm_call_end(self, event: dict):
        """Handle LLM call completion event."""
        metric = LLMCallMetric(
            id=str(uuid4()),
            timestamp=datetime.now(),
            source=event.get("source", "unknown"),
            model=event.get("model", "unknown"),
            thread_id=self._current_thread_id,
            input_tokens=event.get("input_tokens", 0),
            output_tokens=event.get("output_tokens", 0),
            cost_usd=event.get("cost_usd", 0.0),
            duration_ms=event.get("duration_ms", 0),
            stop_reason=event.get("stop_reason", ""),
        )
        self._llm_calls.append(metric)

        # Persist if store is available
        if self.store:
            try:
                self.store.record_llm_call(metric)
            except Exception:
                pass  # Don't crash on persistence failure

    def _on_embedding_call_end(self, event: dict):
        """Handle embedding call completion event."""
        metric = EmbeddingCallMetric(
            id=str(uuid4()),
            timestamp=datetime.now(),
            provider=event.get("provider", "unknown"),
            model=event.get("model", "unknown"),
            text_count=event.get("text_count", 1),
            token_estimate=event.get("token_estimate", 0),
            cost_usd=event.get("cost_usd", 0.0),
            duration_ms=event.get("duration_ms", 0),
            cached=event.get("cached", False),
        )
        self._embedding_calls.append(metric)

        # Persist if store is available
        if self.store:
            try:
                self.store.record_embedding_call(metric)
            except Exception:
                pass

    def _on_tool_start(self, event: dict):
        """Track tool start for duration calculation."""
        # Tool duration is already calculated by the agent
        pass

    def _on_tool_end(self, event: dict):
        """Handle tool completion event."""
        result = event.get("result", {})
        is_error = False
        error_msg = None

        if isinstance(result, dict):
            if "error" in result:
                is_error = True
                error_msg = str(result["error"])
            elif result.get("status") == "error":
                is_error = True
                error_msg = result.get("message", "Unknown error")

        metric = ToolCallMetric(
            id=str(uuid4()),
            timestamp=datetime.now(),
            tool_name=event.get("name", "unknown"),
            thread_id=self._current_thread_id,
            duration_ms=event.get("duration_ms", 0),
            success=not is_error,
            error_message=error_msg,
            input_size_bytes=len(str(event.get("input", ""))),
            output_size_bytes=len(str(result)),
        )
        self._tool_calls.append(metric)

    def set_thread_id(self, thread_id: str):
        """Set the current thread/session ID for metrics."""
        self._current_thread_id = thread_id

    # ═══════════════════════════════════════════════════════════
    # QUERY METHODS
    # ═══════════════════════════════════════════════════════════

    def _get_time_range(self, period: str) -> tuple[datetime, datetime]:
        """Convert period string to datetime range."""
        now = datetime.now()
        if period == "hour":
            start = now - timedelta(hours=1)
        elif period == "day":
            start = now - timedelta(days=1)
        elif period == "week":
            start = now - timedelta(weeks=1)
        elif period == "month":
            start = now - timedelta(days=30)
        else:  # "all"
            start = datetime.min
        return start, now

    def _filter_by_period(self, items: list, period: str) -> list:
        """Filter metrics by time period."""
        start, end = self._get_time_range(period)
        return [m for m in items if start <= m.timestamp <= end]

    def get_summary(self, period: str = "day") -> MetricsSummary:
        """
        Get aggregated metrics summary.

        Args:
            period: Time period ("hour", "day", "week", "month", "all")

        Returns:
            MetricsSummary with aggregated data
        """
        start, end = self._get_time_range(period)

        llm_calls = self._filter_by_period(self._llm_calls, period)
        embed_calls = self._filter_by_period(self._embedding_calls, period)
        tool_calls = self._filter_by_period(self._tool_calls, period)

        # LLM metrics
        total_llm_cost = sum(c.cost_usd for c in llm_calls)
        total_tokens = sum(c.total_tokens for c in llm_calls)
        avg_llm_latency = (
            sum(c.duration_ms for c in llm_calls) / len(llm_calls)
            if llm_calls else 0
        )

        # Embedding metrics
        total_embed_cost = sum(c.cost_usd for c in embed_calls)
        cached_count = sum(1 for c in embed_calls if c.cached)
        cache_hit_rate = (cached_count / len(embed_calls) * 100) if embed_calls else 0

        # Tool metrics
        successful_tools = sum(1 for t in tool_calls if t.success)
        tool_success_rate = (
            (successful_tools / len(tool_calls) * 100) if tool_calls else 100
        )
        avg_tool_latency = (
            sum(t.duration_ms for t in tool_calls) / len(tool_calls)
            if tool_calls else 0
        )

        # Cost by model
        cost_by_model: dict[str, float] = {}
        for c in llm_calls:
            cost_by_model[c.model] = cost_by_model.get(c.model, 0) + c.cost_usd

        # Cost/calls by source
        cost_by_source: dict[str, float] = {}
        calls_by_source: dict[str, int] = {}
        for c in llm_calls:
            cost_by_source[c.source] = cost_by_source.get(c.source, 0) + c.cost_usd
            calls_by_source[c.source] = calls_by_source.get(c.source, 0) + 1

        # Calls by tool
        calls_by_tool: dict[str, int] = {}
        errors_by_tool: dict[str, int] = {}
        for t in tool_calls:
            calls_by_tool[t.tool_name] = calls_by_tool.get(t.tool_name, 0) + 1
            if not t.success:
                errors_by_tool[t.tool_name] = errors_by_tool.get(t.tool_name, 0) + 1

        return MetricsSummary(
            period_start=start,
            period_end=end,
            total_llm_calls=len(llm_calls),
            total_tokens=total_tokens,
            total_llm_cost_usd=total_llm_cost,
            avg_llm_latency_ms=avg_llm_latency,
            total_embedding_calls=len(embed_calls),
            total_embedding_cost_usd=total_embed_cost,
            embedding_cache_hit_rate=cache_hit_rate,
            total_tool_calls=len(tool_calls),
            tool_success_rate=tool_success_rate,
            avg_tool_latency_ms=avg_tool_latency,
            cost_by_model=cost_by_model,
            cost_by_source=cost_by_source,
            calls_by_source=calls_by_source,
            calls_by_tool=calls_by_tool,
            errors_by_tool=errors_by_tool,
        )

    def get_metrics_by_source(self, period: str = "day") -> dict[str, dict]:
        """
        Get metrics breakdown by source.

        Args:
            period: Time period

        Returns:
            Dict mapping source name to metrics
        """
        llm_calls = self._filter_by_period(self._llm_calls, period)
        embed_calls = self._filter_by_period(self._embedding_calls, period)

        # Aggregate by source
        sources: dict[str, dict] = {}

        for c in llm_calls:
            if c.source not in sources:
                sources[c.source] = {
                    "call_count": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "total_latency_ms": 0,
                }
            sources[c.source]["call_count"] += 1
            sources[c.source]["total_tokens"] += c.total_tokens
            sources[c.source]["cost"] += c.cost_usd
            sources[c.source]["total_latency_ms"] += c.duration_ms

        # Calculate averages and percentages
        total_cost = sum(s["cost"] for s in sources.values())

        for source, stats in sources.items():
            stats["avg_latency_ms"] = (
                stats["total_latency_ms"] / stats["call_count"]
                if stats["call_count"] > 0 else 0
            )
            stats["cost_pct"] = (
                (stats["cost"] / total_cost * 100) if total_cost > 0 else 0
            )
            del stats["total_latency_ms"]

        # Add embedding as a source
        if embed_calls:
            embed_cost = sum(c.cost_usd for c in embed_calls)
            embed_latency = sum(c.duration_ms for c in embed_calls)
            cached_count = sum(1 for c in embed_calls if c.cached)

            sources["embedding"] = {
                "call_count": len(embed_calls),
                "total_tokens": sum(c.token_estimate for c in embed_calls),
                "cost": embed_cost,
                "avg_latency_ms": embed_latency / len(embed_calls),
                "cost_pct": (embed_cost / (total_cost + embed_cost) * 100)
                    if (total_cost + embed_cost) > 0 else 0,
                "cache_hit_rate": cached_count / len(embed_calls) * 100,
            }

        return sources

    def get_memory_system_metrics(self, period: str = "day") -> dict:
        """
        Get metrics specifically for memory system components.

        Returns breakdown for extraction, summary, retrieval, embedding.
        """
        llm_calls = self._filter_by_period(self._llm_calls, period)
        embed_calls = self._filter_by_period(self._embedding_calls, period)

        def get_source_stats(source: str) -> dict:
            calls = [c for c in llm_calls if c.source == source]
            return {
                "call_count": len(calls),
                "cost": sum(c.cost_usd for c in calls),
                "avg_latency_ms": (
                    sum(c.duration_ms for c in calls) / len(calls)
                    if calls else 0
                ),
            }

        # Embedding stats
        cached = sum(1 for c in embed_calls if c.cached)
        embed_stats = {
            "call_count": len(embed_calls),
            "cost": sum(c.cost_usd for c in embed_calls),
            "cache_hit_rate": (cached / len(embed_calls) * 100) if embed_calls else 0,
        }

        extraction = get_source_stats("extraction")
        summary = get_source_stats("summary")
        retrieval = get_source_stats("retrieval")

        total_cost = (
            extraction["cost"] + summary["cost"] +
            retrieval["cost"] + embed_stats["cost"]
        )

        return {
            "extraction": extraction,
            "summary": summary,
            "retrieval": retrieval,
            "embedding": embed_stats,
            "total_cost": total_cost,
        }

    def get_embedding_metrics(self, period: str = "day") -> dict:
        """Get detailed embedding metrics."""
        embed_calls = self._filter_by_period(self._embedding_calls, period)

        if not embed_calls:
            return {
                "call_count": 0,
                "text_count": 0,
                "cache_hit_rate": 0,
                "cost": 0,
                "avg_latency_ms": 0,
                "by_model": {},
            }

        cached = sum(1 for c in embed_calls if c.cached)
        total_latency = sum(c.duration_ms for c in embed_calls if not c.cached)
        non_cached = len([c for c in embed_calls if not c.cached])

        # By model
        by_model: dict[str, dict] = {}
        for c in embed_calls:
            if c.model not in by_model:
                by_model[c.model] = {"count": 0, "cost": 0}
            by_model[c.model]["count"] += 1
            by_model[c.model]["cost"] += c.cost_usd

        return {
            "call_count": len(embed_calls),
            "text_count": sum(c.text_count for c in embed_calls),
            "cache_hit_rate": cached / len(embed_calls) * 100,
            "cost": sum(c.cost_usd for c in embed_calls),
            "avg_latency_ms": total_latency / non_cached if non_cached else 0,
            "by_model": by_model,
        }

    def get_tool_metrics(
        self,
        tool_name: str | None = None,
        period: str = "day",
    ) -> dict:
        """Get tool performance metrics."""
        tool_calls = self._filter_by_period(self._tool_calls, period)

        if tool_name:
            calls = [t for t in tool_calls if t.tool_name == tool_name]
            if not calls:
                return {"error": f"No calls for tool: {tool_name}"}

            successful = sum(1 for t in calls if t.success)
            latencies = sorted(t.duration_ms for t in calls)

            return {
                "call_count": len(calls),
                "success_rate": successful / len(calls) * 100,
                "error_count": len(calls) - successful,
                "avg_duration_ms": sum(latencies) / len(calls),
                "p95_duration_ms": latencies[int(len(latencies) * 0.95)] if latencies else 0,
                "last_error": next(
                    (t.error_message for t in reversed(calls) if t.error_message),
                    None
                ),
            }

        # All tools summary
        tools: dict[str, dict] = {}
        for t in tool_calls:
            if t.tool_name not in tools:
                tools[t.tool_name] = {
                    "name": t.tool_name,
                    "call_count": 0,
                    "success_count": 0,
                    "total_duration_ms": 0,
                }
            tools[t.tool_name]["call_count"] += 1
            if t.success:
                tools[t.tool_name]["success_count"] += 1
            tools[t.tool_name]["total_duration_ms"] += t.duration_ms

        result = []
        for name, stats in tools.items():
            result.append({
                "name": name,
                "call_count": stats["call_count"],
                "success_rate": stats["success_count"] / stats["call_count"] * 100,
                "avg_duration_ms": stats["total_duration_ms"] / stats["call_count"],
            })

        # Sort by call count
        result.sort(key=lambda x: x["call_count"], reverse=True)

        return {"tools": result}

    def get_slow_operations(
        self,
        threshold_ms: int = 5000,
        limit: int = 10,
    ) -> list[dict]:
        """Find slow operations across LLM calls and tools."""
        slow = []

        # Slow LLM calls
        for c in self._llm_calls:
            if c.duration_ms >= threshold_ms:
                slow.append({
                    "type": "llm_call",
                    "name": f"{c.source}:{c.model}",
                    "duration_ms": c.duration_ms,
                    "timestamp": c.timestamp.isoformat(),
                    "cost_usd": c.cost_usd,
                })

        # Slow tool calls
        for t in self._tool_calls:
            if t.duration_ms >= threshold_ms:
                slow.append({
                    "type": "tool",
                    "name": t.tool_name,
                    "duration_ms": t.duration_ms,
                    "timestamp": t.timestamp.isoformat(),
                    "success": t.success,
                })

        # Sort by duration descending
        slow.sort(key=lambda x: x["duration_ms"], reverse=True)

        return slow[:limit]

    def get_recent_errors(self, limit: int = 10) -> list[dict]:
        """Get recent tool errors."""
        errors = [
            {
                "tool_name": t.tool_name,
                "error_message": t.error_message,
                "timestamp": t.timestamp.isoformat(),
                "duration_ms": t.duration_ms,
            }
            for t in self._tool_calls
            if not t.success and t.error_message
        ]

        # Most recent first
        errors.sort(key=lambda x: x["timestamp"], reverse=True)

        return errors[:limit]

    def get_error_counts(self) -> dict[str, int]:
        """Get error counts by tool."""
        counts: dict[str, int] = {}
        for t in self._tool_calls:
            if not t.success:
                counts[t.tool_name] = counts.get(t.tool_name, 0) + 1
        return counts

    def get_session_metrics(self, thread_id: str) -> SessionMetrics:
        """Get metrics for a specific session/thread."""
        llm_calls = [c for c in self._llm_calls if c.thread_id == thread_id]
        embed_calls = [c for c in self._embedding_calls]  # Embeddings don't have thread_id
        tool_calls = [t for t in self._tool_calls if t.thread_id == thread_id]

        if not llm_calls and not tool_calls:
            return SessionMetrics(
                thread_id=thread_id,
                started_at=datetime.now(),
                last_activity=datetime.now(),
            )

        all_times = (
            [c.timestamp for c in llm_calls] +
            [t.timestamp for t in tool_calls]
        )

        # Tools used
        tools_used: dict[str, int] = {}
        for t in tool_calls:
            tools_used[t.tool_name] = tools_used.get(t.tool_name, 0) + 1

        # Cost by source
        cost_by_source: dict[str, float] = {}
        calls_by_source: dict[str, int] = {}
        for c in llm_calls:
            cost_by_source[c.source] = cost_by_source.get(c.source, 0) + c.cost_usd
            calls_by_source[c.source] = calls_by_source.get(c.source, 0) + 1

        successful_tools = sum(1 for t in tool_calls if t.success)

        return SessionMetrics(
            thread_id=thread_id,
            started_at=min(all_times) if all_times else datetime.now(),
            last_activity=max(all_times) if all_times else datetime.now(),
            llm_call_count=len(llm_calls),
            total_input_tokens=sum(c.input_tokens for c in llm_calls),
            total_output_tokens=sum(c.output_tokens for c in llm_calls),
            total_llm_cost_usd=sum(c.cost_usd for c in llm_calls),
            avg_llm_latency_ms=(
                sum(c.duration_ms for c in llm_calls) / len(llm_calls)
                if llm_calls else 0
            ),
            tool_call_count=len(tool_calls),
            tool_success_count=successful_tools,
            tool_error_count=len(tool_calls) - successful_tools,
            avg_tool_latency_ms=(
                sum(t.duration_ms for t in tool_calls) / len(tool_calls)
                if tool_calls else 0
            ),
            tools_used=tools_used,
            cost_by_source=cost_by_source,
            calls_by_source=calls_by_source,
        )

    def get_embedding_calls(self, period: str = "day") -> list[EmbeddingCallMetric]:
        """Get embedding call metrics for a period."""
        return self._filter_by_period(self._embedding_calls, period)

    def get_llm_calls(self, period: str = "day") -> list[LLMCallMetric]:
        """Get LLM call metrics for a period."""
        return self._filter_by_period(self._llm_calls, period)

    def get_llm_calls_by_source(self, period: str = "day") -> dict[str, list[LLMCallMetric]]:
        """Get LLM calls grouped by source."""
        calls = self._filter_by_period(self._llm_calls, period)
        by_source: dict[str, list[LLMCallMetric]] = {}
        for c in calls:
            if c.source not in by_source:
                by_source[c.source] = []
            by_source[c.source].append(c)
        return by_source

    def get_errors(self, limit: int = 10) -> list[dict]:
        """Get recent errors from all sources."""
        errors = []

        # Tool errors
        for t in self._tool_calls:
            if not t.success and t.error_message:
                errors.append({
                    "type": "tool",
                    "source": t.tool_name,
                    "error": t.error_message,
                    "timestamp": t.timestamp,
                })

        # Sort by timestamp descending
        errors.sort(key=lambda x: x["timestamp"], reverse=True)
        return errors[:limit]

    def clear(self):
        """Clear all in-memory metrics."""
        self._llm_calls.clear()
        self._embedding_calls.clear()
        self._tool_calls.clear()
