"""
Metrics Tool

Provides the agent with access to its own performance metrics.

Actions:
- summary: Overall metrics summary for a time period
- costs: Cost breakdown by source and model
- memory: Memory system specific metrics (extraction, summaries, retrieval)
- embeddings: Embedding API call statistics
- by_source: Metrics grouped by source (agent, extraction, summary, retrieval)
- tools: Tool execution statistics
- slow: Slow operations for performance analysis
- errors: Recent errors and failures
"""

from tools import tool


@tool
def get_metrics(action: str, period: str = "day", tool_name: str = None, threshold_ms: int = 5000, limit: int = 10, agent=None) -> dict:
    """Query performance metrics - costs, timing, and usage statistics.

    Use this tool to understand your own performance, costs, and identify issues.
    Available actions provide different views of the metrics data.

    Args:
        action: Metrics action - summary, costs, memory, embeddings, by_source, tools, slow, errors
        period: Time period - hour, day, week, month (default: day)
        tool_name: Filter by tool name (for 'tools' action)
        threshold_ms: Threshold in milliseconds for 'slow' action (default: 5000)
        limit: Maximum results for list actions (default: 10)
    """
    # Get metrics collector from agent
    collector = None
    if agent and hasattr(agent, 'metrics_collector'):
        collector = agent.metrics_collector

    if collector is None:
        # Try to get from global instance
        try:
            from agent import _get_metrics_collector
            collector = _get_metrics_collector()
        except (ImportError, AttributeError):
            pass

    if collector is None:
        return {
            "error": "Metrics collector not available",
            "hint": "Metrics tracking may not be initialized"
        }

    actions = {
        "summary": _get_summary,
        "costs": _get_costs,
        "memory": _get_memory_metrics,
        "embeddings": _get_embedding_metrics,
        "by_source": _get_by_source,
        "tools": _get_tool_metrics,
        "slow": _get_slow_operations,
        "errors": _get_errors,
    }

    if action not in actions:
        return {
            "error": f"Unknown action: {action}",
            "available_actions": list(actions.keys()),
            "hint": "Use 'summary' for a quick overview"
        }

    return actions[action](collector, period=period, tool_name=tool_name, threshold_ms=threshold_ms, limit=limit)


def _get_summary(collector, **kwargs) -> dict:
    """Get overall metrics summary."""
    period = kwargs.get("period", "day")
    summary = collector.get_summary(period=period)

    return {
        "period": period,
        "llm_calls": {
            "total": summary.total_llm_calls,
            "total_cost_usd": round(summary.total_llm_cost_usd, 4),
            "total_tokens": summary.total_tokens,
            "avg_duration_ms": round(summary.avg_llm_latency_ms, 1) if summary.avg_llm_latency_ms else None,
        },
        "embeddings": {
            "total": summary.total_embedding_calls,
            "total_cost_usd": round(summary.total_embedding_cost_usd, 4),
            "cache_hit_rate": round(summary.embedding_cache_hit_rate, 1),
        },
        "tools": {
            "total_executions": summary.total_tool_calls,
            "success_rate": round(summary.tool_success_rate, 1),
            "avg_duration_ms": round(summary.avg_tool_latency_ms, 1) if summary.avg_tool_latency_ms else None,
        },
        "totals": {
            "all_costs_usd": round(summary.total_llm_cost_usd + summary.total_embedding_cost_usd, 4),
        },
        "breakdown": {
            "cost_by_source": {k: round(v, 4) for k, v in summary.cost_by_source.items()},
            "cost_by_model": {k: round(v, 4) for k, v in summary.cost_by_model.items()},
        }
    }


def _get_costs(collector, **kwargs) -> dict:
    """Get cost breakdown by source and model."""
    period = kwargs.get("period", "day")
    by_source = collector.get_llm_calls_by_source(period=period)
    embed_calls = collector.get_embedding_calls(period=period)

    result = {
        "period": period,
        "by_source": {},
        "by_model": {},
        "total_cost_usd": 0.0,
    }

    model_stats = {}

    # Aggregate by source
    for source, metrics in by_source.items():
        source_cost = sum(m.cost_usd for m in metrics)
        result["by_source"][source] = {
            "calls": len(metrics),
            "cost_usd": round(source_cost, 4),
            "input_tokens": sum(m.input_tokens for m in metrics),
            "output_tokens": sum(m.output_tokens for m in metrics),
        }
        result["total_cost_usd"] += source_cost

        # Track models
        for m in metrics:
            if m.model not in model_stats:
                model_stats[m.model] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            model_stats[m.model]["calls"] += 1
            model_stats[m.model]["cost_usd"] += m.cost_usd
            model_stats[m.model]["tokens"] += m.input_tokens + m.output_tokens

    # Add embeddings as a source
    if embed_calls:
        embed_cost = sum(e.cost_usd for e in embed_calls)
        result["by_source"]["embeddings"] = {
            "calls": len(embed_calls),
            "cost_usd": round(embed_cost, 4),
            "tokens": sum(e.token_estimate for e in embed_calls),
        }
        result["total_cost_usd"] += embed_cost

        # Track embedding models
        for e in embed_calls:
            if e.model not in model_stats:
                model_stats[e.model] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
            model_stats[e.model]["calls"] += 1
            model_stats[e.model]["cost_usd"] += e.cost_usd
            model_stats[e.model]["tokens"] += e.token_estimate

    result["by_model"] = {
        model: {
            "calls": stats["calls"],
            "cost_usd": round(stats["cost_usd"], 4),
            "total_tokens": stats["tokens"],
        }
        for model, stats in model_stats.items()
    }

    result["total_cost_usd"] = round(result["total_cost_usd"], 4)
    return result


def _get_memory_metrics(collector, **kwargs) -> dict:
    """Get memory system specific metrics."""
    period = kwargs.get("period", "day")
    memory_metrics = collector.get_memory_system_metrics(period=period)

    def format_source(data: dict) -> dict:
        return {
            "calls": data.get("call_count", 0),
            "cost_usd": round(data.get("cost", 0), 4),
            "avg_latency_ms": round(data.get("avg_latency_ms", 0), 1),
        }

    return {
        "period": period,
        "extraction": format_source(memory_metrics.get("extraction", {})),
        "summary": format_source(memory_metrics.get("summary", {})),
        "retrieval": format_source(memory_metrics.get("retrieval", {})),
        "embeddings": {
            "calls": memory_metrics.get("embedding", {}).get("call_count", 0),
            "cost_usd": round(memory_metrics.get("embedding", {}).get("cost", 0), 4),
            "cache_hit_rate": round(memory_metrics.get("embedding", {}).get("cache_hit_rate", 0), 1),
        },
        "total_memory_cost_usd": round(memory_metrics.get("total_cost", 0), 4),
    }


def _get_embedding_metrics(collector, **kwargs) -> dict:
    """Get embedding API call statistics."""
    period = kwargs.get("period", "day")

    # Get embedding calls from collector
    embedding_calls = collector.get_embedding_calls(period=period)

    if not embedding_calls:
        return {
            "period": period,
            "total_calls": 0,
            "total_cost_usd": 0,
            "total_tokens": 0,
            "by_model": {},
        }

    # Aggregate stats
    total_cost = sum(e.cost_usd for e in embedding_calls)
    total_tokens = sum(e.token_estimate for e in embedding_calls)
    cached_count = sum(1 for e in embedding_calls if e.cached)

    # Group by model
    by_model = {}
    for e in embedding_calls:
        if e.model not in by_model:
            by_model[e.model] = {"calls": 0, "cost_usd": 0.0, "tokens": 0}
        by_model[e.model]["calls"] += 1
        by_model[e.model]["cost_usd"] += e.cost_usd
        by_model[e.model]["tokens"] += e.token_estimate

    return {
        "period": period,
        "total_calls": len(embedding_calls),
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": total_tokens,
        "cache_hit_rate": round(cached_count / len(embedding_calls) * 100, 1) if embedding_calls else 0,
        "by_model": {
            model: {
                "calls": stats["calls"],
                "cost_usd": round(stats["cost_usd"], 4),
                "tokens": stats["tokens"],
            }
            for model, stats in by_model.items()
        },
        "avg_tokens_per_call": round(total_tokens / len(embedding_calls), 1) if embedding_calls else 0,
    }


def _get_by_source(collector, **kwargs) -> dict:
    """Get metrics grouped by source."""
    period = kwargs.get("period", "day")
    by_source = collector.get_llm_calls_by_source(period=period)
    embed_calls = collector.get_embedding_calls(period=period)

    result = {
        "period": period,
        "sources": {},
    }

    for source, metrics in by_source.items():
        if not metrics:
            continue

        total_cost = sum(m.cost_usd for m in metrics)
        total_duration = sum(m.duration_ms for m in metrics)
        total_input = sum(m.input_tokens for m in metrics)
        total_output = sum(m.output_tokens for m in metrics)

        result["sources"][source] = {
            "calls": len(metrics),
            "cost_usd": round(total_cost, 4),
            "input_tokens": total_input,
            "output_tokens": total_output,
            "total_duration_ms": total_duration,
            "avg_duration_ms": round(total_duration / len(metrics), 1) if metrics else 0,
            "avg_cost_per_call": round(total_cost / len(metrics), 4) if metrics else 0,
        }

    # Add embeddings as a source
    if embed_calls:
        total_embed_cost = sum(e.cost_usd for e in embed_calls)
        total_embed_duration = sum(e.duration_ms for e in embed_calls if not e.cached)
        non_cached = [e for e in embed_calls if not e.cached]

        result["sources"]["embeddings"] = {
            "calls": len(embed_calls),
            "cost_usd": round(total_embed_cost, 4),
            "tokens": sum(e.token_estimate for e in embed_calls),
            "total_duration_ms": total_embed_duration,
            "avg_duration_ms": round(total_embed_duration / len(non_cached), 1) if non_cached else 0,
            "avg_cost_per_call": round(total_embed_cost / len(embed_calls), 4) if embed_calls else 0,
            "cache_hit_rate": round(sum(1 for e in embed_calls if e.cached) / len(embed_calls) * 100, 1),
        }

    return result


def _get_tool_metrics(collector, **kwargs) -> dict:
    """Get tool execution statistics."""
    period = kwargs.get("period", "day")
    tool_name = kwargs.get("tool_name")
    tool_metrics = collector.get_tool_metrics(tool_name=tool_name, period=period)

    # Handle single tool query
    if tool_name:
        if "error" in tool_metrics:
            return {
                "period": period,
                "tool_name": tool_name,
                "error": tool_metrics["error"],
            }
        return {
            "period": period,
            "tool_name": tool_name,
            "executions": tool_metrics.get("call_count", 0),
            "success_rate": round(tool_metrics.get("success_rate", 100), 1),
            "error_count": tool_metrics.get("error_count", 0),
            "avg_duration_ms": round(tool_metrics.get("avg_duration_ms", 0), 1),
            "p95_duration_ms": tool_metrics.get("p95_duration_ms", 0),
            "last_error": tool_metrics.get("last_error"),
        }

    # Handle all tools query
    tools_list = tool_metrics.get("tools", [])

    return {
        "period": period,
        "total_executions": sum(t.get("call_count", 0) for t in tools_list),
        "tools": {
            t["name"]: {
                "executions": t.get("call_count", 0),
                "avg_duration_ms": round(t.get("avg_duration_ms", 0), 1),
                "success_rate": round(t.get("success_rate", 100), 1),
            }
            for t in tools_list
        },
    }


def _get_slow_operations(collector, **kwargs) -> dict:
    """Get slow operations for performance analysis."""
    threshold_ms = kwargs.get("threshold_ms", 5000)
    limit = kwargs.get("limit", 10)
    slow_ops = collector.get_slow_operations(threshold_ms=threshold_ms, limit=limit)

    return {
        "threshold_ms": threshold_ms,
        "operations": [
            {
                "type": op.get("type"),
                "name": op.get("name"),
                "duration_ms": op.get("duration_ms"),
                "timestamp": op.get("timestamp"),
                "cost_usd": op.get("cost_usd"),
                "success": op.get("success"),
            }
            for op in slow_ops
        ],
        "count": len(slow_ops),
    }


def _get_errors(collector, **kwargs) -> dict:
    """Get recent errors and failures."""
    limit = kwargs.get("limit", 10)

    # Get errors from collector
    errors = collector.get_errors(limit=limit) if hasattr(collector, 'get_errors') else []

    return {
        "recent_errors": [
            {
                "type": e.get("type"),
                "source": e.get("source"),
                "error": e.get("error"),
                "timestamp": e.get("timestamp").isoformat() if e.get("timestamp") else None,
            }
            for e in errors
        ],
        "count": len(errors),
    }
