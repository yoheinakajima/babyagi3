"""
Data models for the metrics system.

These models represent individual metrics and aggregated summaries.
"""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class LLMCallMetric:
    """Record of a single LLM API call."""

    id: str
    timestamp: datetime

    # Source identification
    source: str  # "agent", "extraction", "summary", "retrieval"
    model: str
    thread_id: str | None = None

    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0

    # Cost (calculated)
    cost_usd: float = 0.0

    # Performance
    duration_ms: int = 0

    # Context
    stop_reason: str = ""  # "end_turn", "tool_use", etc.

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def tokens_per_second(self) -> float:
        if self.duration_ms <= 0:
            return 0.0
        return self.total_tokens / (self.duration_ms / 1000)


@dataclass
class EmbeddingCallMetric:
    """Record of a single embedding API call."""

    id: str
    timestamp: datetime

    # Provider details
    provider: str  # "openai", "voyage"
    model: str

    # Usage
    text_count: int = 1  # Number of texts embedded
    token_estimate: int = 0

    # Cost
    cost_usd: float = 0.0

    # Performance
    duration_ms: int = 0

    # Cache info
    cached: bool = False


@dataclass
class ToolCallMetric:
    """Record of a tool execution."""

    id: str
    timestamp: datetime

    tool_name: str
    thread_id: str | None = None

    # Performance
    duration_ms: int = 0

    # Status
    success: bool = True
    error_message: str | None = None

    # Data sizes (for understanding flow)
    input_size_bytes: int = 0
    output_size_bytes: int = 0


@dataclass
class SessionMetrics:
    """Aggregated metrics for a session/thread."""

    thread_id: str
    started_at: datetime
    last_activity: datetime

    # LLM calls
    llm_call_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_llm_cost_usd: float = 0.0
    avg_llm_latency_ms: float = 0.0

    # Embeddings
    embedding_call_count: int = 0
    total_embedding_cost_usd: float = 0.0

    # Tools
    tool_call_count: int = 0
    tool_success_count: int = 0
    tool_error_count: int = 0
    avg_tool_latency_ms: float = 0.0
    tools_used: dict[str, int] = field(default_factory=dict)

    # By source
    cost_by_source: dict[str, float] = field(default_factory=dict)
    calls_by_source: dict[str, int] = field(default_factory=dict)

    @property
    def total_cost_usd(self) -> float:
        return self.total_llm_cost_usd + self.total_embedding_cost_usd

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


@dataclass
class MetricsSummary:
    """High-level summary across all sessions for a time period."""

    period_start: datetime
    period_end: datetime

    # LLM totals
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_llm_cost_usd: float = 0.0
    avg_llm_latency_ms: float = 0.0

    # Embedding totals
    total_embedding_calls: int = 0
    total_embedding_cost_usd: float = 0.0
    embedding_cache_hit_rate: float = 0.0

    # Tool totals
    total_tool_calls: int = 0
    tool_success_rate: float = 100.0
    avg_tool_latency_ms: float = 0.0

    # Breakdowns
    cost_by_model: dict[str, float] = field(default_factory=dict)
    cost_by_source: dict[str, float] = field(default_factory=dict)
    calls_by_source: dict[str, int] = field(default_factory=dict)
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    errors_by_tool: dict[str, int] = field(default_factory=dict)

    # Top items
    slowest_operations: list[dict] = field(default_factory=list)

    @property
    def total_cost_usd(self) -> float:
        return self.total_llm_cost_usd + self.total_embedding_cost_usd


@dataclass
class SourceMetrics:
    """Metrics for a specific source (agent, extraction, summary, retrieval)."""

    source: str
    period_start: datetime
    period_end: datetime

    call_count: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    avg_latency_ms: float = 0.0

    # For embeddings specifically
    cache_hit_rate: float = 0.0

    @property
    def cost_pct(self) -> float:
        """Placeholder - set by collector when computing breakdowns."""
        return 0.0
