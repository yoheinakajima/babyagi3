"""
Memory System - Three-layer architecture for intelligent context management.

Architecture:
    1. Event Log - Immutable record of everything, tagged with deterministic metadata
    2. Graph - Entities, edges, and topics extracted from events via LLM
    3. Summary Tree - Hierarchical summaries that roll up through slices

Key Insight: Everything is organized by "slices" - tag combinations that query the event log.
A slice like "channel:email+person:john" returns all events matching those tags.
Summaries are just cached LLM-generated text for each slice.

Usage:
    from memory import MemorySystem

    memory = MemorySystem(storage_path="~/.babyagi/memory")

    # Log an event (happens automatically via agent integration)
    memory.log_event(
        type="message",
        content={"role": "user", "text": "..."},
        tags={"channel": "email", "direction": "inbound", "person": "john@example.com"}
    )

    # Assemble context for a new event (deterministic, no LLM)
    context = memory.assemble_context(
        tags={"channel": "email", "person": "john@example.com"},
        active_topics=["fundraising"]
    )

    # Quick retrieval (fast, programmatic)
    results = memory.quick_retrieve(
        query="john's email",
        entity_types=["person"],
        limit=10
    )

    # Deep retrieval (thorough, uses agent)
    answer = await memory.deep_retrieve(
        question="What's our history with John's company?",
        agent=agent
    )
"""

from .system import MemorySystem
from .models import Event, Entity, Edge, Summary, SliceKey, OwnerInfo, AgentState
from .event_log import EventLog
from .graph import Graph
from .summaries import SummaryTree
from .context import ContextAssembler
from .retrieval import QuickRetrieval, DeepRetrieval
from .extraction import ExtractionPipeline
from .integration import MemoryIntegration, setup_memory

__all__ = [
    # Main interface
    "MemorySystem",
    "setup_memory",
    "MemoryIntegration",
    # Data models
    "Event",
    "Entity",
    "Edge",
    "Summary",
    "SliceKey",
    "OwnerInfo",
    "AgentState",
    # Layers
    "EventLog",
    "Graph",
    "SummaryTree",
    # Components
    "ContextAssembler",
    "QuickRetrieval",
    "DeepRetrieval",
    "ExtractionPipeline",
]
