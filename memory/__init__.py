"""
Memory System for BabyAGI

A graph-based memory system with three layers:
1. Raw Event Log - Immutable record of everything
2. Extracted Graph - Entities, edges, topics
3. Summary Tree - Pre-computed summaries that roll up

Usage:
    from memory import Memory

    memory = Memory(store_path="~/.babyagi/memory")

    # Log an event
    event = memory.log_event(
        content="Hello from John",
        channel="email",
        direction="inbound",
        person_id=john_entity_id,
    )

    # Assemble context
    context = memory.assemble_context(event)

    # Quick retrieval
    entities = memory.search_entities("venture capitalist")
    edges = memory.search_edges("investment", entity_id=john_id)
"""

from .models import (
    Event,
    Entity,
    Edge,
    Fact,
    Topic,
    Task,
    SummaryNode,
    AgentState,
    ToolRecord,
    EventTopic,
    Learning,
    ExtractedFeedback,
)
from .store import MemoryStore
from .context import assemble_context
from .retrieval import QuickRetrieval, DeepRetrievalAgent
from .extraction import ExtractionPipeline
from .summaries import SummaryManager
from .embeddings import get_embedding, get_embeddings
from .learning import (
    FeedbackExtractor,
    ObjectiveEvaluator,
    LearningRetriever,
    PreferenceSummarizer,
    ensure_user_preferences_node,
)

__all__ = [
    # Models
    "Event",
    "Entity",
    "Edge",
    "Fact",
    "Topic",
    "Task",
    "SummaryNode",
    "AgentState",
    "ToolRecord",
    "EventTopic",
    "Learning",
    "ExtractedFeedback",
    # Core
    "MemoryStore",
    "assemble_context",
    "QuickRetrieval",
    "DeepRetrievalAgent",
    "ExtractionPipeline",
    "SummaryManager",
    # Self-Improvement
    "FeedbackExtractor",
    "ObjectiveEvaluator",
    "LearningRetriever",
    "PreferenceSummarizer",
    "ensure_user_preferences_node",
    # Utilities
    "get_embedding",
    "get_embeddings",
]


class Memory:
    """
    Main facade for the memory system.

    Combines storage, retrieval, extraction, and context assembly.
    """

    def __init__(self, store_path: str = "~/.babyagi/memory", embedding_model: str = "text-embedding-3-small"):
        self.store = MemoryStore(store_path)
        self.retrieval = QuickRetrieval(self.store)
        self.extraction = ExtractionPipeline(self.store)
        self.summaries = SummaryManager(self.store)
        self.embedding_model = embedding_model

        # Initialize database
        self.store.initialize()

    # ═══════════════════════════════════════════════════════════
    # EVENT LOGGING
    # ═══════════════════════════════════════════════════════════

    def log_event(
        self,
        content: str,
        event_type: str = "message",
        channel: str | None = None,
        direction: str = "internal",
        task_id: str | None = None,
        tool_id: str | None = None,
        person_id: str | None = None,
        is_owner: bool = False,
        parent_event_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict | None = None,
    ) -> Event:
        """Log an event to the memory system."""
        return self.store.create_event(
            content=content,
            event_type=event_type,
            channel=channel,
            direction=direction,
            task_id=task_id,
            tool_id=tool_id,
            person_id=person_id,
            is_owner=is_owner,
            parent_event_id=parent_event_id,
            conversation_id=conversation_id,
            metadata=metadata,
        )

    # ═══════════════════════════════════════════════════════════
    # CONTEXT ASSEMBLY
    # ═══════════════════════════════════════════════════════════

    def assemble_context(self, event: Event | None = None) -> dict:
        """Assemble context from pre-computed summaries."""
        state = self.store.get_agent_state()
        return assemble_context(event, state, self.store, self.retrieval)

    # ═══════════════════════════════════════════════════════════
    # QUICK RETRIEVAL
    # ═══════════════════════════════════════════════════════════

    def get_summary(self, key: str) -> SummaryNode | None:
        """Get a summary by key (e.g., 'entity:{id}', 'channel:email')."""
        return self.retrieval.get_summary(key)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self.store.get_entity(entity_id)

    def get_edges(self, entity_id: str, direction: str = "both") -> list[Edge]:
        """Get edges for an entity."""
        return self.retrieval.get_edges(entity_id, direction)

    def find_events(self, **filters) -> list[Event]:
        """Find events by filters."""
        return self.retrieval.find_events(**filters)

    def find_entities(self, query: str = None, type: str = None, limit: int = 10) -> list[Entity]:
        """Find entities by name or type."""
        return self.retrieval.find_entities(query, type, limit)

    # ═══════════════════════════════════════════════════════════
    # SEMANTIC SEARCH
    # ═══════════════════════════════════════════════════════════

    def search_events(self, query: str, limit: int = 10) -> list[Event]:
        """Semantic search over events."""
        return self.retrieval.search_events(query, limit)

    def search_entities(self, query: str, type: str = None, limit: int = 10) -> list[Entity]:
        """Semantic search over entities."""
        return self.retrieval.search_entities(query, type, limit)

    def search_edges(self, query: str, entity_id: str = None, limit: int = 10) -> list[Edge]:
        """Semantic search over relationships."""
        return self.retrieval.search_edges(query, entity_id, limit)

    def search_topics(self, query: str, limit: int = 10) -> list[Topic]:
        """Semantic search over topics."""
        return self.retrieval.search_topics(query, limit)

    def search_tasks(self, query: str, status: str = None, limit: int = 10) -> list[Task]:
        """Semantic search over tasks."""
        return self.retrieval.search_tasks(query, status, limit)

    def search_facts(self, query: str, fact_type: str = None, source_type: str = None, limit: int = 10) -> list[Fact]:
        """Semantic search over facts."""
        return self.retrieval.search_facts(query, fact_type, source_type, limit)

    # ═══════════════════════════════════════════════════════════
    # GRAPH NAVIGATION
    # ═══════════════════════════════════════════════════════════

    def get_sources(self, node_key: str, limit: int = 20) -> list[Event]:
        """Get source events for a summary."""
        return self.retrieval.get_sources(node_key, limit)

    def get_children(self, node_id: str) -> list[SummaryNode]:
        """Get child summary nodes."""
        return self.store.get_children(node_id)

    def get_parent(self, node_id: str) -> SummaryNode | None:
        """Get parent summary node."""
        return self.store.get_parent(node_id)

    # ═══════════════════════════════════════════════════════════
    # EXTRACTION & SUMMARIES
    # ═══════════════════════════════════════════════════════════

    async def extract_from_event(self, event: Event) -> None:
        """Extract entities, edges, and topics from an event."""
        await self.extraction.extract(event)

    async def refresh_stale_summaries(self, threshold: int = 10) -> int:
        """Refresh summaries that have accumulated new events."""
        return await self.summaries.refresh_stale(threshold)

    # ═══════════════════════════════════════════════════════════
    # DEEP RETRIEVAL
    # ═══════════════════════════════════════════════════════════

    async def deep_retrieve(self, query: str, context: dict = None) -> dict:
        """Invoke deep retrieval agent for complex queries."""
        agent = DeepRetrievalAgent(self.store, self.retrieval)
        return await agent.run(query, context)

    # ═══════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════

    def get_agent_state(self) -> AgentState:
        """Get the current agent state."""
        return self.store.get_agent_state()

    def update_agent_state(self, **updates) -> AgentState:
        """Update the agent state."""
        return self.store.update_agent_state(**updates)

    # ═══════════════════════════════════════════════════════════
    # SELF-IMPROVEMENT (Learnings)
    # ═══════════════════════════════════════════════════════════

    def get_learnings(
        self,
        tool_id: str = None,
        objective_type: str = None,
        sentiment: str = None,
        limit: int = 20,
    ) -> list[Learning]:
        """Get learnings, optionally filtered."""
        return self.store.find_learnings(
            tool_id=tool_id,
            objective_type=objective_type,
            sentiment=sentiment,
            limit=limit,
        )

    def search_learnings(self, query: str, limit: int = 10) -> list[Learning]:
        """Semantic search over learnings."""
        embedding = get_embedding(query)
        return self.store.search_learnings(embedding=embedding, limit=limit)

    def get_user_preferences(self) -> str:
        """Get the current user preferences summary."""
        retriever = LearningRetriever(self.store)
        return retriever.get_user_preferences()

    def get_learning_stats(self) -> dict:
        """Get statistics about learnings."""
        return self.store.get_learning_stats()
