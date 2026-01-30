"""
Data models for the memory system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Event:
    """An immutable log entry representing something that happened."""

    id: str
    timestamp: datetime

    # Deterministic tags
    channel: str | None  # "email", "sms", "cli", "api", "web"
    direction: str  # "inbound", "outbound", "internal"
    event_type: str  # "message", "tool_call", "tool_result", "task_created", etc.

    # Associations
    task_id: str | None = None
    tool_id: str | None = None
    person_id: str | None = None  # Links to entity
    is_owner: bool = False

    # Threading
    parent_event_id: str | None = None
    conversation_id: str | None = None

    # Content
    content: str = ""
    content_embedding: list[float] | None = None
    metadata: dict | None = None

    # Extraction tracking
    extraction_status: str = "pending"  # "pending", "processing", "complete", "failed"
    extracted_at: datetime | None = None

    created_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return (
            f"Event(id={self.id[:8]}..., type={self.event_type}, "
            f"channel={self.channel}, direction={self.direction}, "
            f"content='{content_preview}')"
        )


@dataclass
class Entity:
    """A person, organization, tool, concept, or other named thing."""

    id: str
    name: str  # Canonical name
    type: str  # Clustered: "person", "org", "tool", "concept"
    type_raw: str  # Original: "venture capitalist", "Python library"

    # Additional identifiers
    aliases: list[str] = field(default_factory=list)
    description: str | None = None

    # Embedding for semantic search
    name_embedding: list[float] | None = None

    # Flags
    is_owner: bool = False
    is_self: bool = False

    # Stats
    event_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)

    # Provenance
    source_event_ids: list[str] = field(default_factory=list)

    # Links to summary tree
    summary_node_id: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"Entity(id={self.id[:8]}..., name='{self.name}', "
            f"type={self.type}, events={self.event_count})"
        )


@dataclass
class Edge:
    """A relationship between two entities."""

    id: str
    source_entity_id: str
    target_entity_id: str

    # Relationship
    relation: str  # Free-form: "invested in", "works at", "founded"
    relation_type: str | None = None  # Clustered: "financial", "professional"
    relation_embedding: list[float] | None = None

    # Properties
    is_current: bool = True
    strength: float = 0.5  # 0-1

    # Provenance
    source_event_ids: list[str] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"Edge(id={self.id[:8]}..., relation='{self.relation}', "
            f"type={self.relation_type}, strength={self.strength:.2f})"
        )


@dataclass
class Topic:
    """An extracted theme or subject area."""

    id: str
    label: str
    description: str | None = None
    keywords: list[str] = field(default_factory=list)

    # Embedding for semantic search
    embedding: list[float] | None = None

    # Hierarchy
    parent_topic_id: str | None = None

    # Stats
    event_count: int = 0
    entity_count: int = 0

    # Links to summary tree
    summary_node_id: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        keywords_str = ", ".join(self.keywords[:3]) if self.keywords else "none"
        return (
            f"Topic(id={self.id[:8]}..., label='{self.label}', "
            f"events={self.event_count}, keywords=[{keywords_str}])"
        )


@dataclass
class EventTopic:
    """Junction table linking events to topics."""

    event_id: str
    topic_id: str
    relevance: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Task:
    """A unit of work being tracked."""

    id: str
    title: str
    description: str | None = None

    # Type
    type_raw: str | None = None  # "research competitor pricing"
    type_cluster: str | None = None  # "research"
    type_embedding: list[float] | None = None

    # Status
    status: str = "pending"  # "pending", "in_progress", "completed", "failed"
    outcome: str | None = None

    # Associations
    person_id: str | None = None  # If for a specific person
    created_by_event_id: str | None = None

    # Links to summary tree
    summary_node_id: str | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    def __repr__(self) -> str:
        title_preview = self.title[:30] + "..." if len(self.title) > 30 else self.title
        return (
            f"Task(id={self.id[:8]}..., title='{title_preview}', "
            f"status={self.status})"
        )


@dataclass
class SummaryNode:
    """A node in the summary tree. Every queryable dimension has one."""

    id: str

    # Identity
    node_type: str  # "root", "channel", "tool", "entity", "entity_type",
    # "topic", "task", "task_type", "relation_type"
    key: str  # Unique: "root", "channel:email", "entity:{uuid}"
    label: str  # Human-readable: "Email", "John Smith"

    # Hierarchy
    parent_id: str | None = None

    # Summary
    summary: str = ""
    summary_embedding: list[float] | None = None
    summary_updated_at: datetime | None = None

    # Staleness
    events_since_update: int = 0

    # Stats
    event_count: int = 0
    first_event_at: datetime | None = None
    last_event_at: datetime | None = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        summary_preview = self.summary[:40] + "..." if len(self.summary) > 40 else self.summary
        return (
            f"SummaryNode(key='{self.key}', label='{self.label}', "
            f"events={self.event_count}, stale={self.events_since_update}, "
            f"summary='{summary_preview}')"
        )


@dataclass
class AgentState:
    """The agent's current state and configuration."""

    id: str

    # Identity
    name: str
    description: str | None = None
    owner_entity_id: str | None = None
    self_entity_id: str | None = None

    # Current state
    current_topics: list[str] = field(default_factory=list)  # Topic IDs
    mood: str | None = None
    focus: str | None = None
    active_tasks: list[str] = field(default_factory=list)  # Task IDs

    # Settings
    settings: dict = field(default_factory=dict)

    state_updated_at: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"AgentState(name='{self.name}', mood={self.mood}, "
            f"focus={self.focus}, topics={len(self.current_topics)}, "
            f"tasks={len(self.active_tasks)})"
        )


@dataclass
class ToolRecord:
    """Record of a tool for tracking usage and summaries."""

    id: str  # Same as tool_id in events
    name: str
    description: str = ""
    description_embedding: list[float] | None = None

    usage_count: int = 0
    last_used_at: datetime | None = None

    # Links to summary tree
    summary_node_id: str | None = None

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"ToolRecord(id='{self.id}', name='{self.name}', "
            f"usage_count={self.usage_count})"
        )


# ═══════════════════════════════════════════════════════════
# EXTRACTION MODELS (used by extraction pipeline)
# ═══════════════════════════════════════════════════════════


@dataclass
class ExtractedEntity:
    """An entity extracted from an event (before resolution)."""

    name: str
    type_raw: str
    aliases: list[str] = field(default_factory=list)
    description: str | None = None

    # Resolution
    matched_entity_id: str | None = None
    match_confidence: float = 0.0

    # Importance
    importance: float = 0.5


@dataclass
class ExtractedEdge:
    """A relationship extracted from an event."""

    source: str  # Entity name
    target: str  # Entity name
    relation: str

    is_current: bool = True
    strength: float = 0.5


@dataclass
class ExtractedTopic:
    """A topic extracted from an event."""

    label: str
    keywords: list[str] = field(default_factory=list)
    relevance: float = 1.0


@dataclass
class ExtractionResult:
    """The result of extracting from an event."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    edges: list[ExtractedEdge] = field(default_factory=list)
    topics: list[ExtractedTopic] = field(default_factory=list)
    task_type: str | None = None
    notes: str | None = None


# ═══════════════════════════════════════════════════════════
# CONTEXT MODELS (used by context assembly)
# ═══════════════════════════════════════════════════════════


@dataclass
class AssembledContext:
    """The context assembled for an agent invocation."""

    # Always included
    identity: dict = field(default_factory=dict)
    state: dict = field(default_factory=dict)
    knowledge: str = ""
    recent: dict = field(default_factory=dict)

    # Conditional
    channel: dict | None = None
    tool: dict | None = None
    task: dict | None = None
    topics: list[dict] = field(default_factory=list)
    counterparty: dict | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary, excluding None values."""
        result = {
            "identity": self.identity,
            "state": self.state,
            "knowledge": self.knowledge,
            "recent": self.recent,
        }
        if self.channel:
            result["channel"] = self.channel
        if self.tool:
            result["tool"] = self.tool
        if self.task:
            result["task"] = self.task
        if self.topics:
            result["topics"] = self.topics
        if self.counterparty:
            result["counterparty"] = self.counterparty
        return result

    def to_prompt(self) -> str:
        """Convert to a prompt string for the LLM."""
        sections = []

        # Identity
        if self.identity:
            sections.append("## Identity")
            sections.append(f"Name: {self.identity.get('name', 'Unknown')}")
            if self.identity.get("description"):
                sections.append(f"Description: {self.identity['description']}")
            if self.identity.get("owner"):
                sections.append(f"Owner: {self.identity['owner']}")
            if self.identity.get("owner_summary"):
                sections.append(f"Owner context: {self.identity['owner_summary']}")

        # Current state
        if self.state:
            sections.append("\n## Current State")
            if self.state.get("mood"):
                sections.append(f"Mood: {self.state['mood']}")
            if self.state.get("focus"):
                sections.append(f"Focus: {self.state['focus']}")
            if self.state.get("topics"):
                sections.append(f"Active topics: {', '.join(self.state['topics'])}")
            if self.state.get("active_tasks"):
                sections.append(f"Active tasks: {len(self.state['active_tasks'])} tasks")

        # Knowledge
        if self.knowledge:
            sections.append("\n## What I Know")
            sections.append(self.knowledge)

        # Channel context
        if self.channel:
            sections.append(f"\n## Channel: {self.channel.get('name', 'Unknown')}")
            if self.channel.get("summary"):
                sections.append(self.channel["summary"])

        # Tool context
        if self.tool:
            sections.append(f"\n## Tool: {self.tool.get('name', 'Unknown')}")
            if self.tool.get("summary"):
                sections.append(self.tool["summary"])

        # Task context
        if self.task:
            task_obj = self.task.get("task")
            if task_obj:
                sections.append(f"\n## Current Task: {task_obj.title if hasattr(task_obj, 'title') else task_obj.get('title', 'Unknown')}")
            if self.task.get("summary"):
                sections.append(self.task["summary"])

        # Topics
        if self.topics:
            sections.append("\n## Relevant Topics")
            for topic in self.topics:
                sections.append(f"- **{topic.get('label', 'Unknown')}**: {topic.get('summary', '')}")

        # Counterparty
        if self.counterparty:
            entity = self.counterparty.get("entity")
            name = entity.name if hasattr(entity, "name") else entity.get("name", "Unknown") if entity else "Unknown"
            sections.append(f"\n## About {name}")
            if self.counterparty.get("summary"):
                sections.append(self.counterparty["summary"])
            edges = self.counterparty.get("edges", [])
            if edges:
                sections.append("Relationships:")
                for edge in edges[:5]:
                    if hasattr(edge, "relation"):
                        sections.append(f"  - {edge.relation}")
                    elif isinstance(edge, dict):
                        sections.append(f"  - {edge.get('relation', 'unknown')}")

        # Recent
        if self.recent:
            sections.append("\n## Recent Activity")
            if self.recent.get("summary"):
                sections.append(self.recent["summary"])

        return "\n".join(sections)
