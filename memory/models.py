"""
Core data models for the memory system.

Design principles:
- Immutable events, mutable summaries
- Tags are the universal organizer
- SliceKey is the key abstraction for querying
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import hashlib
import json


def _now() -> str:
    """ISO timestamp for now."""
    return datetime.utcnow().isoformat() + "Z"


def _generate_id(prefix: str, content: str) -> str:
    """Generate a short deterministic ID."""
    h = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{h}"


@dataclass
class SliceKey:
    """
    A slice is a query over events/entities defined by tag combinations.

    Examples:
        SliceKey({"channel": "email"})  # All email events
        SliceKey({"person": "john"})    # All events about john
        SliceKey({"channel": "email", "person": "john"})  # Intersection

    The key representation is sorted and canonical:
        "channel:email"
        "channel:email+person:john"
    """

    tags: dict[str, str]

    def __post_init__(self):
        # Ensure tags are lowercase for consistency
        self.tags = {k.lower(): v.lower() for k, v in self.tags.items()}

    @property
    def key(self) -> str:
        """Canonical string representation."""
        if not self.tags:
            return "*"  # Root - all events
        parts = [f"{k}:{v}" for k, v in sorted(self.tags.items())]
        return "+".join(parts)

    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        if isinstance(other, SliceKey):
            return self.key == other.key
        return False

    def __str__(self):
        return self.key

    def __repr__(self):
        return f"SliceKey({self.key})"

    @classmethod
    def root(cls) -> "SliceKey":
        """The root slice - matches all events."""
        return cls({})

    @classmethod
    def from_key(cls, key: str) -> "SliceKey":
        """Parse a canonical key string back to SliceKey."""
        if key == "*":
            return cls({})
        tags = {}
        for part in key.split("+"):
            k, v = part.split(":", 1)
            tags[k] = v
        return cls(tags)

    def parents(self) -> list["SliceKey"]:
        """
        Get all parent slices (less specific).

        For "channel:email+person:john", parents are:
        - "channel:email"
        - "person:john"
        - "*" (root)
        """
        if not self.tags:
            return []  # Root has no parents

        parents = []
        keys = list(self.tags.keys())

        # Single-tag parents
        if len(keys) > 1:
            for k in keys:
                parents.append(SliceKey({k: self.tags[k]}))

        # Root is always a parent
        parents.append(SliceKey.root())

        return parents

    def matches(self, event_tags: dict[str, str]) -> bool:
        """Check if an event's tags match this slice."""
        event_tags_lower = {k.lower(): v.lower() for k, v in event_tags.items()}
        for k, v in self.tags.items():
            if event_tags_lower.get(k) != v:
                return False
        return True


@dataclass
class Event:
    """
    Immutable record of something that happened.

    Events are the source of truth. Everything else (graph, summaries)
    is derived from events.

    Types:
        - message_in: Incoming message from any channel
        - message_out: Outgoing message to any channel
        - tool_call: Agent called a tool
        - tool_result: Tool returned a result
        - task_start: Background task started
        - task_end: Background task completed
        - extraction: Entity/edge extracted (meta-event)
    """

    id: str
    type: str
    content: Any  # Flexible - depends on type
    tags: dict[str, str]  # {channel, direction, person, task, tool, is_owner, ...}
    timestamp: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(**d)

    @classmethod
    def create(cls, type: str, content: Any, tags: dict[str, str]) -> "Event":
        """Create a new event with auto-generated ID."""
        id_content = json.dumps({"type": type, "content": content, "ts": _now()})
        return cls(
            id=_generate_id("evt", id_content),
            type=type,
            content=content,
            tags=tags,
        )


@dataclass
class Entity:
    """
    A node in the knowledge graph - a person, org, topic, concept, etc.

    Entities are extracted from events by LLM. Types are free-form and
    get clustered automatically (e.g., "venture capitalist" → "person").
    """

    id: str
    type: str  # Free-form, gets clustered: "person", "organization", "topic", etc.
    name: str  # Display name
    aliases: list[str] = field(default_factory=list)  # Other names for this entity
    attributes: dict[str, Any] = field(default_factory=dict)  # Free-form metadata
    source_events: list[str] = field(default_factory=list)  # Event IDs that mention this
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "aliases": self.aliases,
            "attributes": self.attributes,
            "source_events": self.source_events,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Entity":
        return cls(**d)

    @classmethod
    def create(cls, type: str, name: str, **kwargs) -> "Entity":
        """Create a new entity with auto-generated ID."""
        id_content = f"{type}:{name.lower()}"
        return cls(
            id=_generate_id("ent", id_content),
            type=type,
            name=name,
            **kwargs,
        )


@dataclass
class Edge:
    """
    A relationship between two entities.

    Edge types are free-form and get clustered automatically
    (e.g., "invested in" → "financial").
    """

    id: str
    source_id: str  # Entity ID
    target_id: str  # Entity ID
    type: str  # Free-form: "knows", "works_at", "invested_in", "mentions", etc.
    attributes: dict[str, Any] = field(default_factory=dict)
    source_events: list[str] = field(default_factory=list)  # Event IDs that establish this
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type,
            "attributes": self.attributes,
            "source_events": self.source_events,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Edge":
        return cls(**d)

    @classmethod
    def create(cls, source_id: str, target_id: str, type: str, **kwargs) -> "Edge":
        """Create a new edge with auto-generated ID."""
        id_content = f"{source_id}:{type}:{target_id}"
        return cls(
            id=_generate_id("edg", id_content),
            source_id=source_id,
            target_id=target_id,
            type=type,
            **kwargs,
        )


@dataclass
class Summary:
    """
    Cached LLM-generated summary for a slice of events.

    Summaries form a tree through the slice hierarchy. When a leaf
    summary updates, its parents get marked stale and refresh
    on schedule or on demand.
    """

    slice_key: str  # Canonical key like "channel:email+person:john"
    text: str  # The actual summary
    event_count: int  # How many events this covers
    last_event_id: str  # Most recent event included
    stale: bool = False  # Needs refresh?
    staleness_score: float = 0.0  # How stale (0-1, for prioritization)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> dict:
        return {
            "slice_key": self.slice_key,
            "text": self.text,
            "event_count": self.event_count,
            "last_event_id": self.last_event_id,
            "stale": self.stale,
            "staleness_score": self.staleness_score,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Summary":
        return cls(**d)

    def mark_stale(self, score: float = 0.5):
        """Mark this summary as needing refresh."""
        self.stale = True
        self.staleness_score = max(self.staleness_score, score)

    def refresh(self, text: str, event_count: int, last_event_id: str):
        """Update this summary with fresh content."""
        self.text = text
        self.event_count = event_count
        self.last_event_id = last_event_id
        self.stale = False
        self.staleness_score = 0.0
        self.updated_at = _now()


@dataclass
class OwnerInfo:
    """Static information about the agent's owner."""

    id: str
    name: str
    email: str
    contacts: dict[str, str] = field(default_factory=dict)
    preferences: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    """Current state of the agent - mutable, frequently updated."""

    active_topics: list[str] = field(default_factory=list)
    current_focus: str | None = None
    mood: str = "neutral"  # Could be inferred from recent interactions
    pending_tasks: list[str] = field(default_factory=list)
    last_interaction: str | None = None


@dataclass
class ContextLayer:
    """
    A piece of context to include in the prompt.

    Context assembly combines multiple layers based on the current situation.
    """

    name: str  # For debugging/logging
    priority: int  # Higher = earlier in context
    content: str  # The actual text
    source: str | None = None  # Where this came from (slice key, etc.)
    token_estimate: int = 0  # Rough token count for budgeting

    def __lt__(self, other):
        """Sort by priority (descending)."""
        return self.priority > other.priority
