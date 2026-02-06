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
    """Record of a tool for tracking usage and summaries.

    DEPRECATED: Use ToolDefinition for new code. This class is kept
    for backward compatibility with existing database records.
    """

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


@dataclass
class ToolDefinition:
    """A persisted tool definition - enables agent self-improvement.

    This stores everything needed to reconstruct a dynamically-created tool
    after restart, plus execution statistics for monitoring and debugging.

    Tools are also represented as Entities in the knowledge graph, allowing
    the agent to query "what tools do I have?" and track relationships.

    Tool Types:
    - "executable": Python code that runs directly (default)
    - "skill": Returns behavioral instructions when called
    - "composio": Thin wrapper that calls Composio library
    """

    id: str
    name: str  # Unique identifier (tool name)
    description: str

    # Tool type (NEW)
    tool_type: str = "executable"  # "executable" | "skill" | "composio"

    # Definition (what makes it executable)
    source_code: str | None = None  # Python code for dynamic tools
    parameters: dict = field(default_factory=dict)  # JSON schema for input
    packages: list[str] = field(default_factory=list)  # Required packages
    env: list[str] = field(default_factory=list)  # Required env vars
    tool_var_name: str | None = None  # Variable name in source code (e.g., "my_tool")

    # For skills (NEW)
    skill_content: str | None = None  # The SKILL.md markdown instructions

    # For composio tools (NEW)
    composio_app: str | None = None  # "SLACK", "GITHUB", etc.
    composio_action: str | None = None  # "SLACK_SEND_MESSAGE", etc.

    # Dependencies - tools/skills this depends on (NEW)
    depends_on: list[str] = field(default_factory=list)

    # Category for organization
    category: str = "custom"  # "core", "builtin", "custom", "skill", "composio", etc.

    # State
    is_enabled: bool = True
    is_dynamic: bool = True  # False for built-in/static tools

    # Execution statistics
    usage_count: int = 0
    success_count: int = 0
    error_count: int = 0
    last_used_at: datetime | None = None
    last_error: str | None = None
    last_error_at: datetime | None = None
    avg_duration_ms: float = 0.0
    total_duration_ms: float = 0.0  # For calculating running average

    # Graph integration
    entity_id: str | None = None  # Links to Entity (type="tool")
    summary_node_id: str | None = None

    # Versioning
    version: int = 1

    # Provenance
    created_by_event_id: str | None = None  # Which event triggered creation

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.usage_count == 0:
            return 100.0
        return (self.success_count / self.usage_count) * 100.0

    @property
    def is_healthy(self) -> bool:
        """Check if tool has acceptable error rate (< 50% errors)."""
        if self.usage_count < 3:
            return True  # Not enough data
        return self.success_rate >= 50.0

    def __repr__(self) -> str:
        status = "enabled" if self.is_enabled else "disabled"
        health = "healthy" if self.is_healthy else "unhealthy"
        return (
            f"ToolDefinition(name='{self.name}', {status}, {health}, "
            f"usage={self.usage_count}, success_rate={self.success_rate:.1f}%)"
        )


# ═══════════════════════════════════════════════════════════
# CREDENTIALS (secure storage for accounts and payment methods)
# ═══════════════════════════════════════════════════════════


@dataclass
class Credential:
    """A securely stored credential - user account or payment method.

    Credentials are stored in the database with sensitive data (passwords,
    card numbers) stored only as references to the keyring/secrets system.

    This enables:
    - Persistent storage of what accounts exist
    - Quick lookup by service name
    - Secure storage of actual secrets via keyring
    """

    id: str
    credential_type: str  # "account", "credit_card"
    service: str  # "yohei.ai", "stripe.com", etc.

    # For user accounts
    username: str | None = None
    email: str | None = None
    password_ref: str | None = None  # Reference to secret in keyring

    # For credit cards
    card_last_four: str | None = None
    card_type: str | None = None  # "visa", "mastercard", etc.
    card_expiry: str | None = None  # "MM/YY"
    card_ref: str | None = None  # Reference to full card in keyring
    billing_name: str | None = None
    billing_address: str | None = None

    # Common fields
    notes: str | None = None
    metadata: dict | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime | None = None

    def __repr__(self) -> str:
        if self.credential_type == "credit_card":
            return (
                f"Credential(type='credit_card', service='{self.service}', "
                f"card=****{self.card_last_four}, type={self.card_type})"
            )
        return (
            f"Credential(type='{self.credential_type}', service='{self.service}', "
            f"username='{self.username or self.email}')"
        )


# ═══════════════════════════════════════════════════════════
# LEARNING MODELS (for self-improvement system)
# ═══════════════════════════════════════════════════════════


@dataclass
class Learning:
    """A piece of learned knowledge from feedback or self-evaluation.

    Learnings are extracted from:
    - User feedback in messages (corrections, preferences, complaints)
    - Self-evaluation of completed objectives
    - Direct observations

    They are stored with embeddings for vector search and tied to:
    - Specific tools (how to use them better)
    - Objective types (how to approach similar tasks)
    - Topics (domain-specific knowledge)
    - Entities (person-specific preferences)
    """

    id: str

    # Source
    source_type: str  # "user_feedback", "self_evaluation", "observation"
    source_event_id: str | None  # Event that triggered this learning

    # Content
    content: str  # The actual learning/insight
    content_embedding: list[float] | None = None  # For vector search

    # Classification
    sentiment: str = "neutral"  # "positive", "negative", "neutral"
    confidence: float = 0.5  # 0-1, how confident we are in this learning
    category: str = "general"  # "general", "owner_profile", "agent_self", "tool_feedback"

    # Associations (what this learning is about)
    tool_id: str | None = None  # If about a specific tool
    topic_ids: list[str] = field(default_factory=list)  # Related topics
    objective_type: str | None = None  # Type of objective (e.g., "research", "code", "email")
    entity_ids: list[str] = field(default_factory=list)  # Related entities (people, orgs)

    # Actionable insight
    applies_when: str | None = None  # Condition when this learning applies
    recommendation: str | None = None  # What to do differently

    # Stats
    times_applied: int = 0  # How often this learning was used in context
    last_applied_at: datetime | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        content_preview = self.content[:40] + "..." if len(self.content) > 40 else self.content
        source = f"{self.source_type}"
        if self.tool_id:
            source += f" (tool: {self.tool_id})"
        elif self.objective_type:
            source += f" (obj: {self.objective_type})"
        return (
            f"Learning(id={self.id[:8]}..., {source}, "
            f"sentiment={self.sentiment}, content='{content_preview}')"
        )


@dataclass
class ExtractedFeedback:
    """Feedback extracted from a user message (before creating Learning)."""

    has_feedback: bool = False
    feedback_type: str | None = None  # "correction", "praise", "preference", "complaint", "profile_info"
    category: str = "general"  # "general", "owner_profile", "agent_self", "tool_feedback"
    about_tool: str | None = None
    about_objective_type: str | None = None
    about_entity_id: str | None = None
    what_was_wrong: str | None = None
    what_to_do_instead: str | None = None
    sentiment: str = "neutral"
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════
# FACT MODELS (unified triplet storage for all sources)
# ═══════════════════════════════════════════════════════════


@dataclass
class Fact:
    """A fact triplet (subject-predicate-object) extracted from any source.

    Facts are the unified knowledge representation that can come from:
    - Conversations (user messages, tool results)
    - Documents (PDFs, Word docs, CSVs)
    - Observations (agent self-observations)
    - Tools (results from web searches, etc.)

    Facts enable multiple retrieval methods:
    - Semantic search on fact_text embedding
    - Graph traversal from subject/object entities
    - Keyword search on predicate/fact_text
    - Temporal queries on valid_from/valid_to
    """

    id: str

    # Triplet Core
    subject_entity_id: str  # Always links to an entity
    predicate: str  # The verb/relationship
    object_entity_id: str | None = None  # If object is an entity
    object_value: str | None = None  # If object is a literal value
    object_type: str = "value"  # "entity" | "value" | "text"

    # Additional entities mentioned in context
    mentioned_entity_ids: list[str] = field(default_factory=list)

    # Classification
    fact_type: str = "relation"  # relation|attribute|event|state|metric
    predicate_type: str | None = None  # Clustered: financial|professional|temporal|etc

    # Human-readable (LLM-generated)
    fact_text: str = ""  # Full sentence: "John invested $500K in TechCorp"
    fact_embedding: list[float] | None = None  # Vector for semantic search

    # Provenance
    source_type: str = "conversation"  # conversation|document|tool|observation
    source_id: str | None = None  # Event ID or file entity ID
    source_event_ids: list[str] = field(default_factory=list)  # Accumulated over time

    # Confidence & Strength
    confidence: float = 0.8  # Extraction confidence
    strength: float = 0.5  # Accumulated mentions (0-1)

    # Temporality
    valid_from: datetime | None = None  # When fact became true
    valid_to: datetime | None = None  # When invalidated (None=current)
    is_current: bool = True

    # Usage tracking
    times_retrieved: int = 0  # How often found in searches
    times_used: int = 0  # How often used in responses
    last_used_at: datetime | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        obj = self.object_entity_id or self.object_value or "?"
        return (
            f"Fact(id={self.id[:8]}..., "
            f"'{self.predicate}', obj={obj[:20] if obj else '?'}..., "
            f"type={self.fact_type}, strength={self.strength:.2f})"
        )


@dataclass
class RetrievalQuery:
    """A tracked retrieval query for learning optimal retrieval strategies."""

    id: str
    query_text: str
    query_embedding: list[float] | None = None
    query_type: str | None = None  # Inferred: relationship|attribute|search|temporal

    # Strategy used
    chain_strategy: list[str] = field(default_factory=list)  # ["keyword", "graph", "semantic"]

    # Outcome
    total_results: int = 0
    results_used: int = 0
    was_successful: bool = False  # Did it answer the question?

    # Performance
    total_time_ms: int = 0

    # Context
    objective_id: str | None = None
    event_id: str | None = None

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalResult:
    """A single result from a retrieval query, for tracking what was useful."""

    id: str
    query_id: str  # FK to RetrievalQuery

    # What was found
    result_type: str  # fact|entity|event|summary
    result_id: str  # ID of the found item

    # How it was found
    retrieval_method: str  # semantic|keyword|graph|direct
    method_step: int = 1  # Position in chain (1, 2, 3...)

    # Relevance
    similarity_score: float | None = None  # If semantic search
    rank_position: int = 0  # Position in results

    # Usage
    was_used: bool = False  # Was this result used in response?

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractionCall:
    """Metrics for a single extraction LLM call."""

    id: str
    source_type: str  # conversation|document|tool
    source_id: str | None = None
    content_length: int = 0  # Characters processed

    # Results
    entities_extracted: int = 0
    facts_extracted: int = 0
    topics_extracted: int = 0

    # Cost tracking
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0

    # Timing decision
    timing_mode: str = "immediate"  # immediate|background|on_demand

    created_at: datetime = field(default_factory=datetime.now)


# ═══════════════════════════════════════════════════════════
# EXTRACTION MODELS (used by extraction pipeline)
# ═══════════════════════════════════════════════════════════


@dataclass
class ExtractedFact:
    """A fact extracted from content (before storage/deduplication)."""

    subject: str  # Entity name
    predicate: str  # The verb/relationship
    object: str  # Entity name OR literal value
    object_type: str = "value"  # "entity" | "value" | "text"

    fact_type: str = "relation"  # relation|attribute|event|state|metric
    fact_text: str = ""  # Full natural sentence
    confidence: float = 0.8

    # Temporality
    valid_from: str | None = None  # ISO date if known
    valid_to: str | None = None  # ISO date if known

    # Additional entities mentioned in context
    mentioned_entities: list[str] = field(default_factory=list)


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
    """The result of extracting from an event or document."""

    entities: list[ExtractedEntity] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    edges: list[ExtractedEdge] = field(default_factory=list)  # Legacy, still populated for compatibility
    topics: list[ExtractedTopic] = field(default_factory=list)
    task_type: str | None = None
    notes: str | None = None

    # Metrics for this extraction
    extraction_call_id: str | None = None


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

    # Self-improvement (learnings from feedback and evaluation)
    user_preferences: str = ""  # Summarized user preferences (always included)
    learnings: list[dict] = field(default_factory=list)  # Context-specific learnings

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
        if self.user_preferences:
            result["user_preferences"] = self.user_preferences
        if self.learnings:
            result["learnings"] = self.learnings
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

        # User Preferences (always included if available)
        if self.user_preferences:
            sections.append("\n## User Preferences")
            sections.append(self.user_preferences)

        # Relevant Learnings (context-specific)
        if self.learnings:
            sections.append("\n## Relevant Learnings")
            for learning in self.learnings:
                learn_type = learning.get("type", "general")
                if learn_type == "tool":
                    sections.append(f"- **{learning.get('tool', 'Tool')}**: {learning.get('learning', '')}")
                else:
                    sections.append(f"- {learning.get('learning', '')}")
                if learning.get("recommendation"):
                    sections.append(f"  \u2192 {learning['recommendation']}")

        return "\n".join(sections)
