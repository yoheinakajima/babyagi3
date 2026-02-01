"""
Context assembly for the memory system.

Builds context deterministically from pre-computed summaries.
No LLM calls required at assembly time.
"""

from dataclasses import dataclass, field
from datetime import datetime

from .models import AgentState, AssembledContext, Event


# ═══════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════


@dataclass
class ContextConfig:
    """Configuration for context assembly."""

    # Token budgets per section (approximate)
    token_budgets: dict = field(
        default_factory=lambda: {
            "identity": 200,
            "state": 150,
            "knowledge": 500,
            "recent": 400,
            "channel": 300,
            "tool": 200,
            "task": 400,
            "counterparty": 400,
            "topics": 400,
            "user_preferences": 300,  # Self-improvement: user preferences summary
            "learnings": 200,  # Self-improvement: context-specific learnings
        }
    )

    # Limits for list items
    recent_events_limit: int = 10
    channel_events_limit: int = 5
    tool_events_limit: int = 5
    task_events_limit: int = 10
    counterparty_events_limit: int = 10
    relationships_limit: int = 10
    max_topics: int = 5

    # Total context budget
    max_context_tokens: int = 4000

    # Chars per token estimate (conservative)
    chars_per_token: int = 4


# Default config instance
DEFAULT_CONFIG = ContextConfig()


# ═══════════════════════════════════════════════════════════
# TOKEN BUDGETING
# ═══════════════════════════════════════════════════════════


def _truncate_to_budget(text: str, budget: int, chars_per_token: int = 4) -> str:
    """Truncate text to fit within token budget."""
    if not text:
        return text
    max_chars = budget * chars_per_token
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count for text."""
    if not text:
        return 0
    return len(text) // chars_per_token


# ═══════════════════════════════════════════════════════════
# CONTEXT ASSEMBLY
# ═══════════════════════════════════════════════════════════


def assemble_context(
    event: Event | None,
    state: AgentState,
    store,  # MemoryStore - avoiding circular import
    retrieval,  # QuickRetrieval - avoiding circular import
    config: ContextConfig | None = None,
) -> AssembledContext:
    """
    Assemble context from pre-computed summaries.

    This function is deterministic and fast - no LLM calls.
    Context is layered:
    - Always: identity, state, knowledge, recent
    - Conditional: channel, tool, task, topics, counterparty

    Args:
        event: Current event being processed (optional)
        state: Current agent state
        store: Memory store instance
        retrieval: Quick retrieval instance
        config: Context configuration (uses defaults if not provided)

    Returns:
        AssembledContext with metrics
    """
    config = config or DEFAULT_CONFIG
    ctx = AssembledContext()

    # Track metrics
    metrics = {
        "sections_included": [],
        "tokens_by_section": {},
        "entities_referenced": set(),
        "truncations": [],
    }

    # ═══════════════════════════════════════════════════════════
    # ALWAYS INCLUDED
    # ═══════════════════════════════════════════════════════════

    # Identity
    ctx.identity = _build_identity(state, store, config, metrics)
    metrics["sections_included"].append("identity")

    # Current state
    ctx.state = _build_state(state, store, config, metrics)
    metrics["sections_included"].append("state")

    # Knowledge (root summary)
    root_node = store.get_summary_node("root")
    if root_node:
        budget = config.token_budgets.get("knowledge", 500)
        ctx.knowledge = _truncate_to_budget(
            root_node.summary, budget, config.chars_per_token
        )
        if len(root_node.summary) > budget * config.chars_per_token:
            metrics["truncations"].append("knowledge")
    metrics["sections_included"].append("knowledge")
    metrics["tokens_by_section"]["knowledge"] = _estimate_tokens(
        ctx.knowledge, config.chars_per_token
    )

    # Recent activity
    ctx.recent = _build_recent(store, config, metrics)
    metrics["sections_included"].append("recent")

    # ═══════════════════════════════════════════════════════════
    # CONDITIONAL (based on event)
    # ═══════════════════════════════════════════════════════════

    if event:
        # Channel context
        if event.channel:
            ctx.channel = _build_channel_context(event.channel, store, config, metrics)
            if ctx.channel:
                metrics["sections_included"].append("channel")

        # Tool context
        if event.tool_id:
            ctx.tool = _build_tool_context(event.tool_id, store, config, metrics)
            if ctx.tool:
                metrics["sections_included"].append("tool")

        # Task context
        if event.task_id:
            ctx.task = _build_task_context(event.task_id, store, config, metrics)
            if ctx.task:
                metrics["sections_included"].append("task")

        # Counterparty context (person/org)
        if event.person_id:
            ctx.counterparty = _build_counterparty_context(
                event.person_id, store, config, metrics
            )
            if ctx.counterparty:
                metrics["sections_included"].append("counterparty")
                metrics["entities_referenced"].add(event.person_id)

    # Topics (from current state, always included if present)
    if state.current_topics:
        ctx.topics = _build_topics_context(
            state.current_topics[: config.max_topics], store, config, metrics
        )
        if ctx.topics:
            metrics["sections_included"].append("topics")

    # ═══════════════════════════════════════════════════════════
    # SELF-IMPROVEMENT (learnings and preferences)
    # ═══════════════════════════════════════════════════════════

    # User preferences (always included if available)
    ctx.user_preferences = _build_user_preferences(store, config, metrics)
    if ctx.user_preferences:
        metrics["sections_included"].append("user_preferences")

    # Context-specific learnings
    ctx.learnings = _build_learnings_context(event, store, config, metrics)
    if ctx.learnings:
        metrics["sections_included"].append("learnings")

    # ═══════════════════════════════════════════════════════════
    # FINALIZE METRICS
    # ═══════════════════════════════════════════════════════════

    # Calculate total tokens
    total_tokens = sum(metrics["tokens_by_section"].values())
    metrics["total_tokens_estimate"] = total_tokens
    metrics["entities_referenced"] = list(metrics["entities_referenced"])
    metrics["budget_used_percent"] = (
        (total_tokens / config.max_context_tokens * 100)
        if config.max_context_tokens
        else 0
    )

    # Store metrics on context
    ctx.metrics = metrics

    return ctx


def _build_identity(
    state: AgentState, store, config: ContextConfig, metrics: dict
) -> dict:
    """Build identity context with budget."""
    budget = config.token_budgets.get("identity", 200)
    identity = {
        "name": state.name,
        "description": state.description,
    }

    # Owner info
    if state.owner_entity_id:
        owner = store.get_entity(state.owner_entity_id)
        if owner:
            identity["owner"] = owner.name
            metrics["entities_referenced"].add(state.owner_entity_id)
            owner_node = store.get_summary_node(f"entity:{owner.id}")
            if owner_node:
                identity["owner_summary"] = _truncate_to_budget(
                    owner_node.summary, budget // 3, config.chars_per_token
                )

    # Self info
    if state.self_entity_id:
        self_entity = store.get_entity(state.self_entity_id)
        if self_entity:
            self_node = store.get_summary_node(f"entity:{self_entity.id}")
            if self_node:
                identity["self_summary"] = _truncate_to_budget(
                    self_node.summary, budget // 3, config.chars_per_token
                )

    metrics["tokens_by_section"]["identity"] = _estimate_tokens(
        str(identity), config.chars_per_token
    )
    return identity


def _build_state(
    state: AgentState, store, config: ContextConfig, metrics: dict
) -> dict:
    """Build current state context."""
    result = {}

    if state.mood:
        result["mood"] = state.mood

    if state.focus:
        result["focus"] = state.focus

    if state.current_topics:
        # Get topic labels
        topic_labels = []
        for topic_id in state.current_topics[: config.max_topics]:
            topic = store.get_topic(topic_id)
            if topic:
                topic_labels.append(topic.label)
        if topic_labels:
            result["topics"] = topic_labels

    if state.active_tasks:
        # Get task titles
        task_titles = []
        for task_id in state.active_tasks[:5]:  # Limit to 5
            task = store.get_task(task_id)
            if task:
                task_titles.append(task.title)
        if task_titles:
            result["active_tasks"] = task_titles

    metrics["tokens_by_section"]["state"] = _estimate_tokens(
        str(result), config.chars_per_token
    )
    return result


def _build_recent(store, config: ContextConfig, metrics: dict) -> dict:
    """Build recent activity context with budget."""
    budget = config.token_budgets.get("recent", 400)
    recent_events = store.get_recent_events(limit=config.recent_events_limit)

    # Format events for context
    formatted_events = []
    content_budget_per_event = (budget * config.chars_per_token) // max(
        len(recent_events), 1
    )

    for event in recent_events:
        content = event.content
        if len(content) > content_budget_per_event:
            content = content[: content_budget_per_event - 3] + "..."

        formatted = {
            "type": event.event_type,
            "content": content,
            "channel": event.channel,
            "direction": event.direction,
        }
        if event.person_id:
            entity = store.get_entity(event.person_id)
            if entity:
                formatted["person"] = entity.name
                metrics["entities_referenced"].add(event.person_id)
        formatted_events.append(formatted)

    result = {
        "events": formatted_events,
        "count": len(formatted_events),
    }

    metrics["tokens_by_section"]["recent"] = _estimate_tokens(
        str(result), config.chars_per_token
    )
    return result


def _build_channel_context(
    channel: str, store, config: ContextConfig, metrics: dict
) -> dict | None:
    """Build channel-specific context with budget."""
    budget = config.token_budgets.get("channel", 300)
    node = store.get_summary_node(f"channel:{channel}")
    if not node:
        return None

    recent = store.get_recent_events(limit=config.channel_events_limit, channel=channel)

    summary = _truncate_to_budget(node.summary, budget // 2, config.chars_per_token)
    if len(node.summary) > (budget // 2) * config.chars_per_token:
        metrics["truncations"].append(f"channel:{channel}")

    result = {
        "name": channel,
        "summary": summary,
        "recent": [
            {
                "content": e.content[:150],
                "direction": e.direction,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in recent
        ],
    }

    metrics["tokens_by_section"]["channel"] = _estimate_tokens(
        str(result), config.chars_per_token
    )
    return result


def _build_tool_context(
    tool_id: str, store, config: ContextConfig, metrics: dict
) -> dict | None:
    """Build tool-specific context with budget."""
    budget = config.token_budgets.get("tool", 200)
    node = store.get_summary_node(f"tool:{tool_id}")
    if not node:
        return None

    recent = store.get_recent_events(limit=config.tool_events_limit, tool_id=tool_id)

    summary = _truncate_to_budget(node.summary, budget, config.chars_per_token)

    result = {
        "name": tool_id,
        "summary": summary,
        "recent_calls": len(recent),
    }

    metrics["tokens_by_section"]["tool"] = _estimate_tokens(
        str(result), config.chars_per_token
    )
    return result


def _build_task_context(
    task_id: str, store, config: ContextConfig, metrics: dict
) -> dict | None:
    """Build task-specific context with budget."""
    budget = config.token_budgets.get("task", 400)
    task = store.get_task(task_id)
    if not task:
        return None

    result = {
        "task": {
            "id": task.id,
            "title": task.title,
            "description": (
                task.description[:200] if task.description else None
            ),
            "status": task.status,
            "type": task.type_raw,
        },
    }

    # Task summary
    task_node = store.get_summary_node(f"task:{task_id}")
    if task_node:
        result["summary"] = _truncate_to_budget(
            task_node.summary, budget // 3, config.chars_per_token
        )

    # Task type summary
    if task.type_cluster:
        type_node = store.get_summary_node(f"task_type:{task.type_cluster}")
        if type_node:
            result["type_summary"] = _truncate_to_budget(
                type_node.summary, budget // 4, config.chars_per_token
            )

    # Task events
    events = store.get_recent_events(limit=config.task_events_limit, task_id=task_id)
    result["events"] = [
        {
            "type": e.event_type,
            "content": e.content[:150],
        }
        for e in events
    ]

    metrics["tokens_by_section"]["task"] = _estimate_tokens(
        str(result), config.chars_per_token
    )
    return result


def _build_counterparty_context(
    person_id: str, store, config: ContextConfig, metrics: dict
) -> dict | None:
    """Build counterparty (person/org) context with budget."""
    budget = config.token_budgets.get("counterparty", 400)
    entity = store.get_entity(person_id)
    if not entity:
        return None

    result = {
        "entity": {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "type_raw": entity.type_raw,
            "description": (
                entity.description[:150] if entity.description else None
            ),
        },
    }

    # Entity summary
    entity_node = store.get_summary_node(f"entity:{entity.id}")
    if entity_node:
        result["summary"] = _truncate_to_budget(
            entity_node.summary, budget // 3, config.chars_per_token
        )

    # Relationships (with limit)
    edges = store.get_edges(entity.id)
    if edges:
        result["edges"] = []
        for edge in edges[: config.relationships_limit]:
            edge_info = {
                "relation": edge.relation,
                "direction": (
                    "outgoing" if edge.source_entity_id == entity.id else "incoming"
                ),
            }
            # Get the other entity's name
            other_id = (
                edge.target_entity_id
                if edge.source_entity_id == entity.id
                else edge.source_entity_id
            )
            other = store.get_entity(other_id)
            if other:
                edge_info["other"] = other.name
                metrics["entities_referenced"].add(other_id)
            result["edges"].append(edge_info)

    # Recent interactions
    recent = store.get_recent_events(
        limit=config.counterparty_events_limit, person_id=person_id
    )
    result["recent"] = [
        {
            "type": e.event_type,
            "content": e.content[:150],
            "direction": e.direction,
            "channel": e.channel,
        }
        for e in recent
    ]

    metrics["tokens_by_section"]["counterparty"] = _estimate_tokens(
        str(result), config.chars_per_token
    )
    return result


def _build_topics_context(
    topic_ids: list[str], store, config: ContextConfig, metrics: dict
) -> list[dict]:
    """Build topics context with budget."""
    budget = config.token_budgets.get("topics", 400)
    budget_per_topic = budget // max(len(topic_ids), 1)

    topics = []
    for topic_id in topic_ids:
        topic = store.get_topic(topic_id)
        if not topic:
            continue

        topic_info = {
            "label": topic.label,
            "keywords": topic.keywords[:5] if topic.keywords else [],
        }

        # Topic summary
        topic_node = store.get_summary_node(f"topic:{topic_id}")
        if topic_node:
            topic_info["summary"] = _truncate_to_budget(
                topic_node.summary, budget_per_topic, config.chars_per_token
            )

        topics.append(topic_info)

    metrics["tokens_by_section"]["topics"] = _estimate_tokens(
        str(topics), config.chars_per_token
    )
    return topics


def _build_user_preferences(
    store, config: ContextConfig, metrics: dict
) -> str:
    """Build user preferences context (always included if available)."""
    budget = config.token_budgets.get("user_preferences", 300)

    # Get user_preferences summary node
    prefs_node = store.get_summary_node("user_preferences")
    if not prefs_node or not prefs_node.summary:
        return ""

    # Skip if it's just the default message
    if prefs_node.summary == "No user preferences recorded yet.":
        return ""

    result = _truncate_to_budget(prefs_node.summary, budget, config.chars_per_token)

    metrics["tokens_by_section"]["user_preferences"] = _estimate_tokens(
        result, config.chars_per_token
    )
    return result


def _build_learnings_context(
    event: Event | None, store, config: ContextConfig, metrics: dict
) -> list[dict]:
    """Build context-specific learnings based on current event."""
    budget = config.token_budgets.get("learnings", 200)
    learnings = []

    if not event:
        return []

    try:
        # Import here to avoid circular imports
        from .learning import LearningRetriever

        retriever = LearningRetriever(store)

        # If using a specific tool, get tool-specific learnings
        if event.tool_id:
            tool_learnings = retriever.get_for_tool(event.tool_id, limit=3)
            for l in tool_learnings:
                learnings.append({
                    "type": "tool",
                    "tool": event.tool_id,
                    "learning": l.content[:200],
                    "recommendation": l.recommendation[:100] if l.recommendation else None,
                })

        # If this looks like an objective start, get similar objective learnings
        if event.event_type in ["objective_start", "task_created"]:
            obj_learnings = retriever.get_for_objective(event.content, limit=3)
            for l in obj_learnings:
                # Avoid duplicates from tool learnings
                if l.tool_id and event.tool_id and l.tool_id == event.tool_id:
                    continue
                learnings.append({
                    "type": "objective",
                    "learning": l.content[:200],
                    "recommendation": l.recommendation[:100] if l.recommendation else None,
                })

    except Exception as e:
        # Don't fail context assembly if learnings fail
        print(f"Error building learnings context: {e}")

    # Limit total learnings to fit budget
    max_learnings = budget // 50  # Rough estimate of tokens per learning
    learnings = learnings[:max_learnings]

    metrics["tokens_by_section"]["learnings"] = _estimate_tokens(
        str(learnings), config.chars_per_token
    )
    return learnings


def format_context_for_prompt(ctx: AssembledContext) -> str:
    """
    Format assembled context as a string for inclusion in prompts.

    This is a convenience method - the AssembledContext.to_prompt()
    method provides similar functionality.
    """
    return ctx.to_prompt()
