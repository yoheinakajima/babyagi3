"""
Context assembly for the memory system.

Builds context deterministically from pre-computed summaries.
No LLM calls required at assembly time.
"""

from .models import AgentState, AssembledContext, Event


def assemble_context(
    event: Event | None,
    state: AgentState,
    store,  # MemoryStore - avoiding circular import
    retrieval,  # QuickRetrieval - avoiding circular import
) -> AssembledContext:
    """
    Assemble context from pre-computed summaries.

    This function is deterministic and fast - no LLM calls.
    Context is layered:
    - Always: identity, state, knowledge, recent
    - Conditional: channel, tool, task, topics, counterparty
    """
    ctx = AssembledContext()

    # ═══════════════════════════════════════════════════════════
    # ALWAYS INCLUDED
    # ═══════════════════════════════════════════════════════════

    # Identity
    ctx.identity = _build_identity(state, store)

    # Current state
    ctx.state = _build_state(state, store)

    # Knowledge (root summary)
    root_node = store.get_summary_node("root")
    ctx.knowledge = root_node.summary if root_node else ""

    # Recent activity
    ctx.recent = _build_recent(store)

    # ═══════════════════════════════════════════════════════════
    # CONDITIONAL (based on event)
    # ═══════════════════════════════════════════════════════════

    if event:
        # Channel context
        if event.channel:
            ctx.channel = _build_channel_context(event.channel, store)

        # Tool context
        if event.tool_id:
            ctx.tool = _build_tool_context(event.tool_id, store)

        # Task context
        if event.task_id:
            ctx.task = _build_task_context(event.task_id, store)

        # Counterparty context (person/org)
        if event.person_id:
            ctx.counterparty = _build_counterparty_context(event.person_id, store)

    # Topics (from current state, always included if present)
    if state.current_topics:
        ctx.topics = _build_topics_context(state.current_topics, store)

    return ctx


def _build_identity(state: AgentState, store) -> dict:
    """Build identity context."""
    identity = {
        "name": state.name,
        "description": state.description,
    }

    # Owner info
    if state.owner_entity_id:
        owner = store.get_entity(state.owner_entity_id)
        if owner:
            identity["owner"] = owner.name
            owner_node = store.get_summary_node(f"entity:{owner.id}")
            if owner_node:
                identity["owner_summary"] = owner_node.summary

    # Self info
    if state.self_entity_id:
        self_entity = store.get_entity(state.self_entity_id)
        if self_entity:
            self_node = store.get_summary_node(f"entity:{self_entity.id}")
            if self_node:
                identity["self_summary"] = self_node.summary

    return identity


def _build_state(state: AgentState, store) -> dict:
    """Build current state context."""
    result = {}

    if state.mood:
        result["mood"] = state.mood

    if state.focus:
        result["focus"] = state.focus

    if state.current_topics:
        # Get topic labels
        topic_labels = []
        for topic_id in state.current_topics:
            topic = store.get_topic(topic_id)
            if topic:
                topic_labels.append(topic.label)
        if topic_labels:
            result["topics"] = topic_labels

    if state.active_tasks:
        # Get task titles
        task_titles = []
        for task_id in state.active_tasks:
            task = store.get_task(task_id)
            if task:
                task_titles.append(task.title)
        if task_titles:
            result["active_tasks"] = task_titles

    return result


def _build_recent(store, limit: int = 10) -> dict:
    """Build recent activity context."""
    recent_events = store.get_recent_events(limit=limit)

    # Format events for context
    formatted_events = []
    for event in recent_events:
        formatted = {
            "type": event.event_type,
            "content": event.content[:500] if len(event.content) > 500 else event.content,
            "channel": event.channel,
            "direction": event.direction,
        }
        if event.person_id:
            entity = store.get_entity(event.person_id)
            if entity:
                formatted["person"] = entity.name
        formatted_events.append(formatted)

    return {
        "events": formatted_events,
        "count": len(formatted_events),
    }


def _build_channel_context(channel: str, store) -> dict | None:
    """Build channel-specific context."""
    node = store.get_summary_node(f"channel:{channel}")
    if not node:
        return None

    recent = store.get_recent_events(limit=5, channel=channel)

    return {
        "name": channel,
        "summary": node.summary,
        "recent": [
            {
                "content": e.content[:200],
                "direction": e.direction,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in recent
        ],
    }


def _build_tool_context(tool_id: str, store) -> dict | None:
    """Build tool-specific context."""
    node = store.get_summary_node(f"tool:{tool_id}")
    if not node:
        return None

    recent = store.get_recent_events(limit=5, tool_id=tool_id)

    return {
        "name": tool_id,
        "summary": node.summary,
        "recent_calls": len(recent),
    }


def _build_task_context(task_id: str, store) -> dict | None:
    """Build task-specific context."""
    task = store.get_task(task_id)
    if not task:
        return None

    result = {
        "task": {
            "id": task.id,
            "title": task.title,
            "description": task.description,
            "status": task.status,
            "type": task.type_raw,
        },
    }

    # Task summary
    task_node = store.get_summary_node(f"task:{task_id}")
    if task_node:
        result["summary"] = task_node.summary

    # Task type summary
    if task.type_cluster:
        type_node = store.get_summary_node(f"task_type:{task.type_cluster}")
        if type_node:
            result["type_summary"] = type_node.summary

    # Task events
    events = store.get_recent_events(limit=10, task_id=task_id)
    result["events"] = [
        {
            "type": e.event_type,
            "content": e.content[:200],
        }
        for e in events
    ]

    return result


def _build_counterparty_context(person_id: str, store) -> dict | None:
    """Build counterparty (person/org) context."""
    entity = store.get_entity(person_id)
    if not entity:
        return None

    result = {
        "entity": {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "type_raw": entity.type_raw,
            "description": entity.description,
        },
    }

    # Entity summary
    entity_node = store.get_summary_node(f"entity:{entity.id}")
    if entity_node:
        result["summary"] = entity_node.summary

    # Relationships
    edges = store.get_edges(entity.id)
    if edges:
        result["edges"] = []
        for edge in edges[:10]:  # Limit to 10 relationships
            edge_info = {
                "relation": edge.relation,
                "direction": "outgoing" if edge.source_entity_id == entity.id else "incoming",
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
            result["edges"].append(edge_info)

    # Recent interactions
    recent = store.get_recent_events(limit=10, person_id=person_id)
    result["recent"] = [
        {
            "type": e.event_type,
            "content": e.content[:200],
            "direction": e.direction,
            "channel": e.channel,
        }
        for e in recent
    ]

    return result


def _build_topics_context(topic_ids: list[str], store) -> list[dict]:
    """Build topics context."""
    topics = []
    for topic_id in topic_ids:
        topic = store.get_topic(topic_id)
        if not topic:
            continue

        topic_info = {
            "label": topic.label,
            "keywords": topic.keywords,
        }

        # Topic summary
        topic_node = store.get_summary_node(f"topic:{topic_id}")
        if topic_node:
            topic_info["summary"] = topic_node.summary

        topics.append(topic_info)

    return topics


def format_context_for_prompt(ctx: AssembledContext) -> str:
    """
    Format assembled context as a string for inclusion in prompts.

    This is a convenience method - the AssembledContext.to_prompt()
    method provides similar functionality.
    """
    return ctx.to_prompt()
