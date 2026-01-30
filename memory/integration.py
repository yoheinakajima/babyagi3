"""
Integration helpers for connecting the memory system to the main agent.

This module provides utilities to:
1. Hook memory logging into the agent event system
2. Provide memory context to the system prompt
3. Create memory-enhanced tools
"""

import asyncio
from datetime import datetime

from .models import Event


def setup_memory_hooks(agent, memory):
    """
    Set up event hooks to automatically log agent activity to memory.

    Call this after creating both the agent and memory system:
        agent = Agent()
        memory = Memory()
        setup_memory_hooks(agent, memory)
    """

    @agent.on("tool_start")
    def on_tool_start(event):
        """Log tool calls to memory."""
        # Create event for tool call
        context = getattr(agent, "_current_context", {})
        memory.log_event(
            content=f"Tool call: {event['name']}\nInput: {event['input']}",
            event_type="tool_call",
            channel=context.get("channel"),
            direction="internal",
            tool_id=event["name"],
            person_id=context.get("person_id"),
            is_owner=context.get("is_owner", True),
        )

    @agent.on("tool_end")
    def on_tool_end(event):
        """Log tool results to memory."""
        context = getattr(agent, "_current_context", {})
        result_str = str(event.get("result", ""))[:1000]  # Truncate large results
        memory.log_event(
            content=f"Tool result: {event['name']}\nResult: {result_str}",
            event_type="tool_result",
            channel=context.get("channel"),
            direction="internal",
            tool_id=event["name"],
            person_id=context.get("person_id"),
            is_owner=context.get("is_owner", True),
        )

    @agent.on("objective_start")
    def on_objective_start(event):
        """Log objective start to memory."""
        memory.log_event(
            content=f"Started objective: {event['goal']}",
            event_type="task_created",
            channel=None,
            direction="internal",
            task_id=event["id"],
            is_owner=True,
        )

    @agent.on("objective_end")
    def on_objective_end(event):
        """Log objective completion to memory."""
        result = event.get("result", event.get("error", "No result"))
        memory.log_event(
            content=f"Objective {event['status']}: {result[:500]}",
            event_type="task_completed" if event["status"] == "completed" else "task_failed",
            channel=None,
            direction="internal",
            task_id=event["id"],
            is_owner=True,
        )


def log_message(memory, content: str, context: dict, direction: str = "inbound") -> Event:
    """
    Log a message event to memory.

    Call this when receiving or sending messages:
        event = log_message(memory, user_input, context, "inbound")
        event = log_message(memory, response, context, "outbound")
    """
    return memory.log_event(
        content=content,
        event_type="message",
        channel=context.get("channel"),
        direction=direction,
        person_id=context.get("person_id"),
        is_owner=context.get("is_owner", True),
        conversation_id=context.get("conversation_id"),
        metadata={
            "sender": context.get("sender"),
            "subject": context.get("subject"),
        },
    )


def get_memory_context_prompt(memory, event: Event | None = None) -> str:
    """
    Get memory context formatted for inclusion in the system prompt.

    Returns a string section to add to the system prompt.
    """
    try:
        ctx = memory.assemble_context(event)
        return f"""
MEMORY CONTEXT:
{ctx.to_prompt()}
"""
    except Exception as e:
        return f"\n(Memory context unavailable: {e})\n"


def create_enhanced_memory_tool(memory):
    """
    Create an enhanced memory tool that uses the new memory system.

    This replaces the simple in-memory MEMORIES list with the full
    graph-based memory system.
    """
    from agent import Tool

    def fn(params: dict, agent) -> dict:
        action = params["action"]

        if action == "store":
            # Store as an observation event
            event = memory.log_event(
                content=params["content"],
                event_type="observation",
                channel=None,
                direction="internal",
                is_owner=True,
            )
            return {"stored": True, "event_id": event.id}

        elif action == "search":
            query = params.get("query", "")
            if query:
                # Semantic search over events
                events = memory.search_events(query, limit=10)
                return {
                    "memories": [
                        {
                            "content": e.content[:500],
                            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                            "channel": e.channel,
                        }
                        for e in events
                    ]
                }
            else:
                # List recent
                events = memory.find_events(limit=10)
                return {
                    "memories": [
                        {
                            "content": e.content[:500],
                            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                        }
                        for e in events
                    ]
                }

        elif action == "list":
            events = memory.find_events(limit=20)
            return {
                "memories": [
                    {
                        "content": e.content[:500],
                        "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                        "channel": e.channel,
                        "type": e.event_type,
                    }
                    for e in events
                ]
            }

        elif action == "find_entity":
            query = params.get("query", "")
            entity_type = params.get("type")
            entities = memory.search_entities(query, type=entity_type, limit=10)
            return {
                "entities": [
                    {
                        "id": e.id,
                        "name": e.name,
                        "type": e.type,
                        "description": e.description,
                    }
                    for e in entities
                ]
            }

        elif action == "find_relationship":
            query = params.get("query", "")
            entity_id = params.get("entity_id")
            edges = memory.search_edges(query, entity_id=entity_id, limit=10)
            results = []
            for edge in edges:
                source = memory.get_entity(edge.source_entity_id)
                target = memory.get_entity(edge.target_entity_id)
                results.append({
                    "source": source.name if source else edge.source_entity_id,
                    "relation": edge.relation,
                    "target": target.name if target else edge.target_entity_id,
                })
            return {"relationships": results}

        elif action == "get_summary":
            key = params.get("key", "root")
            node = memory.get_summary(key)
            if node:
                return {
                    "key": node.key,
                    "label": node.label,
                    "summary": node.summary,
                    "event_count": node.event_count,
                }
            return {"error": f"Summary not found: {key}"}

        elif action == "get_context":
            # Return assembled context
            ctx = memory.assemble_context()
            return {"context": ctx.to_dict()}

        elif action == "deep_search":
            # Trigger deep retrieval
            query = params.get("query", "")
            if not query:
                return {"error": "Query required for deep search"}

            # Run deep retrieval asynchronously
            async def do_search():
                return await memory.deep_retrieve(query)

            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(asyncio.run, do_search()).result()
                return result
            except RuntimeError:
                return asyncio.run(do_search())

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="memory",
        description="""Enhanced memory system with semantic search and graph storage.

Actions:
- store: Store a fact or observation (content)
- search: Semantic search over memories (query)
- list: List recent memories
- find_entity: Find people, orgs, concepts (query, optional type)
- find_relationship: Find relationships (query, optional entity_id)
- get_summary: Get pre-computed summary (key like 'root', 'entity:uuid', 'topic:uuid')
- get_context: Get current assembled context
- deep_search: Thorough agentic search for complex queries (query)

Examples:
- memory(action="store", content="John works at Acme Corp")
- memory(action="search", query="fundraising discussions")
- memory(action="find_entity", query="venture capital", type="person")
- memory(action="find_relationship", query="investment")
- memory(action="get_summary", key="root")
- memory(action="deep_search", query="How has my relationship with Sarah changed?")""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "search", "list", "find_entity", "find_relationship", "get_summary", "get_context", "deep_search"],
                },
                "content": {"type": "string", "description": "Content to store"},
                "query": {"type": "string", "description": "Search query"},
                "key": {"type": "string", "description": "Summary node key"},
                "type": {"type": "string", "description": "Entity type filter"},
                "entity_id": {"type": "string", "description": "Entity ID for relationship search"},
            },
            "required": ["action"],
        },
        fn=fn,
    )


def create_extraction_background_task(memory, interval_seconds: int = 60):
    """
    Create a background task that processes pending extractions.

    Run this as an asyncio task:
        extraction_task = asyncio.create_task(
            create_extraction_background_task(memory)()
        )
    """
    async def extraction_loop():
        while True:
            try:
                # Get pending events
                events = memory.store.get_pending_extraction_events(limit=10)

                for event in events:
                    try:
                        await memory.extract_from_event(event)
                    except Exception as e:
                        print(f"Extraction failed for event {event.id}: {e}")

                # Refresh stale summaries
                await memory.refresh_stale_summaries(threshold=10)

            except Exception as e:
                print(f"Extraction loop error: {e}")

            await asyncio.sleep(interval_seconds)

    return extraction_loop


def create_summary_refresh_task(memory, interval_seconds: int = 300):
    """
    Create a background task that refreshes stale summaries.

    Run this as an asyncio task:
        summary_task = asyncio.create_task(
            create_summary_refresh_task(memory)()
        )
    """
    async def refresh_loop():
        while True:
            try:
                count = await memory.refresh_stale_summaries(threshold=10)
                if count > 0:
                    print(f"Refreshed {count} stale summaries")
            except Exception as e:
                print(f"Summary refresh error: {e}")

            await asyncio.sleep(interval_seconds)

    return refresh_loop
