"""
Integration helpers for connecting the memory system to the main agent.

This module provides utilities to:
1. Hook memory logging into the agent event system
2. Provide memory context to the system prompt
3. Create memory-enhanced tools
4. Extract learnings from feedback and self-evaluation (self-improvement)
"""

import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

from .models import Event
from .learning import (
    FeedbackExtractor,
    ObjectiveEvaluator,
    PreferenceSummarizer,
    ensure_user_preferences_node,
)


def setup_memory_hooks(agent, memory):
    """
    Set up event hooks to automatically log agent activity to memory.

    Call this after creating both the agent and memory system:
        agent = Agent()
        memory = Memory()
        setup_memory_hooks(agent, memory)

    Enhanced hooks include:
    - Tool execution tracking with success/error statistics
    - Tool registration events for persistence
    - Automatic tool entity creation in the knowledge graph
    """

    @agent.on("tool_start")
    def on_tool_start(event):
        """Log tool calls to memory.

        Tool calls are logged but marked as 'skipped' for extraction since
        the tool_result event contains the actual extractable information.
        This eliminates ~37% of extraction LLM calls at high volume.
        """
        context = getattr(agent, "_current_context", {})
        tool_name = event["name"]

        # Log the event but mark as skipped for extraction
        # (tool_result contains the actual information worth extracting)
        logged_event = memory.log_event(
            content=f"Tool call: {tool_name}\nInput: {event['input']}",
            event_type="tool_call",
            channel=context.get("channel"),
            direction="internal",
            tool_id=tool_name,
            person_id=context.get("person_id"),
            is_owner=context.get("is_owner", True),
        )
        # Mark as skipped immediately so the extraction loop never picks it up
        try:
            memory.store.update_event_extraction_status(logged_event.id, "skipped")
        except Exception:
            pass

    @agent.on("tool_end")
    def on_tool_end(event):
        """Log tool results to memory and track statistics."""
        context = getattr(agent, "_current_context", {})
        tool_name = event["name"]
        result = event.get("result", "")
        duration_ms = event.get("duration_ms", 0)

        # Check if result indicates an error
        is_error = False
        error_msg = None

        if isinstance(result, dict):
            if "error" in result:
                is_error = True
                error_msg = str(result["error"])
            elif result.get("status") == "error":
                is_error = True
                error_msg = result.get("message", "Unknown error")

        if is_error:
            # Log error event (marked skipped for extraction — errors rarely contain extractable info)
            error_event = memory.log_event(
                content=f"Tool error: {tool_name}\nError: {error_msg}",
                event_type="tool_error",
                channel=context.get("channel"),
                direction="internal",
                tool_id=tool_name,
                person_id=context.get("person_id"),
                is_owner=context.get("is_owner", True),
                metadata={"error": error_msg, "duration_ms": duration_ms},
            )
            try:
                memory.store.update_event_extraction_status(error_event.id, "skipped")
            except Exception:
                pass
            # Update error statistics (safe - won't crash)
            try:
                memory.store.record_tool_error(tool_name, error_msg, duration_ms)
            except Exception as e:
                logger.debug("Failed to record tool error stats for '%s': %s", tool_name, e)
        else:
            # Log success event
            result_str = str(result)[:1000]  # Truncate large results
            memory.log_event(
                content=f"Tool result: {tool_name}\nResult: {result_str}",
                event_type="tool_result",
                channel=context.get("channel"),
                direction="internal",
                tool_id=tool_name,
                person_id=context.get("person_id"),
                is_owner=context.get("is_owner", True),
                metadata={"duration_ms": duration_ms},
            )
            # Update success statistics (safe - won't crash)
            try:
                memory.store.record_tool_success(tool_name, duration_ms)
            except Exception as e:
                logger.debug("Failed to record tool success stats for '%s': %s", tool_name, e)

    @agent.on("tool_registered")
    def on_tool_registered(event):
        """
        Handle tool registration - persist to database and create graph entity.

        This enables tool persistence across restarts (self-improvement).
        """
        try:
            tool_name = event["name"]
            description = event.get("description", "")
            parameters = event.get("parameters", {})
            source_code = event.get("source_code")
            packages = event.get("packages", [])
            env = event.get("env", [])
            tool_var_name = event.get("tool_var_name")
            category = event.get("category", "custom")
            is_dynamic = event.get("is_dynamic", True)

            # Save tool definition to database
            tool_def = memory.store.save_tool_definition(
                name=tool_name,
                description=description,
                parameters=parameters,
                source_code=source_code,
                packages=packages,
                env=env,
                tool_var_name=tool_var_name,
                category=category,
                is_dynamic=is_dynamic,
            )

            # Log the creation event
            memory.log_event(
                content=f"Tool created: {tool_name}\nDescription: {description}\nCategory: {category}",
                event_type="tool_created",
                channel=None,
                direction="internal",
                tool_id=tool_name,
                is_owner=True,
                metadata={
                    "tool_id": tool_def.id,
                    "version": tool_def.version,
                    "is_dynamic": is_dynamic,
                    "packages": packages,
                },
            )
        except Exception as e:
            # Log but don't crash - tool registration should still work
            # even if persistence fails
            try:
                memory.log_event(
                    content=f"Tool registration warning: {event.get('name', 'unknown')}\nError: {str(e)}",
                    event_type="tool_error",
                    channel=None,
                    direction="internal",
                    tool_id=event.get("name"),
                    is_owner=True,
                    metadata={"error": str(e), "phase": "registration"},
                )
            except Exception as e:
                logger.warning("Failed to persist tool registration event for '%s': %s", event.get('name', 'unknown'), e)

    @agent.on("tool_disabled")
    def on_tool_disabled(event):
        """Handle tool disabling."""
        try:
            tool_name = event["name"]
            reason = event.get("reason", "No reason provided")

            memory.log_event(
                content=f"Tool disabled: {tool_name}\nReason: {reason}",
                event_type="tool_disabled",
                channel=None,
                direction="internal",
                tool_id=tool_name,
                is_owner=True,
                metadata={"reason": reason},
            )
        except Exception as e:
            logger.debug("Failed to log tool disabled event for '%s': %s", event.get("name", "unknown"), e)

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

    # ═══════════════════════════════════════════════════════════
    # STATE SEEDING
    # ═══════════════════════════════════════════════════════════

    # Seed agent state and owner entity from config
    try:
        _seed_agent_state(agent.config, memory)
    except Exception as e:
        logger.debug("Could not seed agent state from config: %s", e)

    # ═══════════════════════════════════════════════════════════
    # SELF-IMPROVEMENT HOOKS
    # ═══════════════════════════════════════════════════════════

    # Ensure user_preferences node exists
    try:
        ensure_user_preferences_node(memory.store)
    except Exception as e:
        logger.debug("Could not ensure user preferences node: %s", e)

    @agent.on("objective_end")
    def on_objective_end_evaluate(event):
        """Evaluate completed objectives for learnings (self-improvement)."""
        if event.get("status") not in ["completed", "failed"]:
            return

        async def do_evaluate():
            try:
                # Get events from this objective
                objective_events = memory.store.get_recent_events(
                    limit=50,
                    task_id=event["id"]
                )

                if len(objective_events) < 3:
                    return  # Too short to evaluate

                evaluator = ObjectiveEvaluator()
                learnings = await evaluator.evaluate(
                    objective_id=event["id"],
                    goal=event.get("goal", ""),
                    status=event["status"],
                    result=event.get("result", event.get("error", "")),
                    events=objective_events
                )

                for learning in learnings:
                    memory.store.create_learning(learning)

                # Trigger preference summary update if learnings were generated
                if learnings:
                    memory.store.increment_staleness("user_preferences")

            except Exception as e:
                logger.warning("Objective evaluation error: %s", e)

        # Run evaluation asynchronously to not block
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(do_evaluate())
        except RuntimeError:
            # No event loop running, skip async evaluation
            pass


def _seed_agent_state(config: dict, memory):
    """Seed agent state and owner entity from config on startup.

    Ensures:
    - AgentState.name and .description reflect current config
    - An owner entity exists with is_owner=True
    - AgentState.owner_entity_id points to it

    This runs once at startup. It's idempotent — safe to call repeatedly.
    Config values always win (the owner controls config, the agent learns the rest).
    """
    agent_config = config.get("agent", {})
    owner_config = config.get("owner", {})

    state = memory.get_agent_state()
    updates = {}

    # Sync agent identity from config → state
    config_name = agent_config.get("name", "Assistant")
    config_desc = agent_config.get("description")
    if config_name and state.name != config_name:
        updates["name"] = config_name
    if config_desc and state.description != config_desc:
        updates["description"] = config_desc

    # Ensure owner entity exists
    if not state.owner_entity_id:
        owner_entity = _find_or_create_owner_entity(memory, owner_config)
        if owner_entity:
            updates["owner_entity_id"] = owner_entity.id

    if updates:
        memory.update_agent_state(**updates)


def _find_or_create_owner_entity(memory, owner_config: dict):
    """Find existing owner entity or create one from config.

    Looks for an entity with is_owner=True. If none exists, creates one
    using the owner's name (or "Owner") and contact info from config.
    """
    # Look for existing owner entity
    cur = memory.store.conn.cursor()
    cur.execute("SELECT id FROM entities WHERE is_owner = 1 LIMIT 1")
    row = cur.fetchone()
    if row:
        return memory.store.get_entity(row["id"])

    # Create owner entity from config
    owner_name = owner_config.get("name") or owner_config.get("id", "Owner")

    # Build description from available contact info
    desc_parts = []
    if owner_config.get("email"):
        desc_parts.append(f"Email: {owner_config['email']}")
    if owner_config.get("phone"):
        desc_parts.append(f"Phone: {owner_config['phone']}")
    if owner_config.get("timezone"):
        desc_parts.append(f"Timezone: {owner_config['timezone']}")
    description = "Agent owner" + (". " + ", ".join(desc_parts) if desc_parts else "")

    return memory.store.create_entity(
        name=owner_name,
        type="person",
        type_raw="owner",
        description=description,
        is_owner=True,
    )


def setup_feedback_extraction(agent, memory):
    """
    Set up feedback extraction from owner messages.

    This should be called separately as it requires async processing.
    Call this after setup_memory_hooks if you want automatic feedback extraction.
    """

    async def extract_feedback_from_message(event: Event):
        """Extract feedback from an owner message."""
        if not event.is_owner:
            return

        try:
            # Get recent AI actions for context
            recent = memory.store.get_recent_events(
                limit=10,
            )
            # Filter to outbound/internal only
            recent = [e for e in recent if e.direction != "inbound"][:5]

            extractor = FeedbackExtractor()
            learning = await extractor.extract(event, recent)

            if learning:
                memory.store.create_learning(learning)
                # Trigger preference summary update
                memory.store.increment_staleness("user_preferences")
                logger.debug("Extracted learning from feedback: %s...", learning.content[:50])

        except Exception as e:
            logger.warning("Feedback extraction error: %s", e)

    return extract_feedback_from_message


async def refresh_user_preferences(memory):
    """
    Manually refresh the user preferences summary.

    Call this periodically or after significant new learnings.
    """
    try:
        summarizer = PreferenceSummarizer()
        new_summary = await summarizer.refresh_preferences(memory.store)

        # Update the summary node
        prefs_node = memory.store.get_summary_node("user_preferences")
        if prefs_node:
            prefs_node.summary = new_summary
            prefs_node.events_since_update = 0
            memory.store.update_summary_node(prefs_node)

        return new_summary

    except Exception as e:
        logger.warning("Preference refresh error: %s", e)
        return None


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

        elif action == "list_tools":
            # List all tools I've created
            try:
                tools = memory.store.get_all_tool_definitions(include_disabled=True)
                return {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description[:100],
                            "category": t.category,
                            "is_enabled": t.is_enabled,
                            "is_dynamic": t.is_dynamic,
                            "usage_count": t.usage_count,
                            "success_rate": f"{t.success_rate:.1f}%",
                            "is_healthy": t.is_healthy,
                            "version": t.version,
                        }
                        for t in tools
                    ]
                }
            except Exception as e:
                return {"error": f"Could not list tools: {e}"}

        elif action == "tool_stats":
            # Get aggregate tool statistics
            try:
                stats = memory.store.get_tool_stats()
                return {"stats": stats}
            except Exception as e:
                return {"error": f"Could not get tool stats: {e}"}

        elif action == "problematic_tools":
            # Get tools with high error rates
            try:
                threshold = params.get("threshold", 5)
                problematic = memory.store.get_problematic_tools(error_threshold=threshold)
                unhealthy = memory.store.get_unhealthy_tools()
                return {
                    "high_error_count": [
                        {
                            "name": t.name,
                            "error_count": t.error_count,
                            "last_error": t.last_error,
                            "success_rate": f"{t.success_rate:.1f}%",
                        }
                        for t in problematic
                    ],
                    "low_success_rate": [
                        {
                            "name": t.name,
                            "usage_count": t.usage_count,
                            "success_rate": f"{t.success_rate:.1f}%",
                        }
                        for t in unhealthy
                    ],
                }
            except Exception as e:
                return {"error": f"Could not get problematic tools: {e}"}

        elif action == "get_tool":
            # Get details about a specific tool
            tool_name = params.get("tool_name")
            if not tool_name:
                return {"error": "tool_name required"}
            try:
                tool_def = memory.store.get_tool_definition(tool_name)
                if tool_def is None:
                    return {"error": f"Tool not found: {tool_name}"}
                return {
                    "tool": {
                        "name": tool_def.name,
                        "description": tool_def.description,
                        "category": tool_def.category,
                        "is_enabled": tool_def.is_enabled,
                        "is_dynamic": tool_def.is_dynamic,
                        "packages": tool_def.packages,
                        "usage_count": tool_def.usage_count,
                        "success_count": tool_def.success_count,
                        "error_count": tool_def.error_count,
                        "success_rate": f"{tool_def.success_rate:.1f}%",
                        "avg_duration_ms": tool_def.avg_duration_ms,
                        "last_error": tool_def.last_error,
                        "version": tool_def.version,
                        "created_at": tool_def.created_at.isoformat() if tool_def.created_at else None,
                    }
                }
            except Exception as e:
                return {"error": f"Could not get tool: {e}"}

        # ═══════════════════════════════════════════════════════════
        # LEARNING ACTIONS (Self-Improvement)
        # ═══════════════════════════════════════════════════════════

        elif action == "list_learnings":
            # List recent learnings
            try:
                tool_filter = params.get("tool_name")
                sentiment_filter = params.get("sentiment")
                limit = params.get("limit", 20)

                learnings = memory.store.find_learnings(
                    tool_id=tool_filter,
                    sentiment=sentiment_filter,
                    limit=limit,
                )
                return {
                    "learnings": [
                        {
                            "id": l.id,
                            "content": l.content[:200],
                            "source_type": l.source_type,
                            "sentiment": l.sentiment,
                            "tool_id": l.tool_id,
                            "objective_type": l.objective_type,
                            "recommendation": l.recommendation,
                            "times_applied": l.times_applied,
                            "created_at": l.created_at.isoformat() if l.created_at else None,
                        }
                        for l in learnings
                    ]
                }
            except Exception as e:
                return {"error": f"Could not list learnings: {e}"}

        elif action == "learning_stats":
            # Get aggregate learning statistics
            try:
                stats = memory.store.get_learning_stats()
                return {"stats": stats}
            except Exception as e:
                return {"error": f"Could not get learning stats: {e}"}

        elif action == "get_preferences":
            # Get current user preferences summary
            try:
                prefs_node = memory.store.get_summary_node("user_preferences")
                if prefs_node:
                    return {
                        "preferences": prefs_node.summary,
                        "last_updated": prefs_node.summary_updated_at.isoformat() if prefs_node.summary_updated_at else None,
                        "events_since_update": prefs_node.events_since_update,
                    }
                return {"preferences": "No preferences recorded yet."}
            except Exception as e:
                return {"error": f"Could not get preferences: {e}"}

        elif action == "search_learnings":
            # Semantic search over learnings
            query = params.get("query", "")
            if not query:
                return {"error": "Query required for learning search"}
            try:
                from .embeddings import get_embedding
                embedding = get_embedding(query)
                learnings = memory.store.search_learnings(
                    embedding=embedding,
                    tool_id=params.get("tool_name"),
                    limit=params.get("limit", 10),
                )
                return {
                    "learnings": [
                        {
                            "content": l.content[:200],
                            "sentiment": l.sentiment,
                            "tool_id": l.tool_id,
                            "recommendation": l.recommendation,
                        }
                        for l in learnings
                    ]
                }
            except Exception as e:
                return {"error": f"Could not search learnings: {e}"}

        elif action == "refresh_preferences":
            # Trigger preferences summary refresh
            try:
                async def do_refresh():
                    return await refresh_user_preferences(memory)

                try:
                    loop = asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        result = pool.submit(asyncio.run, do_refresh()).result()
                    return {"refreshed": True, "preferences": result}
                except RuntimeError:
                    result = asyncio.run(do_refresh())
                    return {"refreshed": True, "preferences": result}
            except Exception as e:
                return {"error": f"Could not refresh preferences: {e}"}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="memory",
        description="""Enhanced memory system with semantic search, graph storage, tool management, and self-improvement.

Memory Actions:
- store: Store a fact or observation (content)
- search: Semantic search over memories (query)
- list: List recent memories
- find_entity: Find people, orgs, concepts (query, optional type)
- find_relationship: Find relationships (query, optional entity_id)
- get_summary: Get pre-computed summary (key like 'root', 'entity:uuid', 'topic:uuid')
- get_context: Get current assembled context
- deep_search: Thorough agentic search for complex queries (query)

Tool Management Actions:
- list_tools: List all tools I've created with their status and health
- tool_stats: Get aggregate statistics about tool usage and success rates
- problematic_tools: Find tools with high error rates (optional threshold)
- get_tool: Get detailed info about a specific tool (tool_name)

Self-Improvement Actions (learnings from feedback and evaluation):
- list_learnings: List learnings (optional: tool_name, sentiment filter)
- learning_stats: Get aggregate statistics about learnings
- get_preferences: Get current user preferences summary
- search_learnings: Semantic search over learnings (query)
- refresh_preferences: Manually refresh the preferences summary

Examples:
- memory(action="store", content="John works at Acme Corp")
- memory(action="search", query="fundraising discussions")
- memory(action="find_entity", query="venture capital", type="person")
- memory(action="list_tools")
- memory(action="list_learnings", sentiment="negative")
- memory(action="get_preferences")
- memory(action="search_learnings", query="email formatting")""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["store", "search", "list", "find_entity", "find_relationship", "get_summary", "get_context", "deep_search", "list_tools", "tool_stats", "problematic_tools", "get_tool", "list_learnings", "learning_stats", "get_preferences", "search_learnings", "refresh_preferences"],
                },
                "content": {"type": "string", "description": "Content to store"},
                "query": {"type": "string", "description": "Search query"},
                "key": {"type": "string", "description": "Summary node key"},
                "type": {"type": "string", "description": "Entity type filter"},
                "entity_id": {"type": "string", "description": "Entity ID for relationship search"},
                "tool_name": {"type": "string", "description": "Tool name for filtering"},
                "threshold": {"type": "integer", "description": "Error threshold for problematic_tools"},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"], "description": "Sentiment filter for learnings"},
                "limit": {"type": "integer", "description": "Max results to return"},
            },
            "required": ["action"],
        },
        fn=fn,
    )


def create_extraction_background_task(
    memory,
    interval_seconds: int = 60,
    agent=None,
):
    """
    Create a priority-aware background task that processes pending extractions.

    Key optimizations over the original:
    - Batch extraction: combines multiple events into one LLM call (3-5x fewer API calls)
    - Activity-aware throttling: backs off when interactive messages are active
    - Adaptive intervals: processes faster when idle, slower when busy
    - Heuristic triage: skips low-value events without any LLM calls

    Args:
        memory: Memory system instance
        interval_seconds: Base interval between extraction cycles
        agent: Optional agent reference for activity-aware throttling

    Run this as an asyncio task:
        extraction_task = asyncio.create_task(
            create_extraction_background_task(memory, agent=agent)()
        )
    """
    from .extraction import ExtractionPipeline

    async def extraction_loop():
        # Initial delay to allow system startup to complete
        # Prevents API contention with CLI greeting and Uvicorn startup
        await asyncio.sleep(10)

        pipeline = ExtractionPipeline(memory.store)

        while True:
            try:
                # Check if interactive traffic is happening — yield to it
                sleep_interval = interval_seconds
                if agent is not None:
                    seconds_since_activity = _seconds_since_interactive_activity(agent)
                    if seconds_since_activity < 10:
                        # Active conversation — back off significantly
                        sleep_interval = max(interval_seconds * 3, 180)
                        await asyncio.sleep(sleep_interval)
                        continue
                    elif seconds_since_activity < 60:
                        # Recently active — use longer interval
                        sleep_interval = interval_seconds * 2

                # Get pending events
                events = memory.store.get_pending_extraction_events(limit=20)

                if events:
                    # Use batch extraction for efficiency
                    batch_size = pipeline.config.max_batch_size
                    for i in range(0, len(events), batch_size):
                        batch = events[i : i + batch_size]

                        # Re-check activity before each batch
                        if agent is not None and _seconds_since_interactive_activity(agent) < 10:
                            break  # Yield to interactive traffic

                        try:
                            await pipeline.extract_batch_combined(batch)
                        except Exception as e:
                            logger.warning("Batch extraction failed: %s", e)
                            # Fall back to individual extraction for this batch
                            for event in batch:
                                try:
                                    await pipeline.extract(event)
                                except Exception as e2:
                                    logger.debug("Individual extraction failed for %s: %s", event.id, e2)

                        # Brief yield between batches to let other tasks run
                        await asyncio.sleep(1)

                # Refresh stale summaries (less frequently)
                await memory.refresh_stale_summaries(threshold=10)

            except Exception as e:
                logger.warning("Extraction loop error: %s", e)

            await asyncio.sleep(sleep_interval)

    return extraction_loop


def _seconds_since_interactive_activity(agent) -> float:
    """Get seconds since the agent last processed an interactive message.

    Returns a large number if the agent has no activity tracking,
    which means extraction runs at full speed.
    """
    import time
    last_activity = getattr(agent, "_last_interactive_activity", None)
    if last_activity is None:
        return 999.0
    return time.time() - last_activity


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
                    logger.info("Refreshed %d stale summaries", count)
            except Exception as e:
                logger.warning("Summary refresh error: %s", e)

            await asyncio.sleep(interval_seconds)

    return refresh_loop
