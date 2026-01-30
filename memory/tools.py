"""
Memory Tools - Agent tools for interacting with the memory system.

Provides tools for:
- Quick retrieval (search, lookup)
- Deep retrieval (think deeper)
- Manual memory operations (store, note)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .system import MemorySystem


def create_memory_tool(memory: "MemorySystem") -> dict:
    """
    Create the memory tool definition.

    This tool gives the agent access to:
    - search: Quick search across events and entities
    - lookup: Get everything about a specific entity
    - recent: Get recent activity
    - think_deeper: Invoke deep retrieval for complex questions
    - store: Manually store a fact
    """

    def execute(params: dict, agent) -> dict:
        action = params.get("action", "search")

        if action == "search":
            query = params.get("query", "")
            entity_types = params.get("entity_types")
            limit = params.get("limit", 20)

            result = memory.quick_retrieve(query, entity_types, limit)

            return {
                "action": "search",
                "query": query,
                "found": not result.is_empty(),
                "events": len(result.events),
                "entities": [
                    {"name": e.name, "type": e.type}
                    for e in result.entities[:10]
                ],
                "summaries": list(result.summaries.keys()),
                "confidence": result.confidence,
            }

        elif action == "lookup":
            name = params.get("name", "")
            result = memory.get_entity(name)

            if result.entities:
                entity = result.entities[0]
                return {
                    "action": "lookup",
                    "found": True,
                    "entity": {
                        "name": entity.name,
                        "type": entity.type,
                        "attributes": entity.attributes,
                    },
                    "relationships": [
                        {
                            "type": e.type,
                            "target": memory.graph.get_entity(
                                e.target_id if e.source_id == entity.id else e.source_id
                            ).name
                        }
                        for e in result.edges[:10]
                        if memory.graph.get_entity(
                            e.target_id if e.source_id == entity.id else e.source_id
                        )
                    ],
                    "summary": result.summaries.get(f"person:{name.lower()}", ""),
                    "recent_events": len(result.events),
                }
            else:
                return {
                    "action": "lookup",
                    "found": False,
                    "name": name,
                }

        elif action == "recent":
            channel = params.get("channel")
            person = params.get("person")
            limit = params.get("limit", 20)

            result = memory.get_recent(channel, person, limit)

            return {
                "action": "recent",
                "events": [
                    {
                        "type": e.type,
                        "timestamp": e.timestamp[:16],
                        "preview": str(e.content)[:100],
                    }
                    for e in result.events
                ],
            }

        elif action == "think_deeper":
            # This requires async - return instruction to use deep retrieval
            question = params.get("question", "")
            return {
                "action": "think_deeper",
                "question": question,
                "instruction": "Call memory.deep_retrieve() with this question",
                "note": "Deep retrieval requires async execution",
            }

        elif action == "store":
            content = params.get("content", "")
            tags = params.get("tags", {})

            if not content:
                return {"action": "store", "stored": False, "error": "No content provided"}

            # Store as a manual memory event
            event = memory.log(
                type="manual_memory",
                content={"text": content},
                tags={"source": "manual", **tags},
            )

            return {
                "action": "store",
                "stored": True,
                "event_id": event.id,
            }

        elif action == "stats":
            return {
                "action": "stats",
                **memory.stats(),
            }

        else:
            return {"error": f"Unknown action: {action}"}

    return {
        "name": "memory",
        "description": """Access and search your memory system.

Actions:
- search: Search events and entities by keyword
- lookup: Get everything about a specific entity (person, org, etc.)
- recent: Get recent activity, optionally filtered by channel/person
- think_deeper: For complex questions, invoke deep retrieval
- store: Manually store a fact or note
- stats: Get memory system statistics

Examples:
- {"action": "search", "query": "funding round"}
- {"action": "lookup", "name": "John Smith"}
- {"action": "recent", "channel": "email", "limit": 10}
- {"action": "store", "content": "User prefers morning meetings"}""",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "lookup", "recent", "think_deeper", "store", "stats"],
                    "description": "The memory action to perform",
                },
                "query": {
                    "type": "string",
                    "description": "Search query (for search action)",
                },
                "name": {
                    "type": "string",
                    "description": "Entity name (for lookup action)",
                },
                "question": {
                    "type": "string",
                    "description": "Question to answer (for think_deeper action)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to store (for store action)",
                },
                "channel": {
                    "type": "string",
                    "description": "Filter by channel (for recent action)",
                },
                "person": {
                    "type": "string",
                    "description": "Filter by person (for recent/search actions)",
                },
                "entity_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by entity types (for search action)",
                },
                "tags": {
                    "type": "object",
                    "description": "Additional tags (for store action)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 20,
                },
            },
            "required": ["action"],
        },
        "execute": execute,
    }


def create_deep_retrieval_tool(memory: "MemorySystem") -> dict:
    """
    Create a dedicated deep retrieval tool.

    This is a separate tool for complex questions that require
    thorough investigation of the memory system.
    """

    async def execute_async(params: dict, agent) -> dict:
        question = params.get("question", "")

        if not question:
            return {"error": "No question provided"}

        result = await memory.deep_retrieve(question, agent)

        return {
            "question": question,
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "events_found": len(result.events),
            "entities_found": len(result.entities),
        }

    return {
        "name": "think_deeper",
        "description": """Thoroughly investigate a complex question using deep retrieval.

Use this when:
- Quick search doesn't give enough information
- Question requires reasoning across multiple sources
- Need to follow chains of relationships
- Answer requires synthesis from multiple events

This spawns a retrieval agent that navigates the memory system.
Takes longer but gives more thorough answers.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The complex question to investigate",
                },
            },
            "required": ["question"],
        },
        "execute_async": execute_async,
    }
