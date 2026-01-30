"""
Retrieval - Quick and deep retrieval mechanisms.

Two modes:
1. Quick Retrieval - Programmatic, deterministic, fast (<100ms)
   Used when context is insufficient but answer is straightforward.
   Returns structured data: entities, edges, events, summaries.

2. Deep Retrieval - Agent-based, thorough, slower
   Used when quick retrieval isn't enough.
   A separate agent navigates the memory system to synthesize an answer.
"""

from dataclasses import dataclass, field
from typing import Any
from .models import Event, Entity, Edge, SliceKey
from .event_log import EventLog
from .graph import Graph
from .summaries import SummaryTree


@dataclass
class RetrievalResult:
    """Result from a retrieval operation."""

    query: str
    events: list[Event] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    summaries: dict[str, str] = field(default_factory=dict)  # slice_key -> summary text
    answer: str | None = None  # Synthesized answer (from deep retrieval)
    sources: list[str] = field(default_factory=list)  # Source references
    confidence: float = 0.0  # 0-1 confidence in the result

    def is_empty(self) -> bool:
        return not (self.events or self.entities or self.edges or self.summaries)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "events": [e.to_dict() for e in self.events],
            "entities": [e.to_dict() for e in self.entities],
            "edges": [e.to_dict() for e in self.edges],
            "summaries": self.summaries,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
        }


class QuickRetrieval:
    """
    Fast, programmatic retrieval without LLM calls.

    Use this when:
    - Looking up specific entities
    - Finding recent events about something
    - Getting pre-computed summaries
    - Checking relationships

    All operations are <100ms.
    """

    def __init__(
        self,
        event_log: EventLog,
        graph: Graph,
        summary_tree: SummaryTree,
    ):
        self.event_log = event_log
        self.graph = graph
        self.summary_tree = summary_tree

    def search(
        self,
        query: str,
        entity_types: list[str] | None = None,
        event_types: list[str] | None = None,
        limit: int = 20,
    ) -> RetrievalResult:
        """
        Search across events and entities.

        This is a simple keyword search. For semantic search,
        add embeddings to the event log and graph.
        """
        result = RetrievalResult(query=query)

        # Search entities
        entities = self.graph.search_entities(
            query=query,
            types=entity_types,
            limit=limit,
        )
        result.entities = entities

        # Get related edges for found entities
        for entity in entities[:5]:  # Limit edge expansion
            edges = self.graph.get_edges(entity.id)
            result.edges.extend(edges)

        # Search events (simple substring match for now)
        query_lower = query.lower()
        matching_events = []
        for event in self.event_log.iter_events(reverse=True):
            content_str = str(event.content).lower()
            if query_lower in content_str:
                matching_events.append(event)
            if len(matching_events) >= limit:
                break

        if event_types:
            matching_events = [e for e in matching_events if e.type in event_types]

        result.events = matching_events

        # Calculate confidence based on matches found
        if result.entities or result.events:
            result.confidence = min(0.8, 0.2 * len(result.entities) + 0.1 * len(result.events))

        return result

    def get_entity_context(self, name: str) -> RetrievalResult:
        """Get everything we know about an entity."""
        result = RetrievalResult(query=f"entity:{name}")

        entity = self.graph.find_entity(name)
        if not entity:
            return result

        result.entities = [entity]
        result.confidence = 0.9

        # Get relationships
        result.edges = self.graph.get_edges(entity.id)

        # Get summary if exists
        person_slice = SliceKey({"person": name.lower()})
        summary_text = self.summary_tree.get_text(person_slice)
        if summary_text:
            result.summaries[person_slice.key] = summary_text

        # Get recent events mentioning this entity
        result.events = self.event_log.recent(20, slice_key=person_slice)

        result.sources = [f"entity:{entity.id}", person_slice.key]
        return result

    def get_slice_context(self, slice_key: SliceKey) -> RetrievalResult:
        """Get summary and events for a slice."""
        result = RetrievalResult(query=slice_key.key)

        # Get summary
        summary_text = self.summary_tree.get_text(slice_key)
        if summary_text:
            result.summaries[slice_key.key] = summary_text
            result.confidence = 0.7

        # Get events
        result.events = self.event_log.recent(20, slice_key=slice_key)
        if result.events:
            result.confidence = max(result.confidence, 0.5)

        result.sources = [slice_key.key]
        return result

    def get_recent_activity(
        self,
        channel: str | None = None,
        person: str | None = None,
        limit: int = 20,
    ) -> RetrievalResult:
        """Get recent events, optionally filtered."""
        tags = {}
        if channel:
            tags["channel"] = channel
        if person:
            tags["person"] = person

        slice_key = SliceKey(tags) if tags else SliceKey.root()
        result = RetrievalResult(query=f"recent:{slice_key.key}")

        result.events = self.event_log.recent(limit, slice_key=slice_key)
        result.confidence = 0.9 if result.events else 0.0
        result.sources = [slice_key.key]

        return result

    def get_relationships(self, entity_name: str) -> RetrievalResult:
        """Get all relationships for an entity."""
        result = RetrievalResult(query=f"relationships:{entity_name}")

        entity = self.graph.find_entity(entity_name)
        if not entity:
            return result

        result.entities = [entity]
        neighbors = self.graph.get_neighbors(entity.id)

        for neighbor, edge in neighbors:
            result.entities.append(neighbor)
            result.edges.append(edge)

        result.confidence = 0.9 if neighbors else 0.3
        result.sources = [f"entity:{entity.id}"]

        return result


class DeepRetrieval:
    """
    Agent-based deep retrieval for complex questions.

    Use this when:
    - Quick retrieval returns insufficient results
    - Question requires reasoning across multiple sources
    - Need to follow chains of relationships
    - Answer requires synthesis from multiple events

    The main agent sends "let me think deeper..." then invokes
    this retrieval, which returns a synthesized answer with sources.
    """

    def __init__(
        self,
        event_log: EventLog,
        graph: Graph,
        summary_tree: SummaryTree,
    ):
        self.event_log = event_log
        self.graph = graph
        self.summary_tree = summary_tree

    def get_tools(self) -> list[dict]:
        """
        Get tool definitions for the deep retrieval agent.

        These tools allow the agent to navigate the memory system.
        """
        return [
            {
                "name": "search_events",
                "description": "Search through event history. Returns events matching the query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query (keywords)",
                        },
                        "event_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by event types",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max events to return",
                            "default": 20,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_summary",
                "description": "Get the pre-computed summary for a slice of data. Slices are like 'channel:email', 'person:john', 'topic:fundraising'.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "slice_key": {
                            "type": "string",
                            "description": "The slice key, e.g., 'channel:email', 'person:john', '*' for root",
                        },
                    },
                    "required": ["slice_key"],
                },
            },
            {
                "name": "find_entity",
                "description": "Find an entity (person, organization, topic) by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Name or alias to search for",
                        },
                    },
                    "required": ["name"],
                },
            },
            {
                "name": "get_relationships",
                "description": "Get all relationships for an entity.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_name": {
                            "type": "string",
                            "description": "Name of the entity",
                        },
                    },
                    "required": ["entity_name"],
                },
            },
            {
                "name": "list_slices",
                "description": "List all available summary slices. Useful to see what data is available.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
            {
                "name": "list_entity_types",
                "description": "List all entity types in the graph.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def execute_tool(self, name: str, params: dict) -> dict:
        """Execute a deep retrieval tool."""
        if name == "search_events":
            query = params.get("query", "")
            event_types = params.get("event_types")
            limit = params.get("limit", 20)

            events = []
            query_lower = query.lower()
            for event in self.event_log.iter_events(reverse=True):
                if event_types and event.type not in event_types:
                    continue
                if query_lower in str(event.content).lower():
                    events.append(event.to_dict())
                if len(events) >= limit:
                    break

            return {"events": events, "count": len(events)}

        elif name == "get_summary":
            slice_key = params.get("slice_key", "*")
            sk = SliceKey.from_key(slice_key)
            text = self.summary_tree.get_text(sk)
            summary = self.summary_tree.get(sk)

            return {
                "slice_key": slice_key,
                "summary": text or "(no summary available)",
                "event_count": summary.event_count if summary else 0,
                "stale": summary.stale if summary else False,
            }

        elif name == "find_entity":
            name_param = params.get("name", "")
            entity = self.graph.find_entity(name_param)

            if entity:
                return {"found": True, "entity": entity.to_dict()}
            else:
                # Try search
                entities = self.graph.search_entities(query=name_param, limit=5)
                return {
                    "found": False,
                    "similar": [e.to_dict() for e in entities],
                }

        elif name == "get_relationships":
            entity_name = params.get("entity_name", "")
            entity = self.graph.find_entity(entity_name)

            if not entity:
                return {"error": f"Entity '{entity_name}' not found"}

            neighbors = self.graph.get_neighbors(entity.id)
            relationships = []
            for neighbor, edge in neighbors:
                rel = {
                    "entity": neighbor.name,
                    "entity_type": neighbor.type,
                    "relationship": edge.type,
                    "direction": "outgoing" if edge.source_id == entity.id else "incoming",
                }
                relationships.append(rel)

            return {
                "entity": entity.name,
                "relationships": relationships,
            }

        elif name == "list_slices":
            slices = self.summary_tree.get_all_slice_keys()
            return {"slices": slices, "count": len(slices)}

        elif name == "list_entity_types":
            types = self.graph.get_entity_types()
            return {"types": types}

        else:
            return {"error": f"Unknown tool: {name}"}

    async def retrieve(
        self,
        question: str,
        agent,  # The main agent instance
        max_turns: int = 5,
    ) -> RetrievalResult:
        """
        Perform deep retrieval using an agent.

        Args:
            question: The question to answer
            agent: The main agent instance (for LLM access)
            max_turns: Maximum agent turns

        Returns:
            RetrievalResult with synthesized answer
        """
        result = RetrievalResult(query=question)

        # Build system prompt for retrieval agent
        system = """You are a memory retrieval agent. Your job is to search through
the memory system to answer a question.

You have access to:
- Event log: Immutable record of everything that happened
- Graph: Entities (people, orgs, topics) and their relationships
- Summary tree: Pre-computed summaries for different slices of data

Use the tools to explore the memory, then synthesize an answer.
Be thorough but efficient. Return your answer with source references."""

        # Create messages for the retrieval agent
        messages = [{"role": "user", "content": question}]

        # Run retrieval loop
        tools = self.get_tools()
        sources = set()

        for _ in range(max_turns):
            # Call LLM
            response = await agent._call_api(
                system=system,
                messages=messages,
                tools=tools,
            )

            # Check for tool calls
            tool_calls = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            if not tool_calls:
                # No more tool calls - extract answer
                for block in response.content:
                    if hasattr(block, "text"):
                        result.answer = block.text
                        break
                break

            # Execute tools
            tool_results = []
            for call in tool_calls:
                tool_result = self.execute_tool(call.name, call.input)
                sources.add(f"tool:{call.name}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": str(tool_result),
                })

            # Add to messages
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        result.sources = list(sources)
        result.confidence = 0.7 if result.answer else 0.2

        return result
