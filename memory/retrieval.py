"""
Retrieval system for the memory system.

Quick Retrieval: Programmatic, fast (<100ms), deterministic
Deep Retrieval: Agentic, thorough, invoked for complex queries
"""

from datetime import datetime
from typing import Any

from metrics import LiteLLMAnthropicAdapter, track_source, get_model_for_use_case
from .embeddings import cosine_similarity, get_embedding
from .models import Edge, Entity, Event, Fact, SummaryNode, Task, Topic


class QuickRetrieval:
    """
    Fast, programmatic retrieval.

    Used when context is insufficient but answer is straightforward.
    Returns structured data: entities, edges, events, summaries.
    """

    def __init__(self, store):
        self.store = store

    # ═══════════════════════════════════════════════════════════
    # DIRECT LOOKUPS
    # ═══════════════════════════════════════════════════════════

    def get_summary(self, key: str) -> SummaryNode | None:
        """Get a summary by key (e.g., 'entity:{id}', 'channel:email')."""
        return self.store.get_summary_node(key)

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        return self.store.get_entity(entity_id)

    def get_topic(self, topic_id: str) -> Topic | None:
        """Get a topic by ID."""
        return self.store.get_topic(topic_id)

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        return self.store.get_task(task_id)

    def get_edges(self, entity_id: str, direction: str = "both") -> list[Edge]:
        """Get edges for an entity."""
        return self.store.get_edges(entity_id, direction)

    # ═══════════════════════════════════════════════════════════
    # FILTERED QUERIES
    # ═══════════════════════════════════════════════════════════

    def find_events(
        self,
        channel: str | None = None,
        person_id: str | None = None,
        task_id: str | None = None,
        tool_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        limit: int = 20,
    ) -> list[Event]:
        """Find events by filters."""
        return self.store.get_recent_events(
            limit=limit,
            channel=channel,
            person_id=person_id,
            task_id=task_id,
            tool_id=tool_id,
        )

    def find_entities(
        self, query: str | None = None, type: str | None = None, limit: int = 10
    ) -> list[Entity]:
        """Find entities by name or type."""
        return self.store.find_entities(query, type, limit)

    def find_topics(self, query: str | None = None, limit: int = 10) -> list[Topic]:
        """Find topics by label search."""
        return self.store.find_topics(query, limit)

    def find_tasks(
        self,
        status: str | None = None,
        type_cluster: str | None = None,
        person_id: str | None = None,
        limit: int = 10,
    ) -> list[Task]:
        """Find tasks by filters."""
        return self.store.find_tasks(status, type_cluster, person_id, limit)

    # ═══════════════════════════════════════════════════════════
    # SEMANTIC SEARCH
    # ═══════════════════════════════════════════════════════════

    def search_events(self, query: str, limit: int = 10) -> list[Event]:
        """
        Find events semantically similar to query.

        Example: search_events("discussions about funding rounds")
        """
        query_embedding = get_embedding(query)

        # Get recent events with embeddings
        events = self.store.get_recent_events(limit=100)  # Get more for filtering
        events_with_embeddings = [e for e in events if e.content_embedding]

        if not events_with_embeddings:
            return []

        # Calculate similarities and sort
        scored = []
        for event in events_with_embeddings:
            sim = cosine_similarity(query_embedding, event.content_embedding)
            scored.append((event, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    def search_entities(
        self, query: str, type: str | None = None, limit: int = 10
    ) -> list[Entity]:
        """
        Find entities similar to query.

        Example: search_entities("venture capital investors")
        Example: search_entities("machine learning frameworks", type="tool")
        """
        query_embedding = get_embedding(query)

        # Get entities with embeddings
        entities = self.store.find_entities(limit=100)  # Get more for filtering
        if type:
            entities = [e for e in entities if e.type == type]

        entities_with_embeddings = [e for e in entities if e.name_embedding]

        if not entities_with_embeddings:
            # Fall back to text search
            return self.store.find_entities(query, type, limit)

        # Calculate similarities and sort
        scored = []
        for entity in entities_with_embeddings:
            sim = cosine_similarity(query_embedding, entity.name_embedding)
            scored.append((entity, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    def search_edges(
        self, query: str, entity_id: str | None = None, limit: int = 10
    ) -> list[Edge]:
        """
        Find relationships similar to query.

        Example: search_edges("investment relationships")
        Example: search_edges("worked together", entity_id=john_id)
        """
        query_embedding = get_embedding(query)

        # Get edges
        if entity_id:
            edges = self.store.get_edges(entity_id)
        else:
            # Get all edges (limited) - this is expensive
            cur = self.store.conn.cursor()
            cur.execute("SELECT * FROM edges WHERE is_current = 1 LIMIT 500")
            edges = [self.store._row_to_edge(row) for row in cur.fetchall()]

        edges_with_embeddings = [e for e in edges if e.relation_embedding]

        if not edges_with_embeddings:
            return edges[:limit]

        # Calculate similarities and sort
        scored = []
        for edge in edges_with_embeddings:
            sim = cosine_similarity(query_embedding, edge.relation_embedding)
            scored.append((edge, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:limit]]

    def search_topics(self, query: str, limit: int = 10) -> list[Topic]:
        """
        Find topics related to query.

        Example: search_topics("startup financing")
        """
        query_embedding = get_embedding(query)

        # Get topics with embeddings
        topics = self.store.find_topics(limit=100)
        topics_with_embeddings = [t for t in topics if t.embedding]

        if not topics_with_embeddings:
            return self.store.find_topics(query, limit)

        # Calculate similarities and sort
        scored = []
        for topic in topics_with_embeddings:
            sim = cosine_similarity(query_embedding, topic.embedding)
            scored.append((topic, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:limit]]

    def search_tasks(
        self, query: str, status: str | None = None, limit: int = 10
    ) -> list[Task]:
        """
        Find tasks similar to query.

        Example: search_tasks("competitive analysis")
        """
        query_embedding = get_embedding(query)

        # Get tasks with embeddings
        tasks = self.store.find_tasks(status=status, limit=100)
        tasks_with_embeddings = [t for t in tasks if t.type_embedding]

        if not tasks_with_embeddings:
            return tasks[:limit]

        # Calculate similarities and sort
        scored = []
        for task in tasks_with_embeddings:
            sim = cosine_similarity(query_embedding, task.type_embedding)
            scored.append((task, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [t for t, _ in scored[:limit]]

    def search_summaries(
        self, query: str, node_type: str | None = None, limit: int = 10
    ) -> list[SummaryNode]:
        """
        Find summaries about a topic across the tree.

        Example: search_summaries("what I know about AI")
        Example: search_summaries("fundraising", node_type="topic")
        """
        query_embedding = get_embedding(query)

        # Get summary nodes
        cur = self.store.conn.cursor()
        if node_type:
            cur.execute(
                "SELECT * FROM summary_nodes WHERE node_type = ? LIMIT 200",
                (node_type,),
            )
        else:
            cur.execute("SELECT * FROM summary_nodes LIMIT 200")

        nodes = [self.store._row_to_summary_node(row) for row in cur.fetchall()]
        nodes_with_embeddings = [n for n in nodes if n.summary_embedding]

        if not nodes_with_embeddings:
            # Fall back to text search on summaries
            scored = []
            query_lower = query.lower()
            for node in nodes:
                if query_lower in node.summary.lower() or query_lower in node.label.lower():
                    scored.append(node)
            return scored[:limit]

        # Calculate similarities and sort
        scored = []
        for node in nodes_with_embeddings:
            sim = cosine_similarity(query_embedding, node.summary_embedding)
            scored.append((node, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [n for n, _ in scored[:limit]]

    def search_facts(
        self, query: str, fact_type: str | None = None, source_type: str | None = None, limit: int = 10
    ) -> list[Fact]:
        """
        Find facts semantically similar to query.

        Example: search_facts("revenue figures for Q4")
        Example: search_facts("investment relationships", fact_type="relation")
        Example: search_facts("company metrics", source_type="document")
        """
        query_embedding = get_embedding(query)

        # Get facts with embeddings
        cur = self.store.conn.cursor()
        conditions = ["is_current = 1", "fact_embedding IS NOT NULL"]
        params = []

        if fact_type:
            conditions.append("fact_type = ?")
            params.append(fact_type)
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)

        where_clause = " AND ".join(conditions)
        cur.execute(
            f"SELECT * FROM facts WHERE {where_clause} LIMIT 500",
            params,
        )
        facts = [self.store._row_to_fact(row) for row in cur.fetchall()]

        if not facts:
            # Fall back to text search on fact_text
            cur.execute(
                "SELECT * FROM facts WHERE is_current = 1 LIMIT 500"
            )
            all_facts = [self.store._row_to_fact(row) for row in cur.fetchall()]
            query_lower = query.lower()
            return [f for f in all_facts if query_lower in f.fact_text.lower()][:limit]

        # Calculate similarities and sort
        scored = []
        for fact in facts:
            sim = cosine_similarity(query_embedding, fact.fact_embedding)
            scored.append((fact, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f for f, _ in scored[:limit]]

    # ═══════════════════════════════════════════════════════════
    # NAVIGATION
    # ═══════════════════════════════════════════════════════════

    def get_children(self, node_id: str) -> list[SummaryNode]:
        """Get child nodes in summary tree."""
        return self.store.get_children(node_id)

    def get_parent(self, node_id: str) -> SummaryNode | None:
        """Get parent node in summary tree."""
        return self.store.get_parent(node_id)

    def get_sources(self, node_key: str, limit: int = 20) -> list[Event]:
        """
        Get the raw events that support a summary.

        The node's key implicitly defines the query:
        - entity:{id} -> events with person_id={id}
        - channel:{name} -> events with channel={name}
        - tool:{id} -> events with tool_id={id}
        - topic:{id} -> events linked to topic
        - task:{id} -> events with task_id={id}
        """
        node = self.store.get_summary_node(node_key)
        if not node:
            return []

        parts = node_key.split(":", 1)
        if len(parts) != 2:
            if node_key == "root":
                return self.store.get_recent_events(limit=limit)
            return []

        prefix, value = parts

        if prefix == "entity":
            return self.store.get_recent_events(limit=limit, person_id=value)
        elif prefix == "channel":
            return self.store.get_recent_events(limit=limit, channel=value)
        elif prefix == "tool":
            return self.store.get_recent_events(limit=limit, tool_id=value)
        elif prefix == "topic":
            return self.store.get_events_for_topic(value, limit=limit)
        elif prefix == "task":
            return self.store.get_recent_events(limit=limit, task_id=value)
        elif prefix in ("entity_type", "task_type", "relation_type"):
            # Get children, then get their events
            children = self.store.get_children(node.id)
            events = []
            for child in children[:10]:  # Limit children
                child_events = self.get_sources(child.key, limit=5)
                events.extend(child_events)
            return events[:limit]
        else:
            return self.store.get_recent_events(limit=limit)

    # ═══════════════════════════════════════════════════════════
    # GRAPH TRAVERSAL
    # ═══════════════════════════════════════════════════════════

    def traverse(
        self,
        entity_id: str,
        max_depth: int = 2,
        relation_types: list[str] | None = None,
    ) -> dict:
        """
        Traverse graph from starting entity.

        Returns a subgraph with nodes and edges.
        """
        visited_entities = set()
        visited_edges = set()
        nodes = []
        edges = []

        def _traverse(eid: str, depth: int):
            if depth > max_depth or eid in visited_entities:
                return

            visited_entities.add(eid)
            entity = self.store.get_entity(eid)
            if entity:
                nodes.append(entity)

            # Get edges
            entity_edges = self.store.get_edges(eid)
            for edge in entity_edges:
                if edge.id in visited_edges:
                    continue
                if relation_types and edge.relation_type not in relation_types:
                    continue

                visited_edges.add(edge.id)
                edges.append(edge)

                # Traverse to connected entity
                next_id = (
                    edge.target_entity_id
                    if edge.source_entity_id == eid
                    else edge.source_entity_id
                )
                _traverse(next_id, depth + 1)

        _traverse(entity_id, 0)

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def find_connections(
        self, entity_a_id: str, entity_b_id: str, max_depth: int = 3
    ) -> list[list[dict]]:
        """
        Find paths connecting two entities.

        Returns list of paths, each path is a list of steps.
        """
        if entity_a_id == entity_b_id:
            return [[]]

        # BFS to find shortest paths
        from collections import deque

        queue = deque([(entity_a_id, [])])
        visited = {entity_a_id}
        paths = []

        while queue and len(paths) < 5:  # Limit to 5 paths
            current_id, path = queue.popleft()

            if len(path) > max_depth:
                continue

            edges = self.store.get_edges(current_id)
            for edge in edges:
                next_id = (
                    edge.target_entity_id
                    if edge.source_entity_id == current_id
                    else edge.source_entity_id
                )

                new_path = path + [
                    {
                        "from": current_id,
                        "to": next_id,
                        "relation": edge.relation,
                        "edge_id": edge.id,
                    }
                ]

                if next_id == entity_b_id:
                    paths.append(new_path)
                elif next_id not in visited:
                    visited.add(next_id)
                    queue.append((next_id, new_path))

        return paths


class DeepRetrievalAgent:
    """
    Agentic retrieval for complex queries.

    Invoked when quick retrieval isn't enough.
    Uses LLM to navigate and synthesize from memory.
    """

    def __init__(self, store, retrieval: QuickRetrieval):
        self.store = store
        self.retrieval = retrieval
        self._client = None

    @property
    def client(self):
        """Get instrumented LLM client for metrics tracking (supports multiple providers)."""
        if self._client is None:
            self._client = LiteLLMAnthropicAdapter()
        return self._client

    @property
    def model(self) -> str:
        """Get the configured model for retrieval operations."""
        return get_model_for_use_case("memory")

    def get_tools(self) -> list[dict]:
        """Get tool definitions for the retrieval agent."""
        return [
            # Navigation
            {
                "name": "get_summary",
                "description": "Get the summary for any node. Key format: 'root', 'channel:email', 'entity:{id}', 'topic:{id}'",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The node key"}
                    },
                    "required": ["key"],
                },
            },
            {
                "name": "get_children",
                "description": "Get child nodes of a summary node",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "node_id": {"type": "string", "description": "The node ID"}
                    },
                    "required": ["node_id"],
                },
            },
            {
                "name": "get_sources",
                "description": "Get the raw events that support a summary",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "node_key": {"type": "string"},
                        "limit": {"type": "integer", "default": 20},
                    },
                    "required": ["node_key"],
                },
            },
            # Semantic search
            {
                "name": "search_events",
                "description": "Find events semantically similar to a query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_entities",
                "description": "Find entities similar to a description. Can filter by type.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "type": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_edges",
                "description": "Find relationships similar to a description",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "entity_id": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_topics",
                "description": "Find topics related to a query",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "search_facts",
                "description": "Find facts semantically similar to a query. Can filter by fact_type (relation, attribute, event, state, metric) or source_type (conversation, document, tool, observation).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "fact_type": {"type": "string"},
                        "source_type": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            # Graph
            {
                "name": "get_entity",
                "description": "Get full entity details",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"}
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "get_edges",
                "description": "Get all relationships for an entity",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "direction": {
                            "type": "string",
                            "enum": ["outgoing", "incoming", "both"],
                            "default": "both",
                        },
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "traverse",
                "description": "Follow relationships from an entity up to N hops",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "max_depth": {"type": "integer", "default": 2},
                        "relation_types": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["entity_id"],
                },
            },
            {
                "name": "find_connections",
                "description": "Find paths connecting two entities",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_a_id": {"type": "string"},
                        "entity_b_id": {"type": "string"},
                        "max_depth": {"type": "integer", "default": 3},
                    },
                    "required": ["entity_a_id", "entity_b_id"],
                },
            },
        ]

    def execute_tool(self, name: str, input: dict) -> Any:
        """Execute a retrieval tool."""
        if name == "get_summary":
            node = self.retrieval.get_summary(input["key"])
            if node:
                return {
                    "key": node.key,
                    "label": node.label,
                    "summary": node.summary,
                    "event_count": node.event_count,
                }
            return None

        elif name == "get_children":
            children = self.retrieval.get_children(input["node_id"])
            return [
                {"id": c.id, "key": c.key, "label": c.label, "summary": c.summary[:200]}
                for c in children
            ]

        elif name == "get_sources":
            events = self.retrieval.get_sources(
                input["node_key"], input.get("limit", 20)
            )
            return [
                {
                    "id": e.id,
                    "content": e.content[:500],
                    "channel": e.channel,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                }
                for e in events
            ]

        elif name == "search_events":
            events = self.retrieval.search_events(
                input["query"], input.get("limit", 10)
            )
            return [
                {
                    "id": e.id,
                    "content": e.content[:500],
                    "channel": e.channel,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                }
                for e in events
            ]

        elif name == "search_entities":
            entities = self.retrieval.search_entities(
                input["query"], input.get("type"), input.get("limit", 10)
            )
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "type_raw": e.type_raw,
                    "description": e.description,
                }
                for e in entities
            ]

        elif name == "search_edges":
            edges = self.retrieval.search_edges(
                input["query"], input.get("entity_id"), input.get("limit", 10)
            )
            return [
                {
                    "id": e.id,
                    "source": e.source_entity_id,
                    "target": e.target_entity_id,
                    "relation": e.relation,
                }
                for e in edges
            ]

        elif name == "search_topics":
            topics = self.retrieval.search_topics(
                input["query"], input.get("limit", 10)
            )
            return [
                {"id": t.id, "label": t.label, "keywords": t.keywords}
                for t in topics
            ]

        elif name == "search_facts":
            facts = self.retrieval.search_facts(
                input["query"],
                input.get("fact_type"),
                input.get("source_type"),
                input.get("limit", 10),
            )
            return [
                {
                    "id": f.id,
                    "fact_text": f.fact_text,
                    "fact_type": f.fact_type,
                    "subject": f.subject_entity_id,
                    "predicate": f.predicate,
                    "source_type": f.source_type,
                    "confidence": f.confidence,
                }
                for f in facts
            ]

        elif name == "get_entity":
            entity = self.retrieval.get_entity(input["entity_id"])
            if entity:
                return {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.type,
                    "type_raw": entity.type_raw,
                    "aliases": entity.aliases,
                    "description": entity.description,
                    "event_count": entity.event_count,
                }
            return None

        elif name == "get_edges":
            edges = self.retrieval.get_edges(
                input["entity_id"], input.get("direction", "both")
            )
            result = []
            for e in edges:
                edge_info = {
                    "id": e.id,
                    "relation": e.relation,
                    "strength": e.strength,
                }
                # Add entity names
                if e.source_entity_id == input["entity_id"]:
                    target = self.store.get_entity(e.target_entity_id)
                    edge_info["direction"] = "outgoing"
                    edge_info["other"] = target.name if target else e.target_entity_id
                else:
                    source = self.store.get_entity(e.source_entity_id)
                    edge_info["direction"] = "incoming"
                    edge_info["other"] = source.name if source else e.source_entity_id
                result.append(edge_info)
            return result

        elif name == "traverse":
            subgraph = self.retrieval.traverse(
                input["entity_id"],
                input.get("max_depth", 2),
                input.get("relation_types"),
            )
            return {
                "nodes": [
                    {"id": n.id, "name": n.name, "type": n.type}
                    for n in subgraph["nodes"]
                ],
                "edges": [
                    {
                        "source": e.source_entity_id,
                        "target": e.target_entity_id,
                        "relation": e.relation,
                    }
                    for e in subgraph["edges"]
                ],
            }

        elif name == "find_connections":
            paths = self.retrieval.find_connections(
                input["entity_a_id"],
                input["entity_b_id"],
                input.get("max_depth", 3),
            )
            # Enrich paths with entity names
            enriched_paths = []
            for path in paths:
                enriched_path = []
                for step in path:
                    from_entity = self.store.get_entity(step["from"])
                    to_entity = self.store.get_entity(step["to"])
                    enriched_path.append({
                        "from": from_entity.name if from_entity else step["from"],
                        "relation": step["relation"],
                        "to": to_entity.name if to_entity else step["to"],
                    })
                enriched_paths.append(enriched_path)
            return enriched_paths

        return {"error": f"Unknown tool: {name}"}

    async def run(self, query: str, context: dict | None = None) -> dict:
        """
        Run deep retrieval for a complex query.

        Returns:
            {
                "answer": str,
                "confidence": "high" | "medium" | "low",
                "sources": [{"type": str, "id": str, "excerpt": str}]
            }
        """
        system_prompt = """You are a Memory Retrieval Agent. Your job is to search through memory to answer questions.

You have access to:
- A summary tree where every dimension (channels, entities, topics, tasks) has a pre-computed summary
- Semantic search across events, entities, relationships, topics, and tasks
- Graph traversal to explore relationships

Strategy:
1. Start with semantic search to find relevant summaries or entities
2. Use get_sources to drill down to supporting events
3. Use graph traversal for relationship questions
4. Always cite sources (event IDs, entity names)

Be thorough but efficient. Stop when you have enough information to answer confidently."""

        messages = [{"role": "user", "content": query}]

        if context:
            messages[0]["content"] = f"Context:\n{context}\n\nQuestion: {query}"

        tools = self.get_tools()
        max_turns = 10

        for _ in range(max_turns):
            with track_source("retrieval"):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )

            # Check if we're done
            if response.stop_reason == "end_turn":
                # Extract final answer
                for block in response.content:
                    if hasattr(block, "text"):
                        return self._parse_answer(block.text)
                return {
                    "answer": "I couldn't find relevant information.",
                    "confidence": "low",
                    "sources": [],
                }

            # Process tool calls
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results = []
            for block in assistant_content:
                if block.type == "tool_use":
                    result = self.execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

        # Max turns reached
        return {
            "answer": "I searched extensively but couldn't find a complete answer.",
            "confidence": "low",
            "sources": [],
        }

    def _parse_answer(self, text: str) -> dict:
        """Parse the agent's response into structured format."""
        # Try to extract structured components
        answer = text
        confidence = "medium"
        sources = []

        # Look for confidence indicators
        text_lower = text.lower()
        if any(w in text_lower for w in ["definitely", "clearly", "certainly", "found that"]):
            confidence = "high"
        elif any(w in text_lower for w in ["might", "possibly", "unclear", "not sure", "couldn't find"]):
            confidence = "low"

        # Extract source references (basic extraction)
        import re

        # Look for entity references like "John Smith" or IDs
        entity_refs = re.findall(r'"([^"]+)"', text)
        for ref in entity_refs[:5]:  # Limit sources
            sources.append({"type": "reference", "value": ref})

        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
        }
