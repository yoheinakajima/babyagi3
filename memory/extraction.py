"""
Extraction Pipeline - LLM-based extraction of entities, edges, and topics from events.

This runs asynchronously after events are logged. It:
1. Extracts entities (people, orgs, topics, etc.) with free-form types
2. Extracts edges (relationships) between entities
3. Tags events with extracted topics
4. Updates the graph
5. Marks affected summaries as stale

The extraction uses free-form types that get clustered automatically.
"""

import json
from typing import Callable, Awaitable
from dataclasses import dataclass
from .models import Event, Entity, Edge, SliceKey
from .event_log import EventLog
from .graph import Graph
from .summaries import SummaryTree


@dataclass
class ExtractionResult:
    """Result from extracting an event."""

    event_id: str
    entities: list[Entity]
    edges: list[Edge]
    topics: list[str]
    tags_added: dict[str, str]


# Extraction prompt template
EXTRACTION_PROMPT = """Analyze this event and extract structured information.

Event type: {event_type}
Event content:
{event_content}

Extract:
1. ENTITIES: People, organizations, topics, concepts, products mentioned
   - For each: name, type (be specific, e.g., "venture capitalist" not just "person"), key attributes
2. EDGES: Relationships between entities
   - For each: source entity, target entity, relationship type (be specific)
3. TOPICS: Main topics/themes (1-3 words each)

Respond in JSON:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{...}}}}
  ],
  "edges": [
    {{"source": "...", "target": "...", "type": "..."}}
  ],
  "topics": ["topic1", "topic2"]
}}

Only include clearly mentioned entities and relationships. Be specific with types.
If nothing notable to extract, return empty arrays."""


class ExtractionPipeline:
    """
    Extracts entities, edges, and topics from events.

    Runs asynchronously and updates the graph and summary tree.
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
        self._extract_fn: Callable[[str], Awaitable[str]] | None = None

        # Track which events have been processed
        self._processed_events: set[str] = set()

    def set_extract_fn(self, fn: Callable[[str], Awaitable[str]]):
        """
        Set the function used to call the LLM for extraction.

        fn takes a prompt string and returns the LLM response.
        """
        self._extract_fn = fn

    async def process_event(self, event: Event) -> ExtractionResult | None:
        """
        Process a single event and extract information.

        Returns None if already processed or no extraction function set.
        """
        if event.id in self._processed_events:
            return None

        if not self._extract_fn:
            return None

        # Skip certain event types
        if event.type in ("extraction", "summary_refresh"):
            return None

        # Build prompt
        content_str = (
            json.dumps(event.content, indent=2)
            if isinstance(event.content, dict)
            else str(event.content)
        )

        prompt = EXTRACTION_PROMPT.format(
            event_type=event.type,
            event_content=content_str[:2000],  # Limit content size
        )

        try:
            response = await self._extract_fn(prompt)
            result = self._parse_extraction(event, response)

            if result:
                self._apply_extraction(result)
                self._processed_events.add(event.id)

            return result

        except Exception:
            # Log error but don't fail
            return None

    def _parse_extraction(self, event: Event, response: str) -> ExtractionResult | None:
        """Parse LLM response into extraction result."""
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1 or end == 0:
                return None

            data = json.loads(response[start:end])

            entities = []
            for e in data.get("entities", []):
                entity = Entity.create(
                    type=e.get("type", "unknown"),
                    name=e.get("name", ""),
                    attributes=e.get("attributes", {}),
                    source_events=[event.id],
                )
                entities.append(entity)

            edges = []
            for ed in data.get("edges", []):
                # We'll resolve entity IDs after adding entities
                edge = Edge.create(
                    source_id=ed.get("source", ""),  # Will be resolved
                    target_id=ed.get("target", ""),  # Will be resolved
                    type=ed.get("type", "related_to"),
                    source_events=[event.id],
                )
                edges.append(edge)

            topics = data.get("topics", [])

            return ExtractionResult(
                event_id=event.id,
                entities=entities,
                edges=edges,
                topics=topics,
                tags_added={},
            )

        except (json.JSONDecodeError, KeyError):
            return None

    def _apply_extraction(self, result: ExtractionResult):
        """Apply extraction result to graph and summary tree."""
        # Add entities to graph
        name_to_id = {}
        for entity in result.entities:
            added = self.graph.add_entity(entity)
            name_to_id[entity.name.lower()] = added.id

            # Mark person summary as stale
            person_slice = SliceKey({"person": entity.name.lower()})
            self.summary_tree.mark_stale(person_slice)

        # Add edges (resolve names to IDs)
        for edge in result.edges:
            source_id = name_to_id.get(edge.source_id.lower())
            target_id = name_to_id.get(edge.target_id.lower())

            # Try to find existing entities
            if not source_id:
                source_entity = self.graph.find_entity(edge.source_id)
                source_id = source_entity.id if source_entity else None
            if not target_id:
                target_entity = self.graph.find_entity(edge.target_id)
                target_id = target_entity.id if target_entity else None

            if source_id and target_id:
                edge.source_id = source_id
                edge.target_id = target_id
                self.graph.add_edge(edge)

        # Mark topic summaries as stale
        for topic in result.topics:
            topic_slice = SliceKey({"topic": topic.lower()})
            self.summary_tree.mark_stale(topic_slice)

    async def process_pending(self, limit: int = 10) -> int:
        """
        Process pending events that haven't been extracted yet.

        Returns number of events processed.
        """
        processed = 0

        for event in self.event_log.iter_events(reverse=True):
            if event.id in self._processed_events:
                continue

            result = await self.process_event(event)
            if result:
                processed += 1

            if processed >= limit:
                break

        return processed

    def on_event(self, event: Event):
        """
        Callback for event log subscription.

        This queues the event for extraction. Actual extraction
        happens asynchronously via process_pending().
        """
        # Could add to a queue here for immediate processing
        pass


# Type clustering prompt
CLUSTERING_PROMPT = """Given these entity/edge types extracted from events, cluster them into canonical types.

Raw types:
{raw_types}

For each raw type, assign it to one of these canonical types or create a new one:
- person
- organization
- topic
- product
- location
- event
- concept

Respond in JSON:
{{
  "clusters": {{
    "raw_type_1": "canonical_type",
    "raw_type_2": "canonical_type"
  }}
}}"""


class TypeClusterer:
    """
    Clusters free-form entity and edge types into canonical types.

    Runs periodically to keep the type system organized.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._cluster_fn: Callable[[str], Awaitable[str]] | None = None

    def set_cluster_fn(self, fn: Callable[[str], Awaitable[str]]):
        """Set the function used to call the LLM for clustering."""
        self._cluster_fn = fn

    async def cluster_entity_types(self) -> dict[str, str]:
        """
        Cluster entity types and update the graph.

        Returns the clustering mapping.
        """
        if not self._cluster_fn:
            return {}

        # Get unique types
        raw_types = self.graph.get_entity_types()
        if len(raw_types) < 3:
            return {}  # Not enough to cluster

        # Build prompt
        prompt = CLUSTERING_PROMPT.format(raw_types=", ".join(raw_types))

        try:
            response = await self._cluster_fn(prompt)

            # Parse response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start == -1:
                return {}

            data = json.loads(response[start:end])
            clusters = data.get("clusters", {})

            # Apply to graph
            for raw_type, canonical in clusters.items():
                self.graph.set_type_cluster(raw_type, canonical)

            return clusters

        except Exception:
            return {}
