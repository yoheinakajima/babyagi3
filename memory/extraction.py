"""
Extraction pipeline for the memory system.

Extracts entities, edges, and topics from events using LLM.
"""

import json
from datetime import datetime

from .embeddings import get_embedding
from .models import (
    Event,
    ExtractionResult,
    ExtractedEdge,
    ExtractedEntity,
    ExtractedTopic,
)


class ExtractionPipeline:
    """
    Extracts structured information from events.

    Uses LLM to identify entities, relationships, and topics.
    """

    def __init__(self, store):
        self.store = store
        self._client = None

    @property
    def client(self):
        """Get Anthropic client."""
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    async def extract(self, event: Event) -> ExtractionResult:
        """
        Extract entities, edges, and topics from an event.

        This is the main entry point for extraction.
        """
        # Skip if already extracted
        if event.extraction_status == "complete":
            return ExtractionResult()

        # Mark as processing
        self.store.update_event_extraction_status(event.id, "processing")

        try:
            # Generate embedding if not present
            if not event.content_embedding:
                embedding = get_embedding(event.content)
                self.store.update_event_embedding(event.id, embedding)
                event.content_embedding = embedding

            # Get context for extraction
            context = await self._build_extraction_context(event)

            # Run LLM extraction
            result = await self._llm_extract(event, context)

            # Process extracted entities
            for extracted in result.entities:
                await self._process_entity(extracted, event)

            # Process extracted edges
            for extracted in result.edges:
                await self._process_edge(extracted, event)

            # Process extracted topics
            for extracted in result.topics:
                await self._process_topic(extracted, event)

            # Mark as complete
            self.store.update_event_extraction_status(
                event.id, "complete", datetime.now()
            )

            return result

        except Exception as e:
            # Mark as failed
            self.store.update_event_extraction_status(event.id, "failed")
            raise e

    async def _build_extraction_context(self, event: Event) -> dict:
        """Build context for extraction prompt."""
        context = {
            "recent_entities": [],
            "recent_topics": [],
            "agent_state": None,
        }

        # Get potentially relevant entities (by embedding similarity)
        if event.content_embedding:
            # Simple approach: get recent entities
            entities = self.store.find_entities(limit=10)
            context["recent_entities"] = [
                {"id": e.id, "name": e.name, "type": e.type, "aliases": e.aliases}
                for e in entities
            ]

        # Get recent topics
        topics = self.store.find_topics(limit=5)
        context["recent_topics"] = [
            {"id": t.id, "label": t.label, "keywords": t.keywords} for t in topics
        ]

        # Get agent state
        state = self.store.get_agent_state()
        context["agent_state"] = {
            "name": state.name,
            "current_topics": state.current_topics,
        }

        return context

    async def _llm_extract(self, event: Event, context: dict) -> ExtractionResult:
        """Run LLM extraction on an event."""
        system_prompt = """You are extracting structured information from an event for an AI agent's memory.

Extract:
1. ENTITIES - People, organizations, tools, concepts, places, projects
   - Be specific with types ("venture capitalist" not just "person")
   - Note if an entity matches one in EXISTING ENTITIES

2. RELATIONSHIPS - Connections between entities
   - Be specific ("invested in" not "is related to")
   - Include direction and whether it's current

3. TOPICS - What themes/subjects does this relate to? (1-5 topics)

Return valid JSON matching this schema:
{
    "entities": [
        {
            "name": "string",
            "type_raw": "string (specific type)",
            "aliases": ["array of alternative names"],
            "description": "brief description if known",
            "matched_entity_id": "id if matches existing entity, null otherwise",
            "match_confidence": 0.0-1.0,
            "importance": 0.0-1.0
        }
    ],
    "edges": [
        {
            "source": "entity name",
            "target": "entity name",
            "relation": "specific relationship",
            "is_current": true/false,
            "strength": 0.0-1.0
        }
    ],
    "topics": [
        {
            "label": "topic name (1-4 words)",
            "keywords": ["related", "keywords"],
            "relevance": 0.0-1.0
        }
    ],
    "notes": "any extraction notes"
}"""

        # Build user message
        user_message = f"""EXISTING ENTITIES:
{json.dumps(context['recent_entities'], indent=2)}

EXISTING TOPICS:
{json.dumps(context['recent_topics'], indent=2)}

EVENT TO EXTRACT FROM:
Type: {event.event_type}
Channel: {event.channel or 'N/A'} ({event.direction})
Timestamp: {event.timestamp.isoformat() if event.timestamp else 'N/A'}

Content:
{event.content}

Extract entities, relationships, and topics from this event."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        # Parse response
        response_text = response.content[0].text

        # Extract JSON from response
        try:
            # Try to find JSON in the response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response_text[start:end]
                data = json.loads(json_str)
            else:
                data = {"entities": [], "edges": [], "topics": []}
        except json.JSONDecodeError:
            data = {"entities": [], "edges": [], "topics": []}

        # Convert to ExtractionResult
        entities = []
        for e in data.get("entities", []):
            entities.append(
                ExtractedEntity(
                    name=e.get("name", ""),
                    type_raw=e.get("type_raw", "unknown"),
                    aliases=e.get("aliases", []),
                    description=e.get("description"),
                    matched_entity_id=e.get("matched_entity_id"),
                    match_confidence=e.get("match_confidence", 0.0),
                    importance=e.get("importance", 0.5),
                )
            )

        edges = []
        for e in data.get("edges", []):
            edges.append(
                ExtractedEdge(
                    source=e.get("source", ""),
                    target=e.get("target", ""),
                    relation=e.get("relation", "related to"),
                    is_current=e.get("is_current", True),
                    strength=e.get("strength", 0.5),
                )
            )

        topics = []
        for t in data.get("topics", []):
            topics.append(
                ExtractedTopic(
                    label=t.get("label", ""),
                    keywords=t.get("keywords", []),
                    relevance=t.get("relevance", 1.0),
                )
            )

        return ExtractionResult(
            entities=entities,
            edges=edges,
            topics=topics,
            notes=data.get("notes"),
        )

    async def _process_entity(self, extracted: ExtractedEntity, event: Event):
        """Process an extracted entity - resolve and store."""
        if not extracted.name:
            return

        entity = None

        # Check if matched to existing entity
        if extracted.matched_entity_id and extracted.match_confidence > 0.7:
            entity = self.store.get_entity(extracted.matched_entity_id)
            if entity:
                # Update existing entity
                entity.event_count += 1
                entity.last_seen = event.timestamp
                if event.id not in entity.source_event_ids:
                    entity.source_event_ids.append(event.id)
                # Add any new aliases
                for alias in extracted.aliases:
                    if alias not in entity.aliases:
                        entity.aliases.append(alias)
                self.store.update_entity(entity)
                return

        # Try to find by name
        existing = self.store.find_entities(query=extracted.name, limit=1)
        if existing and existing[0].name.lower() == extracted.name.lower():
            entity = existing[0]
            entity.event_count += 1
            entity.last_seen = event.timestamp
            if event.id not in entity.source_event_ids:
                entity.source_event_ids.append(event.id)
            for alias in extracted.aliases:
                if alias not in entity.aliases:
                    entity.aliases.append(alias)
            self.store.update_entity(entity)
            return

        # Create new entity
        # Determine type cluster from type_raw
        type_cluster = self._cluster_entity_type(extracted.type_raw)

        # Generate embedding
        embed_text = f"{extracted.name} {extracted.type_raw}"
        if extracted.description:
            embed_text += f" {extracted.description}"
        embedding = get_embedding(embed_text)

        entity = self.store.create_entity(
            name=extracted.name,
            type=type_cluster,
            type_raw=extracted.type_raw,
            aliases=extracted.aliases,
            description=extracted.description,
            name_embedding=embedding,
            source_event_ids=[event.id],
        )

    async def _process_edge(self, extracted: ExtractedEdge, event: Event):
        """Process an extracted edge - resolve entities and store."""
        if not extracted.source or not extracted.target:
            return

        # Resolve source entity
        source_entities = self.store.find_entities(query=extracted.source, limit=1)
        if not source_entities:
            return
        source = source_entities[0]

        # Resolve target entity
        target_entities = self.store.find_entities(query=extracted.target, limit=1)
        if not target_entities:
            return
        target = target_entities[0]

        # Check for existing edge
        existing = self.store.find_edge(source.id, target.id, extracted.relation)
        if existing:
            # Update existing edge
            if event.id not in existing.source_event_ids:
                existing.source_event_ids.append(event.id)
            existing.strength = min(1.0, existing.strength + 0.1)
            self.store.update_edge(existing)
            return

        # Create new edge
        embedding = get_embedding(extracted.relation)
        relation_type = self._cluster_relation_type(extracted.relation)

        self.store.create_edge(
            source_entity_id=source.id,
            target_entity_id=target.id,
            relation=extracted.relation,
            relation_type=relation_type,
            relation_embedding=embedding,
            is_current=extracted.is_current,
            strength=extracted.strength,
            source_event_ids=[event.id],
        )

    async def _process_topic(self, extracted: ExtractedTopic, event: Event):
        """Process an extracted topic - find or create, then link."""
        if not extracted.label:
            return

        # Try to find existing topic
        existing = self.store.find_topic_by_label(extracted.label)
        if existing:
            # Link event to existing topic
            self.store.link_event_topic(event.id, existing.id, extracted.relevance)
            return

        # Create new topic
        embedding = get_embedding(extracted.label)

        topic = self.store.create_topic(
            label=extracted.label,
            keywords=extracted.keywords,
            embedding=embedding,
        )

        # Link event to topic
        self.store.link_event_topic(event.id, topic.id, extracted.relevance)

    def _cluster_entity_type(self, type_raw: str) -> str:
        """Map raw entity type to cluster."""
        type_lower = type_raw.lower()

        # Person types
        if any(
            w in type_lower
            for w in [
                "person",
                "investor",
                "founder",
                "ceo",
                "engineer",
                "developer",
                "designer",
                "manager",
                "analyst",
                "consultant",
                "executive",
                "employee",
                "colleague",
                "friend",
                "contact",
            ]
        ):
            return "person"

        # Organization types
        if any(
            w in type_lower
            for w in [
                "company",
                "organization",
                "startup",
                "corporation",
                "firm",
                "agency",
                "institution",
                "fund",
                "vc",
                "venture",
                "bank",
                "university",
            ]
        ):
            return "org"

        # Tool types
        if any(
            w in type_lower
            for w in [
                "tool",
                "software",
                "app",
                "platform",
                "library",
                "framework",
                "api",
                "service",
                "product",
            ]
        ):
            return "tool"

        # Location types
        if any(
            w in type_lower
            for w in ["city", "country", "location", "place", "region", "office"]
        ):
            return "location"

        # Default to concept
        return "concept"

    def _cluster_relation_type(self, relation: str) -> str:
        """Map raw relation to cluster."""
        relation_lower = relation.lower()

        # Professional relationships
        if any(
            w in relation_lower
            for w in [
                "works",
                "employed",
                "hired",
                "manages",
                "reports",
                "colleague",
                "team",
                "founded",
                "ceo",
                "leads",
            ]
        ):
            return "professional"

        # Financial relationships
        if any(
            w in relation_lower
            for w in [
                "invest",
                "fund",
                "paid",
                "bought",
                "sold",
                "owns",
                "acquired",
                "raised",
                "valued",
            ]
        ):
            return "financial"

        # Social relationships
        if any(
            w in relation_lower
            for w in [
                "knows",
                "friend",
                "met",
                "introduced",
                "connected",
                "family",
                "married",
            ]
        ):
            return "social"

        # Technical relationships
        if any(
            w in relation_lower
            for w in ["uses", "built", "created", "developed", "integrates", "depends"]
        ):
            return "technical"

        # Default
        return "other"


async def extract_batch(
    store, events: list[Event], batch_size: int = 10
) -> list[ExtractionResult]:
    """
    Extract from multiple events in batch.

    More efficient than extracting one at a time.
    """
    pipeline = ExtractionPipeline(store)
    results = []

    for event in events:
        try:
            result = await pipeline.extract(event)
            results.append(result)
        except Exception as e:
            print(f"Extraction failed for event {event.id}: {e}")
            results.append(ExtractionResult())

    return results
