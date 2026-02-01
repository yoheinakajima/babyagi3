"""
Extraction pipeline for the memory system.

Extracts entities, edges, and topics from events using LLM.
"""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime

from metrics import InstrumentedAnthropic, track_source
from .embeddings import get_embedding
from .models import (
    Event,
    ExtractionResult,
    ExtractedEdge,
    ExtractedEntity,
    ExtractedFact,
    ExtractedTopic,
)


# ═══════════════════════════════════════════════════════════
# EXTRACTION RETRY CONFIGURATION
# ═══════════════════════════════════════════════════════════


@dataclass
class ExtractionConfig:
    """Configuration for extraction with retry logic."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    retry_on_rate_limit: bool = True
    retry_on_api_error: bool = True


class ExtractionPipeline:
    """
    Extracts structured information from events.

    Uses LLM to identify entities, relationships, and topics.
    Includes retry logic with exponential backoff.
    """

    def __init__(self, store, config: ExtractionConfig | None = None):
        self.store = store
        self.config = config or ExtractionConfig()
        self._client = None

    @property
    def client(self):
        """Get instrumented Anthropic client for metrics tracking."""
        if self._client is None:
            self._client = InstrumentedAnthropic()
        return self._client

    async def extract(self, event: Event) -> ExtractionResult:
        """
        Extract entities, edges, and topics from an event.

        This is the main entry point for extraction.
        Includes retry logic with exponential backoff.
        """
        # Skip if already extracted
        if event.extraction_status == "complete":
            return ExtractionResult()

        # Check retry count
        retry_count = self.store.get_extraction_retry_count(event.id)
        if retry_count >= self.config.max_retries:
            # Mark as permanently failed
            self.store.update_event_extraction_status(event.id, "failed_permanent")
            return ExtractionResult()

        # Mark as processing
        self.store.update_event_extraction_status(event.id, "processing")

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                return await self._do_extract(event)

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if we should retry
                should_retry = False
                if self.config.retry_on_rate_limit and "rate" in error_str:
                    should_retry = True
                elif self.config.retry_on_api_error and (
                    "api" in error_str or "connection" in error_str or "timeout" in error_str
                ):
                    should_retry = True

                if should_retry and attempt < self.config.max_retries - 1:
                    # Increment retry count
                    self.store.increment_extraction_retry(event.id)

                    # Calculate delay with exponential backoff
                    delay = self.config.retry_delay_seconds * (
                        self.config.retry_backoff ** attempt
                    )
                    await asyncio.sleep(delay)
                    continue

                # Don't retry, mark as failed
                break

        # All retries exhausted
        self.store.update_event_extraction_status(event.id, "failed")
        self.store.increment_extraction_retry(event.id)
        raise last_error if last_error else Exception("Extraction failed")

    async def _do_extract(self, event: Event) -> ExtractionResult:
        """Perform the actual extraction (called by extract with retry)."""
        # Generate embedding if not present
        if not event.content_embedding:
            embedding = get_embedding(event.content)
            self.store.update_event_embedding(event.id, embedding)
            event.content_embedding = embedding

        # Get context for extraction
        context = await self._build_extraction_context(event)

        # Run LLM extraction
        result = await self._llm_extract(event, context)

        # Process extracted entities first (facts depend on them)
        for extracted in result.entities:
            await self._process_entity(extracted, event)

        # Process extracted facts
        for extracted in result.facts:
            await self._process_fact(extracted, event)

        # Process extracted edges (legacy, for compatibility)
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

    async def retry_failed(self, limit: int = 10) -> list[ExtractionResult]:
        """
        Retry extraction for failed events.

        Returns list of successful extraction results.
        """
        failed_events = self.store.get_failed_extraction_events(
            max_retries=self.config.max_retries, limit=limit
        )

        results = []
        for event in failed_events:
            try:
                # Reset status to pending before retry
                self.store.update_event_extraction_status(event.id, "pending")
                result = await self.extract(event)
                results.append(result)
            except Exception as e:
                print(f"Retry failed for event {event.id}: {e}")

        return results

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

2. FACTS - Discrete pieces of information as triplets (subject-predicate-object)
   - Extract EVERY factual statement as a separate fact
   - Facts can have entities or literal values as objects
   - Be comprehensive - dates, numbers, relationships, events, metrics
   - Use natural, complete sentences for fact_text

   Fact types:
   - relation: Relationship between entities (John works at Acme)
   - attribute: Property of an entity (John's age is 35)
   - event: Something that happened (Company was founded in 2020)
   - state: Current status (Project status is active)
   - metric: Numerical data (Revenue was $4.2M)

3. EDGES - Connections between entities (legacy, still extract for compatibility)

4. TOPICS - What themes/subjects does this relate to? (1-5 topics)

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
    "facts": [
        {
            "subject": "entity name (required)",
            "predicate": "verb or relationship",
            "object": "entity name OR literal value",
            "object_type": "entity|value|text",
            "fact_type": "relation|attribute|event|state|metric",
            "fact_text": "Full natural sentence expressing this fact",
            "confidence": 0.0-1.0,
            "valid_from": "ISO date if known, null otherwise",
            "valid_to": "ISO date if known, null if current",
            "mentioned_entities": ["other entities mentioned in context"]
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

        with track_source("extraction"):
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
                data = {"entities": [], "facts": [], "edges": [], "topics": []}
        except json.JSONDecodeError:
            data = {"entities": [], "facts": [], "edges": [], "topics": []}

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

        facts = []
        for f in data.get("facts", []):
            facts.append(
                ExtractedFact(
                    subject=f.get("subject", ""),
                    predicate=f.get("predicate", ""),
                    object=f.get("object", ""),
                    object_type=f.get("object_type", "value"),
                    fact_type=f.get("fact_type", "relation"),
                    fact_text=f.get("fact_text", ""),
                    confidence=f.get("confidence", 0.8),
                    valid_from=f.get("valid_from"),
                    valid_to=f.get("valid_to"),
                    mentioned_entities=f.get("mentioned_entities", []),
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
            facts=facts,
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
        # Determine type cluster from type_raw using LLM
        type_cluster = await self._cluster_entity_type_llm(extracted.type_raw)

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
        relation_type = await self._cluster_relation_type_llm(extracted.relation)

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

    async def _process_fact(self, extracted: ExtractedFact, event: Event):
        """Process an extracted fact - resolve entities and store."""
        if not extracted.subject or not extracted.predicate:
            return

        # Resolve subject entity (required)
        subject_entities = self.store.find_entities(query=extracted.subject, limit=1)
        if not subject_entities:
            # Auto-create minimal entity for subject
            subject_entity = self.store.create_entity(
                name=extracted.subject,
                type="concept",  # Default type
                type_raw="extracted entity",
                source_event_ids=[event.id],
            )
            subject_id = subject_entity.id
        else:
            subject_id = subject_entities[0].id

        # Resolve object entity if object_type is "entity"
        object_entity_id = None
        object_value = None

        if extracted.object_type == "entity" and extracted.object:
            object_entities = self.store.find_entities(query=extracted.object, limit=1)
            if object_entities:
                object_entity_id = object_entities[0].id
            else:
                # Auto-create minimal entity for object
                object_entity = self.store.create_entity(
                    name=extracted.object,
                    type="concept",
                    type_raw="extracted entity",
                    source_event_ids=[event.id],
                )
                object_entity_id = object_entity.id
        else:
            object_value = extracted.object

        # Resolve mentioned entities
        mentioned_entity_ids = []
        for name in extracted.mentioned_entities:
            entities = self.store.find_entities(query=name, limit=1)
            if entities:
                mentioned_entity_ids.append(entities[0].id)

        # Check for existing fact (deduplication)
        existing = self.store.find_fact(
            subject_entity_id=subject_id,
            predicate=extracted.predicate,
            object_entity_id=object_entity_id,
            object_value=object_value,
        )

        if existing:
            # Increment strength and add source event
            self.store.increment_fact_strength(existing.id, event.id)
            return

        # Generate embedding for fact text
        fact_embedding = get_embedding(extracted.fact_text) if extracted.fact_text else None

        # Cluster predicate type
        predicate_type = await self._cluster_relation_type_llm(extracted.predicate)

        # Create new fact
        self.store.create_fact(
            subject_entity_id=subject_id,
            predicate=extracted.predicate,
            fact_text=extracted.fact_text or f"{extracted.subject} {extracted.predicate} {extracted.object}",
            object_entity_id=object_entity_id,
            object_value=object_value,
            object_type=extracted.object_type,
            mentioned_entity_ids=mentioned_entity_ids,
            fact_type=extracted.fact_type,
            predicate_type=predicate_type,
            fact_embedding=fact_embedding,
            source_type="conversation",
            source_id=event.id,
            source_event_ids=[event.id],
            confidence=extracted.confidence,
            strength=0.5,
            valid_from=extracted.valid_from,
            valid_to=extracted.valid_to,
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

    # Type clustering constants
    ENTITY_TYPE_CLUSTERS = ["person", "org", "tool", "location", "concept"]
    RELATION_TYPE_CLUSTERS = ["professional", "financial", "social", "technical", "other"]

    # Cache for type clustering results to avoid repeated LLM calls
    _type_cache: dict[str, str] = {}
    _relation_cache: dict[str, str] = {}

    async def _cluster_entity_type_llm(self, type_raw: str) -> str:
        """Map raw entity type to cluster using LLM."""
        # Check cache first
        cache_key = type_raw.lower().strip()
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]

        # Try quick heuristics first for obvious cases
        quick_result = self._quick_entity_type_match(type_raw)
        if quick_result:
            self._type_cache[cache_key] = quick_result
            return quick_result

        # Use LLM for ambiguous cases
        prompt = f"""Classify this entity type into exactly one category.

Entity type: "{type_raw}"

Categories:
- person: Individual humans (investors, founders, engineers, friends, etc.)
- org: Organizations, companies, institutions, funds
- tool: Software, apps, platforms, libraries, APIs, services
- location: Cities, countries, regions, physical places
- concept: Ideas, topics, abstract things, everything else

Respond with only the category name, nothing else."""

        try:
            with track_source("extraction"):
                response = self.client.messages.create(
                    model="claude-haiku-3-5-20241022",
                    max_tokens=20,
                    messages=[{"role": "user", "content": prompt}],
                )
            result = response.content[0].text.strip().lower()

            # Validate result
            if result in self.ENTITY_TYPE_CLUSTERS:
                self._type_cache[cache_key] = result
                return result
        except Exception:
            pass

        # Fallback to concept
        self._type_cache[cache_key] = "concept"
        return "concept"

    async def _cluster_relation_type_llm(self, relation: str) -> str:
        """Map raw relation to cluster using LLM."""
        # Check cache first
        cache_key = relation.lower().strip()
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]

        # Try quick heuristics first
        quick_result = self._quick_relation_type_match(relation)
        if quick_result:
            self._relation_cache[cache_key] = quick_result
            return quick_result

        # Use LLM for ambiguous cases
        prompt = f"""Classify this relationship type into exactly one category.

Relationship: "{relation}"

Categories:
- professional: Work, employment, business partnerships, leadership
- financial: Investment, funding, purchases, ownership, transactions
- social: Friendships, family, personal connections, introductions
- technical: Software dependencies, integrations, uses, creates
- other: Everything else

Respond with only the category name, nothing else."""

        try:
            with track_source("extraction"):
                response = self.client.messages.create(
                    model="claude-haiku-3-5-20241022",
                    max_tokens=20,
                    messages=[{"role": "user", "content": prompt}],
                )
            result = response.content[0].text.strip().lower()

            # Validate result
            if result in self.RELATION_TYPE_CLUSTERS:
                self._relation_cache[cache_key] = result
                return result
        except Exception:
            pass

        # Fallback to other
        self._relation_cache[cache_key] = "other"
        return "other"

    def _quick_entity_type_match(self, type_raw: str) -> str | None:
        """Quick heuristic matching for obvious entity types."""
        type_lower = type_raw.lower()

        person_keywords = ["person", "human", "individual", "investor", "founder", "ceo",
                          "engineer", "developer", "designer", "manager", "employee",
                          "colleague", "friend", "contact", "user", "customer"]
        org_keywords = ["company", "organization", "org", "startup", "corporation",
                       "firm", "agency", "institution", "fund", "vc", "venture",
                       "bank", "university", "school", "government"]
        tool_keywords = ["tool", "software", "app", "application", "platform",
                        "library", "framework", "api", "service", "product", "website"]
        location_keywords = ["city", "country", "location", "place", "region",
                            "office", "building", "address", "state", "town"]

        for kw in person_keywords:
            if kw in type_lower:
                return "person"
        for kw in org_keywords:
            if kw in type_lower:
                return "org"
        for kw in tool_keywords:
            if kw in type_lower:
                return "tool"
        for kw in location_keywords:
            if kw in type_lower:
                return "location"

        return None  # Need LLM for this one

    def _quick_relation_type_match(self, relation: str) -> str | None:
        """Quick heuristic matching for obvious relation types."""
        relation_lower = relation.lower()

        professional_keywords = ["works", "employed", "hired", "manages", "reports",
                                "colleague", "team", "founded", "ceo", "leads",
                                "employee", "boss", "coworker", "partner"]
        financial_keywords = ["invest", "fund", "paid", "bought", "sold", "owns",
                             "acquired", "raised", "valued", "purchased", "financed"]
        social_keywords = ["knows", "friend", "met", "introduced", "connected",
                          "family", "married", "related", "sibling", "parent"]
        technical_keywords = ["uses", "built", "created", "developed", "integrates",
                             "depends", "implements", "extends", "imports"]

        for kw in professional_keywords:
            if kw in relation_lower:
                return "professional"
        for kw in financial_keywords:
            if kw in relation_lower:
                return "financial"
        for kw in social_keywords:
            if kw in relation_lower:
                return "social"
        for kw in technical_keywords:
            if kw in relation_lower:
                return "technical"

        return None  # Need LLM for this one

    def _cluster_entity_type(self, type_raw: str) -> str:
        """Synchronous wrapper for entity type clustering."""
        # For sync contexts, use heuristics only
        quick_result = self._quick_entity_type_match(type_raw)
        return quick_result or "concept"

    def _cluster_relation_type(self, relation: str) -> str:
        """Synchronous wrapper for relation type clustering."""
        # For sync contexts, use heuristics only
        quick_result = self._quick_relation_type_match(relation)
        return quick_result or "other"


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
