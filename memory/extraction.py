"""
Extraction pipeline for the memory system.

Extracts entities, edges, and topics from events using LLM.

Optimized for high-volume scenarios:
- Heuristic event triage (skip low-value events without LLM calls)
- Batch extraction (combine multiple events into one LLM call)
- Pure-heuristic type clustering (no LLM calls for entity/relation types)
- Content truncation (prevent output token overflow)
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime

from metrics import LiteLLMAnthropicAdapter, track_source, get_model_for_use_case
import logging

logger = logging.getLogger(__name__)

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

# Maximum characters of event content to send to the LLM for extraction.
# Longer content is truncated to reduce input tokens and prevent output overflow.
MAX_EXTRACTION_CONTENT_CHARS = 2000

# Event types that are not worth extracting (internal plumbing).
SKIP_EVENT_TYPES = {"tool_call", "tool_error", "tool_disabled", "thread_repaired"}

# Minimum content length for extraction to be worthwhile.
MIN_CONTENT_LENGTH = 20


@dataclass
class ExtractionConfig:
    """Configuration for extraction with retry logic."""

    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    retry_on_rate_limit: bool = True
    retry_on_api_error: bool = True
    max_tokens: int = 4096  # Output token limit for extraction calls
    max_batch_size: int = 5  # Max events to combine in a single batch extraction


class ExtractionPipeline:
    """
    Extracts structured information from events.

    Uses LLM to identify entities, relationships, and topics.
    Includes retry logic with exponential backoff.

    Optimized for high volume:
    - should_extract() triages events without LLM calls
    - extract_batch_combined() processes multiple events in one LLM call
    - Type clustering uses pure heuristics (no LLM calls)
    - Content is truncated to prevent output overflow
    """

    def __init__(self, store, config: ExtractionConfig | None = None):
        self.store = store
        self.config = config or ExtractionConfig()
        self._client = None

    @property
    def client(self):
        """Get instrumented LLM client for metrics tracking (supports multiple providers)."""
        if self._client is None:
            self._client = LiteLLMAnthropicAdapter()
        return self._client

    @property
    def model(self) -> str:
        """Get the configured model for memory operations."""
        return get_model_for_use_case("memory")

    @property
    def fast_model(self) -> str:
        """Get the configured fast model for quick classification tasks."""
        return get_model_for_use_case("fast")

    # ═══════════════════════════════════════════════════════════
    # EVENT TRIAGE (no LLM calls — pure heuristics)
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def should_extract(event: Event) -> bool:
        """Decide if an event is worth extracting.

        This is a fast, heuristic-only check. No LLM calls.
        Returns False for low-value events (internal plumbing, duplicate info, etc.)
        """
        # Already processed
        if event.extraction_status in ("complete", "skipped", "failed_permanent"):
            return False

        # Skip event types that are internal plumbing
        if event.event_type in SKIP_EVENT_TYPES:
            return False

        # Skip empty or trivially short content
        if not event.content or len(event.content.strip()) < MIN_CONTENT_LENGTH:
            return False

        # Skip tool results that are just status confirmations
        if event.event_type == "tool_result":
            content_lower = event.content.lower()
            # Skip "Tool result: X\nResult: {'status': 'ok'}" type messages
            if any(
                marker in content_lower
                for marker in [
                    "result: {'status': 'ok'}",
                    "result: {'success': true}",
                    "result: ok",
                    "result: done",
                    "result: true",
                ]
            ):
                return False

        return True

    @staticmethod
    def _truncate_content(content: str, max_chars: int = MAX_EXTRACTION_CONTENT_CHARS) -> str:
        """Truncate content for extraction, preserving useful structure.

        Long tool outputs and web scrapes are the main culprits for output
        token overflow. Truncating input content prevents the LLM from trying
        to extract from too much data, which causes truncated JSON output.
        """
        if len(content) <= max_chars:
            return content

        # Keep the beginning and indicate truncation
        return content[:max_chars] + "\n\n[... content truncated for extraction ...]"

    async def extract(self, event: Event) -> ExtractionResult:
        """
        Extract entities, edges, and topics from an event.

        This is the main entry point for extraction.
        Includes retry logic with exponential backoff.
        Performs heuristic triage before making any LLM calls.
        """
        # Heuristic triage — skip low-value events without any LLM calls
        if not self.should_extract(event):
            if event.extraction_status not in ("complete", "skipped", "failed_permanent"):
                self.store.update_event_extraction_status(event.id, "skipped")
            return ExtractionResult()

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
                logger.warning("Retry failed for event %s: %s", event.id, e)

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
        """Run LLM extraction on an event.

        Content is truncated to prevent output token overflow.
        Uses the memory model (configured to be fast/cheap for high-volume extraction).
        """
        system_prompt = """Extract structured information from an event for an AI agent's memory. Return valid JSON only.

Extract:
1. ENTITIES - People, organizations, tools, concepts, places, projects. Note matches to EXISTING ENTITIES.
2. FACTS - Discrete triplets (subject-predicate-object). Types: relation, attribute, event, state, metric.
3. TOPICS - Themes/subjects (1-3 topics max).

Return JSON:
{
    "entities": [{"name": "str", "type_raw": "str", "aliases": [], "description": "str or null", "matched_entity_id": "id or null", "match_confidence": 0.0, "importance": 0.5}],
    "facts": [{"subject": "str", "predicate": "str", "object": "str", "object_type": "entity|value|text", "fact_type": "relation|attribute|event|state|metric", "fact_text": "full sentence", "confidence": 0.8, "valid_from": null, "valid_to": null, "mentioned_entities": []}],
    "topics": [{"label": "1-4 words", "keywords": [], "relevance": 1.0}]
}"""

        # Truncate content to prevent input/output overflow
        truncated_content = self._truncate_content(event.content)

        # Build user message (compact format to reduce input tokens)
        user_message = f"""EXISTING ENTITIES:
{json.dumps(context['recent_entities'], indent=None)}

EXISTING TOPICS:
{json.dumps(context['recent_topics'], indent=None)}

EVENT ({event.event_type}, {event.channel or 'internal'}, {event.direction}):
{truncated_content}"""

        with track_source("extraction"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
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
                logger.debug("No JSON found in extraction response")
                data = {"entities": [], "facts": [], "topics": []}
        except json.JSONDecodeError:
            logger.debug("JSON parse error in extraction response")
            data = {"entities": [], "facts": [], "topics": []}

        # Parse into ExtractionResult using shared parser
        return self._parse_extraction_data(data)

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
        """Map raw entity type to cluster using heuristics only.

        Previously used LLM for ambiguous cases, but the expanded heuristic list
        covers 95%+ of real-world types. Eliminating LLM calls here removes ~1
        API call per new entity, which is critical at high extraction volumes.
        """
        cache_key = type_raw.lower().strip()
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]

        result = self._quick_entity_type_match(type_raw) or "concept"
        self._type_cache[cache_key] = result
        return result

    async def _cluster_relation_type_llm(self, relation: str) -> str:
        """Map raw relation to cluster using heuristics only.

        Previously used LLM for ambiguous cases, but the expanded heuristic list
        covers 95%+ of real-world relations. Eliminating LLM calls here removes ~1
        API call per new fact/edge, which is critical at high extraction volumes.
        """
        cache_key = relation.lower().strip()
        if cache_key in self._relation_cache:
            return self._relation_cache[cache_key]

        result = self._quick_relation_type_match(relation) or "other"
        self._relation_cache[cache_key] = result
        return result

    @staticmethod
    def _word_match(text_lower: str, keywords: list[str]) -> bool:
        """Check if any keyword matches in the text.

        Multi-word keywords (e.g. 'venture capitalist') and keywords >= 4 chars
        use substring matching (catches verb stems like 'invest' in 'invested').
        Short keywords (<= 3 chars like 'ide', 'os', 'db', 'vc') use exact
        word matching to avoid false positives ('ide' in 'idea', 'os' in 'loss').
        """
        words = set(text_lower.split())
        for kw in keywords:
            if " " in kw or len(kw) >= 4:
                # Multi-word or long keyword: substring match
                if kw in text_lower:
                    return True
            else:
                # Short keyword: exact word match
                if kw in words:
                    return True
        return False

    def _quick_entity_type_match(self, type_raw: str) -> str | None:
        """Heuristic matching for entity types. Covers 95%+ of real-world types."""
        type_lower = type_raw.lower()

        # Person keywords (comprehensive)
        person_keywords = [
            "person", "human", "individual", "investor", "founder", "ceo", "cto", "cfo",
            "coo", "vp", "director", "president", "chairman", "engineer", "developer",
            "designer", "manager", "employee", "colleague", "friend", "contact", "user",
            "customer", "client", "advisor", "mentor", "consultant", "analyst", "researcher",
            "scientist", "professor", "teacher", "doctor", "lawyer", "accountant", "author",
            "writer", "journalist", "artist", "musician", "actor", "athlete", "coach",
            "partner", "spouse", "parent", "child", "sibling", "relative", "boss",
            "cofounder", "co-founder", "entrepreneur", "executive", "principal", "agent",
            "representative", "delegate", "ambassador", "minister", "senator", "mayor",
            "governor", "candidate", "volunteer", "intern", "associate", "fellow",
            "contributor", "speaker", "panelist", "moderator", "host", "guest",
            "attendee", "participant", "member", "subscriber", "follower", "fan",
            "recruiter", "headhunter", "stakeholder",
            "venture capitalist", "angel investor", "limited partner",
            "general partner", "board member",
        ]

        # Organization keywords (comprehensive)
        org_keywords = [
            "company", "organization", "org", "startup", "corporation", "corp",
            "firm", "agency", "institution", "fund", "vc", "venture", "capital",
            "bank", "university", "school", "college", "government", "gov",
            "ministry", "department", "bureau", "commission", "council",
            "foundation", "nonprofit", "ngo", "charity", "trust", "association",
            "consortium", "alliance", "federation", "union", "cooperative",
            "partnership", "llc", "inc", "ltd", "gmbh", "group", "holding",
            "conglomerate", "syndicate", "network", "marketplace", "exchange",
            "accelerator", "incubator", "lab", "studio", "collective",
            "committee", "board", "team", "division", "subsidiary",
            "brand", "publisher", "media", "newspaper", "magazine", "broadcaster",
            "airline", "carrier", "manufacturer", "supplier", "retailer",
            "hospital", "clinic", "pharmacy", "insurer", "church", "temple",
            "mosque", "military", "navy", "army", "police", "embassy",
        ]

        # Tool/software keywords (comprehensive)
        tool_keywords = [
            "tool", "software", "app", "application", "platform", "library",
            "framework", "api", "service", "product", "website",
            "extension", "plugin", "addon",
            "package", "module", "sdk", "cli", "gui", "ide", "editor",
            "database", "db", "server", "cloud", "saas", "paas", "iaas",
            "bot", "assistant", "model", "algorithm", "protocol", "standard",
            "language", "runtime", "compiler", "interpreter", "browser",
            "kernel", "driver", "firmware",
            "integration", "connector", "middleware", "gateway", "proxy",
            "dashboard", "portal", "console", "interface", "system",
            "engine", "pipeline", "workflow", "automation", "script",
            "spreadsheet", "document", "file", "format", "specification",
            "technology", "tech", "stack", "infrastructure", "resource",
            "crm", "erp", "cms", "lms",
            "web app", "mobile app", "desktop app",
            "operating system", "email service",
        ]

        # Location keywords (comprehensive)
        location_keywords = [
            "city", "country", "location", "place", "region", "office",
            "building", "address", "state", "town", "village", "district",
            "county", "province", "territory", "neighborhood", "area",
            "zone", "continent", "island", "peninsula", "mountain", "valley",
            "river", "lake", "ocean", "sea", "coast", "beach", "port",
            "airport", "station", "terminal", "campus", "park", "garden",
            "street", "avenue", "boulevard", "highway", "road", "bridge",
            "headquarters", "hq", "hub", "center", "centre", "venue",
            "arena", "stadium", "theater", "theatre", "museum", "gallery",
            "market", "mall", "plaza", "square", "warehouse", "factory",
        ]

        if self._word_match(type_lower, person_keywords):
            return "person"
        if self._word_match(type_lower, org_keywords):
            return "org"
        if self._word_match(type_lower, tool_keywords):
            return "tool"
        if self._word_match(type_lower, location_keywords):
            return "location"

        return None  # Falls back to "concept"

    def _quick_relation_type_match(self, relation: str) -> str | None:
        """Heuristic matching for relation types. Covers 95%+ of real-world relations."""
        relation_lower = relation.lower()

        professional_keywords = [
            "works", "employed", "hired", "manages", "reports", "colleague",
            "team", "founded", "ceo", "leads", "employee", "boss", "coworker",
            "partner", "advises", "consults", "mentors", "coaches", "trains",
            "supervises", "directs", "heads", "chairs", "serves", "appointed",
            "promoted", "resigned", "fired", "quit", "retired", "joined",
            "recruited", "onboarded", "collaborated", "co-founded", "cofounded",
            "operates", "runs", "oversees", "coordinates", "administers",
            "represents", "advocates", "negotiates", "delegates", "assigned",
            "affiliated", "associated", "contracted", "freelanced", "interned",
            "employed by", "works at", "works for", "reports to", "member of",
        ]

        financial_keywords = [
            "invest", "fund", "paid", "bought", "sold", "owns", "acquired",
            "raised", "valued", "purchased", "financed", "backed", "sponsored",
            "donated", "granted", "loaned", "borrowed", "owed", "charged",
            "billed", "invoiced", "refunded", "compensated", "earned",
            "revenue", "profit", "loss", "dividend", "equity", "stake",
            "share", "stock", "option", "warrant", "bond", "debt",
            "capital", "budget", "cost", "price", "fee", "rate",
            "payment", "transaction", "transfer", "deposit", "withdrawal",
            "subscription", "licensing", "royalty", "commission",
        ]

        social_keywords = [
            "knows", "friend", "met", "introduced", "connected", "family",
            "married", "related", "sibling", "parent", "child", "dating",
            "engaged", "divorced", "neighbor", "roommate", "classmate",
            "alumni", "schoolmate", "childhood", "visited", "traveled", "attended",
            "invited", "hosted", "celebrated", "recommended", "referred",
            "endorsed", "vouched", "trusted", "befriended",
            "grew up", "born in", "lives in", "moved to", "mentored by",
        ]

        technical_keywords = [
            "uses", "built", "created", "developed", "integrates", "depends",
            "implements", "extends", "imports", "exports", "configures",
            "deploys", "hosts", "compiled", "tested", "debugged", "optimized",
            "migrated", "upgraded", "patched", "forked", "cloned",
            "installed", "uninstalled", "configured", "customized",
            "automated", "scripted", "coded", "programmed", "designed",
            "architected", "modeled", "prototyped", "launched", "shipped",
            "released", "published", "licensed",
            "compatible", "incompatible", "supports", "requires",
            "generates", "processes", "analyzes", "transforms", "converts",
            "stores", "caches", "indexes", "queries", "fetches", "syncs",
            "runs on", "powered by", "built with", "written in",
            "open sourced",
        ]

        if self._word_match(relation_lower, professional_keywords):
            return "professional"
        if self._word_match(relation_lower, financial_keywords):
            return "financial"
        if self._word_match(relation_lower, social_keywords):
            return "social"
        if self._word_match(relation_lower, technical_keywords):
            return "technical"

        return None  # Falls back to "other"

    def _cluster_entity_type(self, type_raw: str) -> str:
        """Synchronous wrapper for entity type clustering."""
        return self._quick_entity_type_match(type_raw) or "concept"

    def _cluster_relation_type(self, relation: str) -> str:
        """Synchronous wrapper for relation type clustering."""
        return self._quick_relation_type_match(relation) or "other"


    # ═══════════════════════════════════════════════════════════
    # BATCH EXTRACTION (combine multiple events into one LLM call)
    # ═══════════════════════════════════════════════════════════

    async def extract_batch_combined(self, events: list[Event]) -> list[ExtractionResult]:
        """Extract from multiple events in a single LLM call.

        Instead of making one LLM call per event (~18s each with Sonnet),
        this combines up to max_batch_size events into one prompt.
        At high volume, this reduces LLM calls by 3-5x.

        Events that fail triage are skipped without any LLM calls.
        """
        # Triage: filter to extractable events only
        extractable = []
        skipped_results = {}
        for event in events:
            if self.should_extract(event):
                extractable.append(event)
            else:
                self.store.update_event_extraction_status(event.id, "skipped")
                skipped_results[event.id] = ExtractionResult()

        if not extractable:
            return [skipped_results.get(e.id, ExtractionResult()) for e in events]

        # If only 1 event, use standard extraction
        if len(extractable) == 1:
            try:
                result = await self.extract(extractable[0])
                return [
                    skipped_results.get(e.id, result if e.id == extractable[0].id else ExtractionResult())
                    for e in events
                ]
            except Exception as exc:
                logger.warning("Single extraction failed for event %s: %s", extractable[0].id, exc)
                return [ExtractionResult() for _ in events]

        # Build combined prompt
        context = await self._build_extraction_context(extractable[0])

        system_prompt = """Extract structured information from MULTIPLE events for an AI agent's memory. Return valid JSON only.

For each event, extract entities, facts, and topics. Combine and deduplicate across events.

Return JSON:
{
    "entities": [{"name": "str", "type_raw": "str", "aliases": [], "description": "str or null", "importance": 0.5}],
    "facts": [{"subject": "str", "predicate": "str", "object": "str", "object_type": "entity|value|text", "fact_type": "relation|attribute|event|state|metric", "fact_text": "full sentence", "confidence": 0.8}],
    "topics": [{"label": "1-4 words", "keywords": [], "relevance": 1.0}]
}"""

        # Build multi-event user message
        event_sections = []
        for i, event in enumerate(extractable[:self.config.max_batch_size], 1):
            truncated = self._truncate_content(event.content)
            event_sections.append(
                f"--- EVENT {i} ({event.event_type}, {event.channel or 'internal'}) ---\n{truncated}"
            )

        user_message = f"""EXISTING ENTITIES:
{json.dumps(context['recent_entities'], indent=None)}

{chr(10).join(event_sections)}

Extract and deduplicate entities, facts, and topics from ALL events above."""

        try:
            # Mark all as processing
            for event in extractable[:self.config.max_batch_size]:
                self.store.update_event_extraction_status(event.id, "processing")

            with track_source("extraction"):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.config.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}],
                )

            # Parse response
            response_text = response.content[0].text
            try:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start != -1 and end > start:
                    data = json.loads(response_text[start:end])
                else:
                    data = {"entities": [], "facts": [], "topics": []}
            except json.JSONDecodeError:
                data = {"entities": [], "facts": [], "topics": []}

            # Build a single ExtractionResult from the combined data
            result = self._parse_extraction_data(data)

            # Process the combined result against all source events
            first_event = extractable[0]
            for extracted in result.entities:
                await self._process_entity(extracted, first_event)
            for extracted in result.facts:
                await self._process_fact(extracted, first_event)
            for extracted in result.edges:
                await self._process_edge(extracted, first_event)
            for extracted in result.topics:
                await self._process_topic(extracted, first_event)

            # Mark all batch events as complete
            for event in extractable[:self.config.max_batch_size]:
                self.store.update_event_extraction_status(event.id, "complete", datetime.now())

            # Return results aligned with input order
            return [
                skipped_results.get(e.id, result if e in extractable[:self.config.max_batch_size] else ExtractionResult())
                for e in events
            ]

        except Exception as exc:
            logger.warning("Batch extraction failed: %s", exc)
            # Mark as failed so they can be retried individually
            for event in extractable[:self.config.max_batch_size]:
                self.store.update_event_extraction_status(event.id, "failed")
                self.store.increment_extraction_retry(event.id)
            return [ExtractionResult() for _ in events]

    def _parse_extraction_data(self, data: dict) -> ExtractionResult:
        """Parse LLM extraction JSON into an ExtractionResult."""
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


async def extract_batch(
    store, events: list[Event], batch_size: int = 10
) -> list[ExtractionResult]:
    """
    Extract from multiple events using batch-combined extraction.

    Uses ExtractionPipeline.extract_batch_combined() to process events
    in groups, dramatically reducing LLM calls at high volume.
    """
    pipeline = ExtractionPipeline(store)
    results = []

    # Process in batches
    for i in range(0, len(events), batch_size):
        batch = events[i : i + batch_size]
        try:
            batch_results = await pipeline.extract_batch_combined(batch)
            results.extend(batch_results)
        except Exception as e:
            logger.warning("Batch extraction failed for batch starting at %d: %s", i, e)
            results.extend([ExtractionResult() for _ in batch])

    return results
