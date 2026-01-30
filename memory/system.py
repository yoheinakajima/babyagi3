"""
Memory System - The main interface that ties all layers together.

This is the primary class that agent.py uses to interact with memory.
It provides a clean, intuitive API while hiding the complexity of the
three-layer architecture.
"""

import asyncio
from pathlib import Path
from typing import Callable, Awaitable, Any

from .models import Event, Entity, Edge, SliceKey, OwnerInfo, ContextLayer
from .event_log import EventLog
from .graph import Graph
from .summaries import SummaryTree, SummaryRefresher
from .context import ContextAssembler, ContextConfig
from .retrieval import QuickRetrieval, DeepRetrieval, RetrievalResult
from .extraction import ExtractionPipeline, TypeClusterer


class MemorySystem:
    """
    Unified interface to the three-layer memory architecture.

    Layers:
        1. Event Log - Immutable record of everything
        2. Graph - Extracted entities and relationships
        3. Summary Tree - Pre-computed summaries for fast context

    Usage:
        memory = MemorySystem("~/.babyagi/memory")
        memory.set_llm_fn(agent.call_llm)  # For extraction/summarization

        # Log events (called automatically by agent)
        memory.log("message_in", content, tags)

        # Assemble context (called when building system prompt)
        context = memory.assemble_context(tags, active_topics)

        # Retrieval (called when agent needs more info)
        result = memory.quick_retrieve("john's email")
        result = await memory.deep_retrieve("history with john?", agent)
    """

    def __init__(self, storage_path: str | Path | None = None):
        """
        Initialize the memory system.

        Args:
            storage_path: Directory for persistent storage. If None, runs in-memory.
        """
        if storage_path:
            base = Path(storage_path).expanduser()
            event_path = base / "events.jsonl"
            graph_path = base / "graph.json"
            summary_path = base / "summaries.json"
        else:
            event_path = None
            graph_path = None
            summary_path = None

        # Initialize layers
        self.event_log = EventLog(event_path)
        self.graph = Graph(graph_path)
        self.summary_tree = SummaryTree(summary_path)

        # Initialize components
        self.context = ContextAssembler(
            self.event_log,
            self.summary_tree,
            self.graph,
        )
        self.quick = QuickRetrieval(
            self.event_log,
            self.graph,
            self.summary_tree,
        )
        self.deep = DeepRetrieval(
            self.event_log,
            self.graph,
            self.summary_tree,
        )
        self.extraction = ExtractionPipeline(
            self.event_log,
            self.graph,
            self.summary_tree,
        )
        self.clusterer = TypeClusterer(self.graph)

        # Background refresher (not started by default)
        self._refresher: SummaryRefresher | None = None
        self._refresher_task: asyncio.Task | None = None

        # Wire up event subscription for extraction
        self.event_log.subscribe(self.extraction.on_event)

    def set_llm_fn(self, fn: Callable[[str], Awaitable[str]]):
        """
        Set the LLM function used for extraction and summarization.

        The function should take a prompt string and return the LLM response.

        Example:
            async def call_llm(prompt):
                response = await agent._call_api(
                    system="You are a helpful assistant.",
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text

            memory.set_llm_fn(call_llm)
        """
        self.extraction.set_extract_fn(fn)
        self.clusterer.set_cluster_fn(fn)

        # Set summarize function for the summary tree
        async def summarize(slice_key: SliceKey, events: list[Event]) -> str:
            prompt = self._build_summary_prompt(slice_key, events)
            return await fn(prompt)

        self.summary_tree.set_summarize_fn(summarize)

    def _build_summary_prompt(self, slice_key: SliceKey, events: list[Event]) -> str:
        """Build a prompt for summarizing a slice."""
        event_text = "\n".join(
            f"- [{e.type}] {str(e.content)[:200]}"
            for e in events[:50]
        )

        return f"""Summarize the following events for the slice "{slice_key.key}".

Events:
{event_text}

Write a concise summary (2-4 sentences) that captures:
- Key facts and information
- Important patterns or themes
- Notable relationships or entities

Keep it factual and specific. This summary will be used for context in future conversations."""

    def set_owner(self, owner: OwnerInfo):
        """Set owner information for context assembly."""
        self.context.set_owner_info(owner)

    def set_agent_info(self, info: str):
        """Set static agent information for context assembly."""
        self.context.set_agent_info(info)

    # === Event Logging ===

    def log(
        self,
        type: str,
        content: Any,
        tags: dict[str, str],
    ) -> Event:
        """
        Log an event to the event log.

        This is the primary way to record things that happen.
        Tags are deterministic metadata used for slicing.

        Args:
            type: Event type (message_in, message_out, tool_call, etc.)
            content: Event content (flexible - dict, string, etc.)
            tags: Deterministic tags for indexing

        Returns:
            The created event
        """
        event = self.event_log.log(type, content, tags)

        # Mark affected summaries as stale
        self.summary_tree.mark_stale_for_event(event)

        return event

    def log_message(
        self,
        direction: str,  # "in" or "out"
        content: str,
        channel: str,
        person: str | None = None,
        is_owner: bool = False,
        **extra_tags,
    ) -> Event:
        """
        Convenience method for logging messages.

        Args:
            direction: "in" for incoming, "out" for outgoing
            content: Message content
            channel: Channel (email, cli, sms, etc.)
            person: Person involved (if known)
            is_owner: Whether message is to/from owner
            **extra_tags: Additional tags
        """
        tags = {
            "channel": channel,
            "direction": direction,
            "is_owner": str(is_owner).lower(),
        }
        if person:
            tags["person"] = person
        tags.update(extra_tags)

        return self.log(
            type=f"message_{direction}",
            content={"text": content},
            tags=tags,
        )

    def log_tool_call(
        self,
        tool_name: str,
        input: dict,
        result: Any,
        **extra_tags,
    ) -> Event:
        """Convenience method for logging tool calls."""
        tags = {"tool": tool_name, **extra_tags}
        return self.log(
            type="tool_call",
            content={"tool": tool_name, "input": input, "result": result},
            tags=tags,
        )

    # === Context Assembly ===

    def assemble_context(
        self,
        tags: dict[str, str],
        active_topics: list[str] | None = None,
        current_tool: str | None = None,
        current_task: str | None = None,
    ) -> str:
        """
        Assemble context for a new interaction.

        This is called when building the system prompt. It returns
        pre-computed summaries and relevant information based on
        the current context.

        No LLM calls - just fast lookups.

        Args:
            tags: Tags for the current context (channel, person, etc.)
            active_topics: Currently active topics
            current_tool: Tool being used (if any)
            current_task: Task being executed (if any)

        Returns:
            Formatted context string for the system prompt
        """
        return self.context.assemble(
            event_tags=tags,
            active_topics=active_topics,
            current_tool=current_tool,
            current_task=current_task,
        )

    # === Retrieval ===

    def quick_retrieve(
        self,
        query: str,
        entity_types: list[str] | None = None,
        limit: int = 20,
    ) -> RetrievalResult:
        """
        Fast, programmatic retrieval.

        Use this when context is insufficient but the answer is
        straightforward. Returns structured data that can be
        used directly without further processing.

        Args:
            query: Search query
            entity_types: Filter by entity types
            limit: Maximum results

        Returns:
            RetrievalResult with events, entities, edges, summaries
        """
        return self.quick.search(
            query=query,
            entity_types=entity_types,
            limit=limit,
        )

    def get_entity(self, name: str) -> RetrievalResult:
        """Get everything we know about an entity."""
        return self.quick.get_entity_context(name)

    def get_recent(
        self,
        channel: str | None = None,
        person: str | None = None,
        limit: int = 20,
    ) -> RetrievalResult:
        """Get recent activity."""
        return self.quick.get_recent_activity(channel, person, limit)

    async def deep_retrieve(
        self,
        question: str,
        agent,  # The main agent instance
    ) -> RetrievalResult:
        """
        Thorough agent-based retrieval.

        Use this when quick retrieval isn't enough. Spawns a
        retrieval agent that navigates the memory system to
        synthesize an answer.

        The main agent should send "let me think deeper..." before
        calling this, as it may take a moment.

        Args:
            question: The question to answer
            agent: The main agent instance

        Returns:
            RetrievalResult with synthesized answer and sources
        """
        return await self.deep.retrieve(question, agent)

    # === Background Processing ===

    async def process_extractions(self, limit: int = 10) -> int:
        """
        Process pending events for extraction.

        Call this periodically or after batches of events.
        Extracts entities, edges, and topics from events.

        Returns number of events processed.
        """
        return await self.extraction.process_pending(limit)

    async def refresh_summaries(self, limit: int = 10) -> int:
        """
        Refresh stale summaries.

        Call this periodically to keep summaries up to date.

        Returns number of summaries refreshed.
        """
        return await self.summary_tree.refresh_stale_async(
            self.event_log,
            max_refreshes=limit,
        )

    async def cluster_types(self) -> dict[str, str]:
        """
        Cluster free-form entity types into canonical types.

        Call this periodically to keep the type system organized.

        Returns the clustering mapping.
        """
        return await self.clusterer.cluster_entity_types()

    def start_background_refresh(self, interval: float = 60.0):
        """Start background summary refresh loop."""
        if self._refresher_task:
            return

        self._refresher = SummaryRefresher(
            self.summary_tree,
            self.event_log,
            refresh_interval=interval,
        )

        async def run():
            await self._refresher.start()

        self._refresher_task = asyncio.create_task(run())

    def stop_background_refresh(self):
        """Stop background summary refresh loop."""
        if self._refresher:
            self._refresher.stop()
        if self._refresher_task:
            self._refresher_task.cancel()
            self._refresher_task = None

    # === Statistics ===

    def stats(self) -> dict:
        """Get memory system statistics."""
        return {
            "events": len(self.event_log),
            "entities": self.graph.entity_count(),
            "edges": self.graph.edge_count(),
            "summaries": self.summary_tree.stats(),
            "slices": len(self.event_log.get_all_slices()),
        }

    # === Direct Access (for advanced use) ===

    def get_event_log(self) -> EventLog:
        """Direct access to event log."""
        return self.event_log

    def get_graph(self) -> Graph:
        """Direct access to graph."""
        return self.graph

    def get_summary_tree(self) -> SummaryTree:
        """Direct access to summary tree."""
        return self.summary_tree
