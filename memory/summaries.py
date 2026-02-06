"""
Summary management for the memory system.

Handles summary generation, updates, and staleness propagation.
"""

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

from metrics import LiteLLMAnthropicAdapter, track_source, get_model_for_use_case
from .embeddings import get_embedding
from .models import SummaryNode


# ═══════════════════════════════════════════════════════════
# STALENESS PRIORITY QUEUE
# ═══════════════════════════════════════════════════════════


@dataclass
class StalenessConfig:
    """Configuration for staleness-based refresh."""

    threshold: int = 10  # Events before refresh
    max_batch_size: int = 20  # Max nodes to refresh at once
    leaf_priority_boost: float = 1.5  # Multiply staleness for leaves
    age_factor: float = 0.1  # Add staleness per day since last update


@dataclass(order=True)
class PriorityNode:
    """A node with computed priority for the refresh queue."""

    priority: float
    node: SummaryNode = field(compare=False)

    @classmethod
    def compute(cls, node: SummaryNode, config: StalenessConfig) -> "PriorityNode":
        """Compute priority for a node."""
        # Base priority is events_since_update
        priority = float(node.events_since_update)

        # Boost priority for leaf nodes (they feed into branches)
        if node.node_type in ("entity", "topic", "task"):
            priority *= config.leaf_priority_boost

        # Add age factor
        if node.summary_updated_at:
            days_since_update = (datetime.now() - node.summary_updated_at).days
            priority += days_since_update * config.age_factor

        # Higher priority = more negative (for min-heap with max priority)
        return cls(priority=-priority, node=node)


class StalenessQueue:
    """
    Priority queue for stale summary nodes.

    Prioritizes nodes based on:
    - Number of events since last update
    - Node type (leaves before branches)
    - Time since last update
    """

    def __init__(self, store, config: StalenessConfig | None = None):
        self.store = store
        self.config = config or StalenessConfig()
        self._heap: list[PriorityNode] = []
        self._node_ids: set[str] = set()  # Track nodes in queue

    def refresh(self):
        """Refresh the queue from database."""
        stale_nodes = self.store.get_stale_nodes(self.config.threshold)
        self._heap = []
        self._node_ids = set()

        for node in stale_nodes:
            self._add_node(node)

    def _add_node(self, node: SummaryNode):
        """Add a node to the queue."""
        if node.id in self._node_ids:
            return
        priority_node = PriorityNode.compute(node, self.config)
        heapq.heappush(self._heap, priority_node)
        self._node_ids.add(node.id)

    def pop(self) -> SummaryNode | None:
        """Get the highest priority node."""
        while self._heap:
            priority_node = heapq.heappop(self._heap)
            self._node_ids.discard(priority_node.node.id)

            # Verify node is still stale (may have been refreshed)
            current = self.store.get_summary_node_by_id(priority_node.node.id)
            if current and current.events_since_update >= self.config.threshold:
                return current

        return None

    def get_batch(self, size: int | None = None) -> list[SummaryNode]:
        """Get a batch of high-priority nodes."""
        size = size or self.config.max_batch_size
        batch = []

        while len(batch) < size:
            node = self.pop()
            if node is None:
                break
            batch.append(node)

        return batch

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0


# ═══════════════════════════════════════════════════════════
# SUMMARY MANAGER
# ═══════════════════════════════════════════════════════════


class SummaryManager:
    """
    Manages summary node updates and refreshes.

    Summaries roll up through the tree:
    - Leaf nodes (entities, topics, tasks) summarize their events
    - Branch nodes (types, channels) synthesize their children
    - Root summarizes everything
    """

    def __init__(self, store, staleness_config: StalenessConfig | None = None):
        self.store = store
        self.staleness_config = staleness_config or StalenessConfig()
        self._client = None
        self._queue: StalenessQueue | None = None

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
    def queue(self) -> StalenessQueue:
        """Get the staleness priority queue."""
        if self._queue is None:
            self._queue = StalenessQueue(self.store, self.staleness_config)
        return self._queue

    async def refresh_stale(self, threshold: int | None = None) -> int:
        """
        Refresh stale summary nodes using priority queue.

        Returns the number of summaries refreshed.
        """
        # Update threshold if provided
        if threshold is not None:
            self.staleness_config.threshold = threshold

        # Refresh the queue from database
        self.queue.refresh()

        refreshed = 0
        batch = self.queue.get_batch()

        # Process in priority order
        for node in batch:
            try:
                await self.refresh_node(node)
                refreshed += 1
            except Exception as e:
                logger.warning("Failed to refresh summary %s: %s", node.key, e)

        return refreshed

    async def refresh_stale_batch(self, batch_size: int = 10) -> int:
        """
        Refresh a specific number of stale nodes by priority.

        Returns the number of summaries refreshed.
        """
        self.queue.refresh()
        batch = self.queue.get_batch(batch_size)

        refreshed = 0
        for node in batch:
            try:
                await self.refresh_node(node)
                refreshed += 1
            except Exception as e:
                logger.warning("Failed to refresh summary %s: %s", node.key, e)

        return refreshed

    async def refresh_node(self, node: SummaryNode):
        """
        Refresh a single summary node.

        The strategy depends on node type:
        - Leaf nodes: Summarize from events
        - Branch nodes: Synthesize from children
        - Root: High-level overview
        """
        if node.node_type in ("entity", "topic", "task"):
            await self._refresh_leaf_node(node)
        elif node.node_type == "root":
            await self._refresh_root_node(node)
        else:
            await self._refresh_branch_node(node)

        # Mark parent as needing update
        if node.parent_id:
            self.store.increment_staleness(node.parent_id)

    async def _refresh_leaf_node(self, node: SummaryNode):
        """Refresh a leaf node by summarizing its events."""
        # Get source events
        events = self._get_events_for_node(node, limit=30)

        if not events:
            # No events, keep existing summary or set default
            if not node.summary:
                node.summary = f"No information yet about {node.label}."
                node.events_since_update = 0
                node.summary_updated_at = datetime.now()
                self.store.update_summary_node(node)
            return

        # Build prompt
        events_text = "\n".join(
            [
                f"[{e.timestamp.strftime('%Y-%m-%d %H:%M') if e.timestamp else 'unknown'}] "
                f"({e.channel or 'internal'}, {e.direction}): {e.content[:500]}"
                for e in events
            ]
        )

        prompt = f"""Summarize what is known about "{node.label}" based on these events.

Previous summary:
{node.summary or 'None'}

Recent events ({len(events)}):
{events_text}

Generate a concise summary (2-5 sentences) that:
- Captures the key facts and patterns
- Preserves important historical context
- Is written as factual knowledge, not a log
- Highlights any significant changes

Summary:"""

        with track_source("summary"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

        new_summary = response.content[0].text.strip()

        # Update node
        node.summary = new_summary
        node.summary_embedding = get_embedding(new_summary)
        node.summary_updated_at = datetime.now()
        node.events_since_update = 0

        self.store.update_summary_node(node)

    async def _refresh_branch_node(self, node: SummaryNode):
        """Refresh a branch node by synthesizing children."""
        # Get children
        children = self.store.get_children(node.id)

        if not children:
            # No children, keep existing or set default
            if not node.summary:
                node.summary = f"No information yet for {node.label}."
                node.events_since_update = 0
                node.summary_updated_at = datetime.now()
                self.store.update_summary_node(node)
            return

        # Build prompt with child summaries
        children_text = "\n".join(
            [f"- **{c.label}**: {c.summary or 'No summary yet'}" for c in children]
        )

        prompt = f"""Synthesize a summary for the category "{node.label}" based on its sub-categories.

Previous summary:
{node.summary or 'None'}

Sub-categories ({len(children)}):
{children_text}

Generate a concise summary (3-7 sentences) that:
- Captures key themes across sub-categories
- Notes patterns or notable items
- Provides a useful overview
- Is written as factual knowledge

Summary:"""

        with track_source("summary"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )

        new_summary = response.content[0].text.strip()

        # Update node
        node.summary = new_summary
        node.summary_embedding = get_embedding(new_summary)
        node.summary_updated_at = datetime.now()
        node.events_since_update = 0

        self.store.update_summary_node(node)

    async def _refresh_root_node(self, node: SummaryNode):
        """Refresh the root node with a high-level overview."""
        # Get top-level children
        children = self.store.get_children(node.id)

        if not children:
            node.summary = "No information stored yet."
            node.events_since_update = 0
            node.summary_updated_at = datetime.now()
            self.store.update_summary_node(node)
            return

        # Get agent state for context
        state = self.store.get_agent_state()

        # Build overview from children
        children_text = "\n".join(
            [
                f"- **{c.label}** ({c.event_count} events): {c.summary[:200] if c.summary else 'No summary'}..."
                for c in children
                if c.event_count > 0
            ]
        )

        prompt = f"""Create a high-level knowledge summary for an AI assistant named "{state.name}".

Categories of knowledge:
{children_text}

Generate a brief overview (3-5 sentences) that:
- Summarizes what the assistant knows
- Highlights key areas of expertise/information
- Notes any active areas of focus
- Is written in first person ("I know about...")

Overview:"""

        with track_source("summary"):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )

        new_summary = response.content[0].text.strip()

        # Update node
        node.summary = new_summary
        node.summary_embedding = get_embedding(new_summary)
        node.summary_updated_at = datetime.now()
        node.events_since_update = 0

        self.store.update_summary_node(node)

    def _get_events_for_node(self, node: SummaryNode, limit: int = 30):
        """Get events for a summary node based on its type."""
        parts = node.key.split(":", 1)
        if len(parts) != 2:
            return self.store.get_recent_events(limit=limit)

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
        else:
            return self.store.get_recent_events(limit=limit)

    async def create_initial_summaries(self):
        """
        Create initial summaries for all nodes that don't have one.

        Useful for bootstrapping or after bulk import.
        """
        cur = self.store.conn.cursor()
        cur.execute(
            "SELECT * FROM summary_nodes WHERE summary IS NULL OR summary = ''"
        )
        nodes = [self.store._row_to_summary_node(row) for row in cur.fetchall()]

        for node in nodes:
            try:
                await self.refresh_node(node)
            except Exception as e:
                logger.warning("Failed to create initial summary for %s: %s", node.key, e)

    def get_recent_summary(self, limit: int = 10) -> str:
        """
        Generate a summary of recent activity.

        This is computed on-demand, not cached.
        """
        events = self.store.get_recent_events(limit=limit)

        if not events:
            return "No recent activity."

        # Simple summary without LLM call
        channels = set()
        people = set()
        event_types = set()

        for e in events:
            if e.channel:
                channels.add(e.channel)
            if e.person_id:
                entity = self.store.get_entity(e.person_id)
                if entity:
                    people.add(entity.name)
            event_types.add(e.event_type)

        parts = []
        if channels:
            parts.append(f"Active channels: {', '.join(channels)}")
        if people:
            parts.append(f"Recent interactions: {', '.join(list(people)[:5])}")
        if event_types:
            parts.append(f"Activity types: {', '.join(event_types)}")

        return ". ".join(parts) + "." if parts else "Recent activity across various channels."


async def ensure_default_summaries(store):
    """
    Ensure default summary nodes exist for common categories.

    Call this during initialization.
    """
    manager = SummaryManager(store)

    # Default channels
    for channel in ["email", "sms", "cli", "api", "web"]:
        store.create_summary_node(
            node_type="channel",
            key=f"channel:{channel}",
            label=channel.upper(),
            parent_key="root",
            summary=f"No {channel} activity yet.",
        )

    # Default entity types
    for etype in ["person", "org", "tool", "concept", "location"]:
        store.create_summary_node(
            node_type="entity_type",
            key=f"entity_type:{etype}",
            label=etype.replace("_", " ").title(),
            parent_key="root",
            summary=f"No {etype} entities tracked yet.",
        )

    # Default relation types
    for rtype in ["professional", "financial", "social", "technical"]:
        store.create_summary_node(
            node_type="relation_type",
            key=f"relation_type:{rtype}",
            label=f"{rtype.title()} Relationships",
            parent_key="root",
            summary=f"No {rtype} relationships tracked yet.",
        )
