"""
Summary management for the memory system.

Handles summary generation, updates, and staleness propagation.
"""

from datetime import datetime

from .embeddings import get_embedding
from .models import SummaryNode


class SummaryManager:
    """
    Manages summary node updates and refreshes.

    Summaries roll up through the tree:
    - Leaf nodes (entities, topics, tasks) summarize their events
    - Branch nodes (types, channels) synthesize their children
    - Root summarizes everything
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

    async def refresh_stale(self, threshold: int = 10) -> int:
        """
        Refresh all stale summary nodes.

        Returns the number of summaries refreshed.
        """
        stale_nodes = self.store.get_stale_nodes(threshold)
        refreshed = 0

        # Process in order (leaves first due to ordering in query)
        for node in stale_nodes:
            try:
                await self.refresh_node(node)
                refreshed += 1
            except Exception as e:
                print(f"Failed to refresh summary {node.key}: {e}")

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

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
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

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
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

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
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
                print(f"Failed to create initial summary for {node.key}: {e}")

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
