"""
Context Assembly - Deterministic construction of context from pre-computed summaries.

The key insight: No LLM calls needed at assembly time. Everything is lookups.

Context is built in layers:
1. Always included: agent info, owner info, root summary, recent history
2. Conditional: channel, tool, task, topic, person summaries based on tags
"""

from dataclasses import dataclass, field
from .models import SliceKey, ContextLayer, OwnerInfo, AgentState, Event, Entity, Edge
from .summaries import SummaryTree
from .event_log import EventLog
from .graph import Graph


@dataclass
class ContextConfig:
    """Configuration for context assembly."""

    # Token budgets
    max_total_tokens: int = 8000
    root_summary_tokens: int = 500
    recent_history_tokens: int = 1500
    channel_tokens: int = 400
    person_tokens: int = 600
    topic_tokens: int = 400
    tool_tokens: int = 300
    task_tokens: int = 400

    # Event limits
    recent_event_count: int = 20
    channel_event_count: int = 10
    person_event_count: int = 10

    # Enable/disable layers
    include_root_summary: bool = True
    include_recent_history: bool = True
    include_channel_context: bool = True
    include_person_context: bool = True
    include_topic_context: bool = True
    include_tool_context: bool = True
    include_task_context: bool = True


class ContextAssembler:
    """
    Assembles context deterministically from pre-computed summaries.

    No LLM calls - just fast lookups and formatting.
    """

    def __init__(
        self,
        event_log: EventLog,
        summary_tree: SummaryTree,
        graph: Graph,
        config: ContextConfig | None = None,
    ):
        self.event_log = event_log
        self.summary_tree = summary_tree
        self.graph = graph
        self.config = config or ContextConfig()

        # Static context (set once)
        self.agent_info: str = ""
        self.owner_info: OwnerInfo | None = None

    def set_agent_info(self, info: str):
        """Set static agent information."""
        self.agent_info = info

    def set_owner_info(self, owner: OwnerInfo):
        """Set owner information."""
        self.owner_info = owner

    def assemble(
        self,
        event_tags: dict[str, str],
        active_topics: list[str] | None = None,
        current_tool: str | None = None,
        current_task: str | None = None,
    ) -> str:
        """
        Assemble context for a new event.

        Args:
            event_tags: Tags for the current event (channel, person, is_owner, etc.)
            active_topics: Currently active topics from agent state
            current_tool: Tool being used (if any)
            current_task: Task being executed (if any)

        Returns:
            Formatted context string ready for system prompt
        """
        layers = []

        # === Always Included ===

        # Agent info
        if self.agent_info:
            layers.append(ContextLayer(
                name="agent_info",
                priority=100,
                content=self.agent_info,
            ))

        # Owner info
        if self.owner_info:
            owner_text = self._format_owner_info()
            layers.append(ContextLayer(
                name="owner_info",
                priority=95,
                content=owner_text,
            ))

        # Root summary (everything you know)
        if self.config.include_root_summary:
            root_summary = self.summary_tree.get_text(SliceKey.root())
            if root_summary:
                layers.append(ContextLayer(
                    name="knowledge_summary",
                    priority=90,
                    content=f"## What You Know\n{root_summary}",
                    source="*",
                ))

        # Recent history
        if self.config.include_recent_history:
            recent = self.event_log.recent(self.config.recent_event_count)
            if recent:
                recent_text = self._format_events(recent, "Recent History")
                layers.append(ContextLayer(
                    name="recent_history",
                    priority=85,
                    content=recent_text,
                ))

        # === Conditional Layers ===

        # Channel context
        channel = event_tags.get("channel")
        if channel and self.config.include_channel_context:
            channel_slice = SliceKey({"channel": channel})
            channel_summary = self.summary_tree.get_text(channel_slice)
            if channel_summary:
                layers.append(ContextLayer(
                    name=f"channel_{channel}",
                    priority=70,
                    content=f"## {channel.title()} Channel Context\n{channel_summary}",
                    source=channel_slice.key,
                ))

            # Recent channel events
            channel_events = self.event_log.recent(
                self.config.channel_event_count,
                slice_key=channel_slice,
            )
            if channel_events:
                layers.append(ContextLayer(
                    name=f"channel_{channel}_events",
                    priority=65,
                    content=self._format_events(channel_events, f"Recent {channel.title()} Events"),
                ))

        # Person context
        person = event_tags.get("person")
        if person and self.config.include_person_context:
            person_layer = self._build_person_context(person)
            if person_layer:
                layers.append(person_layer)

        # Topic contexts
        if active_topics and self.config.include_topic_context:
            for topic in active_topics[:3]:  # Limit to top 3 topics
                topic_slice = SliceKey({"topic": topic})
                topic_summary = self.summary_tree.get_text(topic_slice)
                if topic_summary:
                    layers.append(ContextLayer(
                        name=f"topic_{topic}",
                        priority=50,
                        content=f"## Topic: {topic.title()}\n{topic_summary}",
                        source=topic_slice.key,
                    ))

        # Tool context
        if current_tool and self.config.include_tool_context:
            tool_slice = SliceKey({"tool": current_tool})
            tool_summary = self.summary_tree.get_text(tool_slice)
            if tool_summary:
                layers.append(ContextLayer(
                    name=f"tool_{current_tool}",
                    priority=40,
                    content=f"## Tool: {current_tool}\n{tool_summary}",
                    source=tool_slice.key,
                ))

        # Task context
        if current_task and self.config.include_task_context:
            task_slice = SliceKey({"task": current_task})
            task_summary = self.summary_tree.get_text(task_slice)
            if task_summary:
                layers.append(ContextLayer(
                    name=f"task_{current_task}",
                    priority=45,
                    content=f"## Current Task\n{task_summary}",
                    source=task_slice.key,
                ))

        # Sort by priority and assemble
        layers.sort()  # Uses __lt__ which sorts by priority descending
        return self._combine_layers(layers)

    def _build_person_context(self, person: str) -> ContextLayer | None:
        """Build context about a specific person."""
        parts = []

        # Summary from slice
        person_slice = SliceKey({"person": person})
        person_summary = self.summary_tree.get_text(person_slice)
        if person_summary:
            parts.append(person_summary)

        # Entity info from graph
        entity = self.graph.find_entity(person)
        if entity:
            if entity.attributes:
                attrs = ", ".join(f"{k}: {v}" for k, v in entity.attributes.items())
                parts.append(f"Known attributes: {attrs}")

            # Relationships
            neighbors = self.graph.get_neighbors(entity.id)
            if neighbors:
                rels = []
                for neighbor, edge in neighbors[:5]:  # Limit relationships
                    if edge.source_id == entity.id:
                        rels.append(f"{edge.type} {neighbor.name}")
                    else:
                        rels.append(f"{neighbor.name} {edge.type} them")
                parts.append(f"Relationships: {', '.join(rels)}")

        # Recent interactions
        person_events = self.event_log.recent(
            self.config.person_event_count,
            slice_key=person_slice,
        )
        if person_events:
            parts.append(self._format_events(person_events, "Recent Interactions", compact=True))

        if parts:
            return ContextLayer(
                name=f"person_{person}",
                priority=60,
                content=f"## About: {person}\n" + "\n\n".join(parts),
                source=person_slice.key,
            )
        return None

    def _format_owner_info(self) -> str:
        """Format owner information."""
        if not self.owner_info:
            return ""

        lines = [f"## Your Owner: {self.owner_info.name}"]
        lines.append(f"Email: {self.owner_info.email}")

        if self.owner_info.contacts:
            contacts = ", ".join(f"{k}: {v}" for k, v in self.owner_info.contacts.items())
            lines.append(f"Contact methods: {contacts}")

        if self.owner_info.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.owner_info.preferences.items())
            lines.append(f"Preferences: {prefs}")

        return "\n".join(lines)

    def _format_events(
        self,
        events: list[Event],
        title: str,
        compact: bool = False,
    ) -> str:
        """Format a list of events for context."""
        lines = [f"## {title}"]

        for event in events:
            if compact:
                # One line per event
                content_preview = str(event.content)[:100]
                lines.append(f"- [{event.type}] {content_preview}")
            else:
                # More detail
                lines.append(f"\n### {event.type} ({event.timestamp[:16]})")
                if isinstance(event.content, dict):
                    for k, v in event.content.items():
                        v_str = str(v)[:200]
                        lines.append(f"  {k}: {v_str}")
                else:
                    lines.append(f"  {str(event.content)[:300]}")

        return "\n".join(lines)

    def _combine_layers(self, layers: list[ContextLayer]) -> str:
        """Combine layers into final context string."""
        # Simple concatenation for now
        # Could add token budget enforcement here
        parts = [layer.content for layer in layers if layer.content]
        return "\n\n---\n\n".join(parts)

    def get_context_sources(
        self,
        event_tags: dict[str, str],
        active_topics: list[str] | None = None,
    ) -> list[str]:
        """
        Get the slice keys that would be used for context.

        Useful for debugging and understanding what context was assembled.
        """
        sources = ["*"]  # Root always included

        channel = event_tags.get("channel")
        if channel:
            sources.append(f"channel:{channel}")

        person = event_tags.get("person")
        if person:
            sources.append(f"person:{person}")

        if active_topics:
            for topic in active_topics:
                sources.append(f"topic:{topic}")

        return sources
