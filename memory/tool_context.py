"""
Tool Context Builder - Intelligent tool selection and summarization.

Problem: As the agent creates more tools, sending all tool schemas on every
API call becomes expensive and fills the context window.

Solution: Select a subset of relevant tools based on:
- Core tools (always available)
- Usage frequency (most used tools)
- Recency (recently used tools)
- Relevance (semantically similar to current context)

Plus: Generate summaries of all available tools so the agent knows what
exists and can discover more via the memory tool.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .store import MemoryStore


@dataclass
class ToolContextConfig:
    """Configuration for tool context building."""

    # Maximum tools to include in API calls
    max_active_tools: int = 25

    # Allocation of tool slots
    core_tools_reserved: int = 20  # Core + default tools always included
    most_used_slots: int = 3  # High-usage tools
    recent_slots: int = 3  # Recently used tools
    relevant_slots: int = 3  # Contextually relevant tools

    # Time window for "recent" tools
    recent_window_hours: int = 24

    # Minimum usage count to be considered "most used"
    min_usage_for_popular: int = 3

    # Summary generation
    max_tools_per_category_summary: int = 10
    summary_token_budget: int = 500


# Core tools that are always included (agent fundamentals)
CORE_TOOLS = frozenset([
    "memory",
    "objective",
    "notes",
    "schedule",
    "register_tool",
    "send_message",
    "join_meeting",
])

# Default tools that should always be available alongside core tools.
# These represent the agent's standard capabilities (email, meetings, web)
# and must be included regardless of usage history or query relevance,
# otherwise the agent won't know it can use them.
DEFAULT_TOOLS = frozenset([
    # Meeting tools (Recall.ai) — join_meeting is in CORE_TOOLS
    "get_meeting_status",
    "leave_meeting",
    "get_meeting_transcript",
    "list_meeting_bots",
    "search_meeting_memories",
    "configure_meeting_bot",
    # Email tools (AgentMail)
    "send_email",
    "check_inbox",
    "read_email",
    "get_agent_email",
    # Web tools
    "web_search",
    # Calendar tools
    "get_calendar_events",
])


@dataclass
class ToolSelection:
    """Result of tool selection - which tools to include and why."""

    # Tools to include in API call (name -> reason)
    selected_tools: dict[str, str] = field(default_factory=dict)

    # Summary of all available tools for system prompt
    tool_inventory_summary: str = ""

    # Statistics
    total_available: int = 0
    total_selected: int = 0
    categories: list[str] = field(default_factory=list)


class ToolContextBuilder:
    """
    Builds intelligent tool context for API calls.

    Instead of sending all tools on every call, this selects a relevant
    subset and provides a summary of what else is available.

    Usage:
        builder = ToolContextBuilder(store, config)
        selection = builder.select_tools(
            all_tools=agent.tools,
            current_query="search for information about...",
            current_topics=["research", "web"],
        )

        # Use selection.selected_tools for API call
        # Use selection.tool_inventory_summary for system prompt
    """

    def __init__(
        self,
        store: "MemoryStore",
        config: ToolContextConfig | None = None,
    ):
        self.store = store
        self.config = config or ToolContextConfig()

    def select_tools(
        self,
        all_tools: dict,
        current_query: str | None = None,
        current_topics: list[str] | None = None,
        current_channel: str | None = None,
    ) -> ToolSelection:
        """
        Select which tools to include in the next API call.

        Args:
            all_tools: All registered tools (agent.tools)
            current_query: Current user query for relevance matching
            current_topics: Active topic IDs for context
            current_channel: Current channel (email, cli, etc.)

        Returns:
            ToolSelection with selected tools and inventory summary
        """
        selection = ToolSelection()
        selection.total_available = len(all_tools)

        # Step 1: Always include core tools
        core_selected = self._select_core_tools(all_tools)

        # Step 2: Get tool statistics from database
        tool_stats = self._get_tool_stats()

        # Step 3: Select most used tools
        most_used = self._select_most_used(
            all_tools, tool_stats, exclude=core_selected
        )

        # Step 4: Select recently used tools
        recent = self._select_recent(
            all_tools, tool_stats, exclude=core_selected | most_used
        )

        # Step 5: Select contextually relevant tools
        relevant = self._select_relevant(
            all_tools,
            current_query,
            current_topics,
            exclude=core_selected | most_used | recent,
        )

        # Combine selections
        for name in core_selected:
            selection.selected_tools[name] = "core"
        for name in most_used:
            selection.selected_tools[name] = "most_used"
        for name in recent:
            selection.selected_tools[name] = "recent"
        for name in relevant:
            selection.selected_tools[name] = "relevant"

        selection.total_selected = len(selection.selected_tools)

        # Step 6: Build inventory summary
        selection.tool_inventory_summary = self._build_inventory_summary(
            all_tools, tool_stats, selection.selected_tools
        )

        # Collect categories
        selection.categories = list(set(
            self._get_tool_category(name, tool_stats)
            for name in all_tools.keys()
        ))

        return selection

    def _select_core_tools(self, all_tools: dict) -> set[str]:
        """Select core and default tools that are always included."""
        always_include = CORE_TOOLS | DEFAULT_TOOLS
        return {name for name in always_include if name in all_tools}

    def _get_tool_stats(self) -> dict[str, dict]:
        """Get tool statistics from database."""
        stats = {}
        try:
            tool_defs = self.store.get_all_tool_definitions(include_disabled=True)
            for td in tool_defs:
                stats[td.name] = {
                    "usage_count": td.usage_count,
                    "success_count": td.success_count,
                    "error_count": td.error_count,
                    "last_used_at": td.last_used_at,
                    "category": td.category,
                    "is_enabled": td.is_enabled,
                    "is_healthy": td.is_healthy,
                    "description": td.description,
                }
        except Exception as e:
            logger.debug("Could not retrieve tool stats from database: %s", e)
        return stats

    def _select_most_used(
        self,
        all_tools: dict,
        tool_stats: dict,
        exclude: set[str],
    ) -> set[str]:
        """Select most frequently used tools."""
        candidates = []
        for name in all_tools:
            if name in exclude:
                continue
            stats = tool_stats.get(name, {})
            usage = stats.get("usage_count", 0)
            if usage >= self.config.min_usage_for_popular:
                # Only include healthy tools
                if stats.get("is_healthy", True):
                    candidates.append((name, usage))

        # Sort by usage count descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        return {name for name, _ in candidates[: self.config.most_used_slots]}

    def _select_recent(
        self,
        all_tools: dict,
        tool_stats: dict,
        exclude: set[str],
    ) -> set[str]:
        """Select recently used tools."""
        cutoff = datetime.now() - timedelta(hours=self.config.recent_window_hours)
        candidates = []

        for name in all_tools:
            if name in exclude:
                continue
            stats = tool_stats.get(name, {})
            last_used = stats.get("last_used_at")
            if last_used and last_used > cutoff:
                # Only include healthy tools
                if stats.get("is_healthy", True):
                    candidates.append((name, last_used))

        # Sort by recency descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        return {name for name, _ in candidates[: self.config.recent_slots]}

    def _select_relevant(
        self,
        all_tools: dict,
        current_query: str | None,
        current_topics: list[str] | None,
        exclude: set[str],
    ) -> set[str]:
        """
        Select tools relevant to current context.

        For now, uses keyword matching. Can be enhanced with embeddings.
        """
        if not current_query and not current_topics:
            return set()

        # Build context keywords
        keywords = set()
        if current_query:
            # Simple keyword extraction
            keywords.update(
                word.lower()
                for word in current_query.split()
                if len(word) > 3
            )
        if current_topics:
            for topic_id in current_topics:
                try:
                    topic = self.store.get_topic(topic_id)
                    if topic:
                        keywords.update(kw.lower() for kw in topic.keywords)
                except Exception as e:
                    logger.debug("Could not retrieve topic '%s' for keyword extraction: %s", topic_id, e)

        if not keywords:
            return set()

        # Score tools by keyword relevance
        candidates = []
        for name, tool in all_tools.items():
            if name in exclude:
                continue

            # Get tool description
            desc = tool.schema.get("description", "").lower()
            tool_keywords = set(desc.split())

            # Count keyword matches
            matches = len(keywords & tool_keywords)
            if matches > 0:
                candidates.append((name, matches))

        # Sort by relevance descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top N
        return {name for name, _ in candidates[: self.config.relevant_slots]}

    def _get_tool_category(self, name: str, tool_stats: dict) -> str:
        """Get category for a tool."""
        if name in CORE_TOOLS:
            return "core"
        if name in DEFAULT_TOOLS:
            return "default"
        stats = tool_stats.get(name, {})
        return stats.get("category", "custom")

    def _build_inventory_summary(
        self,
        all_tools: dict,
        tool_stats: dict,
        selected: dict[str, str],
    ) -> str:
        """
        Build a summary of all available tools for the system prompt.

        This lets the agent know what tools exist beyond the active set.
        """
        # Group tools by category
        by_category: dict[str, list[tuple[str, str, int]]] = {}
        for name, tool in all_tools.items():
            category = self._get_tool_category(name, tool_stats)
            desc = tool.schema.get("description", "")
            # Get first sentence of description
            first_sentence = desc.split(".")[0] if desc else name
            if len(first_sentence) > 80:
                first_sentence = first_sentence[:77] + "..."

            usage = tool_stats.get(name, {}).get("usage_count", 0)

            if category not in by_category:
                by_category[category] = []
            by_category[category].append((name, first_sentence, usage))

        # Sort categories: core first, then default, then by tool count
        sorted_categories = sorted(
            by_category.keys(),
            key=lambda c: (0 if c == "core" else 1 if c == "default" else 2, -len(by_category[c])),
        )

        # Build summary
        lines = []
        lines.append("## Available Tools")
        lines.append("")
        lines.append(
            f"You have {len(all_tools)} tools available. "
            f"{len(selected)} are currently active (shown in tool list). "
            "Use `memory(action=\"list_tools\")` to see all tools with details."
        )
        lines.append("")

        for category in sorted_categories:
            tools = by_category[category]
            # Sort by usage within category
            tools.sort(key=lambda x: x[2], reverse=True)

            lines.append(f"### {category.title()} ({len(tools)} tools)")

            # Show top tools in each category
            shown = 0
            for name, desc, usage in tools:
                if shown >= self.config.max_tools_per_category_summary:
                    remaining = len(tools) - shown
                    if remaining > 0:
                        lines.append(f"  ... and {remaining} more")
                    break

                active_marker = " [ACTIVE]" if name in selected else ""
                usage_marker = f" (used {usage}x)" if usage > 0 else ""
                lines.append(f"- **{name}**{active_marker}: {desc}{usage_marker}")
                shown += 1

            lines.append("")

        # Add discovery instructions
        lines.append("### Tool Discovery")
        lines.append(
            "To find or inspect tools, use the memory tool:\n"
            "- `memory(action=\"list_tools\")` - List all tools with health/stats\n"
            "- `memory(action=\"get_tool\", tool_name=\"...\")` - Get tool details\n"
            "- `memory(action=\"tool_stats\")` - Aggregate tool statistics\n"
            "- `memory(action=\"problematic_tools\")` - Find unhealthy tools"
        )

        return "\n".join(lines)

    def get_active_tool_schemas(
        self,
        all_tools: dict,
        selection: ToolSelection,
    ) -> list[dict]:
        """
        Get tool schemas for the selected tools only.

        Use this instead of agent._tool_schemas() to send only relevant tools.
        """
        return [
            tool.schema
            for name, tool in all_tools.items()
            if name in selection.selected_tools
        ]


def create_tool_context_builder(store: "MemoryStore") -> ToolContextBuilder:
    """Factory function to create a ToolContextBuilder with default config."""
    return ToolContextBuilder(store)
