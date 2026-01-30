"""
Memory System Integration - Connects the memory system to the agent.

This module provides:
1. Automatic event logging from agent events
2. Context injection into system prompts
3. Memory tools for the agent
4. Background processing hooks
"""

from typing import TYPE_CHECKING
from pathlib import Path

from .system import MemorySystem
from .models import OwnerInfo
from .tools import create_memory_tool, create_deep_retrieval_tool

if TYPE_CHECKING:
    from agent import Agent


class MemoryIntegration:
    """
    Integrates the memory system with the agent.

    Usage:
        from memory.integration import MemoryIntegration

        agent = Agent()
        memory_integration = MemoryIntegration(agent, "~/.babyagi/memory")
        memory_integration.setup()

        # Now the agent has memory capabilities:
        # - Events are automatically logged
        # - Context includes relevant summaries
        # - Memory tools are available
    """

    def __init__(
        self,
        agent: "Agent",
        storage_path: str | Path = "~/.babyagi/memory",
    ):
        self.agent = agent
        self.memory = MemorySystem(storage_path)
        self._original_system_prompt = None

    def setup(self):
        """
        Set up the memory integration.

        This:
        1. Subscribes to agent events for automatic logging
        2. Registers memory tools
        3. Sets up the LLM function for extraction/summarization
        4. Injects context into the system prompt
        """
        self._setup_event_logging()
        self._setup_tools()
        self._setup_llm()
        self._setup_context_injection()

        # Set owner info from agent config
        owner_config = self.agent.config.get("owner", {})
        if owner_config:
            self.memory.set_owner(OwnerInfo(
                id=owner_config.get("id", "owner"),
                name=owner_config.get("name", "Owner"),
                email=owner_config.get("email", ""),
                contacts=owner_config.get("contacts", {}),
            ))

    def _setup_event_logging(self):
        """Subscribe to agent events and log them."""

        @self.agent.on("tool_start")
        def on_tool_start(event):
            tags = {
                "tool": event["name"],
                **self._get_context_tags(),
            }
            self.memory.log("tool_start", event, tags)

        @self.agent.on("tool_end")
        def on_tool_end(event):
            tags = {
                "tool": event["name"],
                **self._get_context_tags(),
            }
            self.memory.log("tool_end", event, tags)

        @self.agent.on("objective_start")
        def on_objective_start(event):
            tags = {
                "task": event["id"],
                "type": "objective",
            }
            self.memory.log("objective_start", event, tags)

        @self.agent.on("objective_end")
        def on_objective_end(event):
            tags = {
                "task": event["id"],
                "type": "objective",
            }
            self.memory.log("objective_end", event, tags)

        @self.agent.on("task_start")
        def on_task_start(event):
            tags = {
                "task": event["id"],
                "type": "scheduled",
            }
            self.memory.log("task_start", event, tags)

        @self.agent.on("task_end")
        def on_task_end(event):
            tags = {
                "task": event["id"],
                "type": "scheduled",
            }
            self.memory.log("task_end", event, tags)

    def _get_context_tags(self) -> dict:
        """Get tags from the current agent context."""
        ctx = getattr(self.agent, "_current_context", {})
        tags = {}

        if ctx.get("channel"):
            tags["channel"] = ctx["channel"]
        if ctx.get("sender"):
            tags["person"] = ctx["sender"]
        if ctx.get("is_owner") is not None:
            tags["is_owner"] = str(ctx["is_owner"]).lower()

        return tags

    def _setup_tools(self):
        """Register memory tools with the agent."""
        from agent import Tool

        # Create memory tool
        memory_tool_def = create_memory_tool(self.memory)
        memory_tool = Tool(
            name=memory_tool_def["name"],
            description=memory_tool_def["description"],
            parameters=memory_tool_def["input_schema"],
            fn=memory_tool_def["execute"],
        )

        # Replace the default memory tool
        self.agent.tools["memory"] = memory_tool

    def _setup_llm(self):
        """Set up the LLM function for extraction and summarization."""

        async def call_llm(prompt: str) -> str:
            import asyncio

            response = await asyncio.to_thread(
                self.agent.client.messages.create,
                model=self.agent.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        self.memory.set_llm_fn(call_llm)

    def _setup_context_injection(self):
        """
        Inject memory context into the system prompt.

        This wraps the agent's _system_prompt method to add memory context.
        """
        original_method = self.agent._system_prompt

        def enhanced_system_prompt(thread_id: str = "main", is_owner: bool = True, context: dict = None) -> str:
            # Get the original prompt
            base_prompt = original_method(thread_id, is_owner, context)

            # Assemble memory context
            tags = self._get_context_tags()

            # Get active topics from recent interactions (simplified)
            active_topics = []  # Could be derived from agent state

            memory_context = self.memory.assemble_context(
                tags=tags,
                active_topics=active_topics,
            )

            if memory_context:
                return f"{base_prompt}\n\n--- MEMORY CONTEXT ---\n{memory_context}"
            return base_prompt

        self.agent._system_prompt = enhanced_system_prompt

    def log_message(
        self,
        direction: str,
        content: str,
        channel: str,
        person: str | None = None,
        is_owner: bool = False,
    ):
        """
        Log a message event.

        Call this from listeners when messages arrive or are sent.
        """
        return self.memory.log_message(
            direction=direction,
            content=content,
            channel=channel,
            person=person,
            is_owner=is_owner,
        )

    async def process_background(self):
        """
        Run background processing tasks.

        Call this periodically or in a background loop.
        Processes extractions and refreshes stale summaries.
        """
        # Process pending extractions
        await self.memory.process_extractions(limit=5)

        # Refresh stale summaries
        await self.memory.refresh_summaries(limit=3)

    def start_background_tasks(self, interval: float = 60.0):
        """Start background processing loop."""
        import asyncio

        async def background_loop():
            while True:
                try:
                    await self.process_background()
                except Exception:
                    pass  # Log but don't crash
                await asyncio.sleep(interval)

        asyncio.create_task(background_loop())

    def get_memory_system(self) -> MemorySystem:
        """Get the underlying memory system for direct access."""
        return self.memory


def setup_memory(agent: "Agent", storage_path: str = "~/.babyagi/memory") -> MemoryIntegration:
    """
    Convenience function to set up memory integration.

    Usage:
        from memory.integration import setup_memory

        agent = Agent()
        memory = setup_memory(agent)
    """
    integration = MemoryIntegration(agent, storage_path)
    integration.setup()
    return integration
