"""
Unified Message Abstraction Agent with Background Objectives

Core insight: Everything is still a message, but some messages trigger background work.

- User input → message
- Assistant response → message
- Tool execution → message
- Background objective → messages in its own thread

The elegance: objectives are just agent runs in separate threads.
Chat continues while objectives work. Objectives can spawn sub-objectives.
"""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

import anthropic


# =============================================================================
# Core Data Model
# =============================================================================

@dataclass
class Objective:
    """
    An objective is background work with its own conversation thread.

    Simple objectives: "search for X and summarize"
    Complex objectives: "research Y, create a report, email it to Z"

    The agent handles complexity naturally through its loop.
    """
    id: str
    goal: str
    status: str = "pending"  # pending, running, completed, failed
    thread_id: str = ""
    schedule: str | None = None  # cron expression for recurring
    result: str | None = None
    error: str | None = None
    created: str = ""
    completed: str | None = None

    def __post_init__(self):
        if not self.thread_id:
            self.thread_id = f"objective_{self.id}"
        if not self.created:
            self.created = datetime.now().isoformat()


class Tool:
    """A capability the agent can use."""

    def __init__(self, name: str, description: str, parameters: dict, fn: Callable):
        self.name = name
        self.fn = fn
        self.schema = {
            "name": name,
            "description": description,
            "input_schema": parameters
        }

    def execute(self, params: dict, agent: "Agent"):
        return self.fn(params, agent)


class Agent:
    """
    The agent: a loop that processes messages, with support for background objectives.

    ┌─────────────────────────────────────────┐
    │              Agent                       │
    │  ┌─────────┐  ┌────────────┐            │
    │  │ Threads │  │ Objectives │            │
    │  │  (chat) │  │ (bg work)  │            │
    │  └────┬────┘  └─────┬──────┘            │
    │       └──────┬──────┘                   │
    │              ▼                          │
    │         ┌────────┐                      │
    │         │ Tools  │                      │
    │         └────────┘                      │
    └─────────────────────────────────────────┘
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", load_tools: bool = True):
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools: dict[str, Tool] = {}
        self.threads: dict[str, list] = {"main": []}
        self.objectives: dict[str, Objective] = {}
        self._running_objectives: set[str] = set()
        self._lock = asyncio.Lock()

        # Register core tools
        self._register_core_tools()

        # Load external tools
        if load_tools:
            self._load_external_tools()

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def _register_core_tools(self):
        """Register the built-in tools."""
        self.register(_memory_tool(self))
        self.register(_objective_tool(self))
        self.register(_notes_tool(self))

    def _load_external_tools(self):
        """Load tools from the tools/ folder."""
        try:
            from tools import get_all_tools
            for tool in get_all_tools(Tool):
                self.register(tool)
        except ImportError:
            pass

    # -------------------------------------------------------------------------
    # Message Processing
    # -------------------------------------------------------------------------

    def run(self, user_input: str, thread_id: str = "main") -> str:
        """
        Process user input synchronously.
        For async with background objectives, use run_async().
        """
        return asyncio.get_event_loop().run_until_complete(
            self.run_async(user_input, thread_id)
        )

    async def run_async(self, user_input: str, thread_id: str = "main") -> str:
        """Process user input and return response. Objectives run in background."""
        thread = self.threads.setdefault(thread_id, [])
        thread.append({"role": "user", "content": user_input})

        while True:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=8096,
                system=self._system_prompt(),
                tools=self._tool_schemas(),
                messages=thread
            )

            thread.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            # Execute tools
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.tools[block.name].execute(block.input, self)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            thread.append({"role": "user", "content": tool_results})

    async def run_objective(self, objective_id: str):
        """Execute an objective in the background."""
        obj = self.objectives.get(objective_id)
        if not obj or obj.status != "pending":
            return

        async with self._lock:
            if objective_id in self._running_objectives:
                return
            self._running_objectives.add(objective_id)
            obj.status = "running"

        try:
            # Run the objective with its own thread
            prompt = f"""Complete this objective: {obj.goal}

Work autonomously. Use tools as needed. When done, provide a final summary."""

            result = await self.run_async(prompt, obj.thread_id)
            obj.result = result
            obj.status = "completed"
            obj.completed = datetime.now().isoformat()
        except Exception as e:
            obj.status = "failed"
            obj.error = str(e)
        finally:
            self._running_objectives.discard(objective_id)

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    async def run_scheduler(self):
        """Run the scheduler loop for recurring objectives."""
        while True:
            now = datetime.now()
            for obj in list(self.objectives.values()):
                if obj.schedule and obj.status in ("pending", "completed"):
                    if self._should_run(obj.schedule, now):
                        # Reset and re-run
                        obj.status = "pending"
                        obj.result = None
                        obj.completed = None
                        asyncio.create_task(self.run_objective(obj.id))
            await asyncio.sleep(60)  # Check every minute

    def _should_run(self, schedule: str, now: datetime) -> bool:
        """Simple schedule matching: 'hourly', 'daily', or cron-like."""
        if schedule == "hourly":
            return now.minute == 0
        elif schedule == "daily":
            return now.hour == 0 and now.minute == 0
        elif schedule.startswith("every "):
            # "every 5 minutes", "every 2 hours"
            parts = schedule.split()
            if len(parts) == 3:
                n, unit = int(parts[1]), parts[2]
                if "minute" in unit:
                    return now.minute % n == 0
                elif "hour" in unit:
                    return now.hour % n == 0 and now.minute == 0
        return False

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _system_prompt(self) -> str:
        # Build objective status summary
        obj_summary = ""
        active = [o for o in self.objectives.values() if o.status in ("pending", "running")]
        if active:
            obj_summary = "\n\nActive objectives:\n" + "\n".join(
                f"- [{o.id[:8]}] {o.goal} ({o.status})" for o in active
            )

        return f"""You are a helpful assistant with access to powerful tools.

CAPABILITIES:

1. **Direct Response**: Answer questions and have conversations normally.

2. **Background Objectives**: For tasks that take time (research, multi-step work),
   use the 'objective' tool to spawn background work. Chat can continue while
   objectives run. Use this when:
   - The task is complex or multi-step
   - The user might want to do other things while waiting
   - You say things like "I'll work on that" or "Let me research that"

3. **Recurring Tasks**: Objectives can have a schedule ('hourly', 'daily',
   'every 5 minutes') for recurring work.

4. **Notes**: Simple reminders and todos (passive, unlike objectives).

5. **Memory**: Store and recall facts.

6. **Web/Email/Secrets**: If configured, search web, browse pages, send emails.

7. **Code Execution**: Python code with ANY package can be executed. External packages
   (pandas, numpy, requests, etc.) are auto-detected and installed in a sandbox.
   Stdlib-only code runs locally for speed.
{obj_summary}

Be proactive about using background objectives for complex work."""

    def _tool_schemas(self) -> list:
        return [t.schema for t in self.tools.values()]

    def _extract_text(self, response) -> str:
        return "".join(b.text for b in response.content if hasattr(b, "text"))

    def get_thread(self, thread_id: str = "main") -> list:
        return self.threads.get(thread_id, [])

    def clear_thread(self, thread_id: str = "main"):
        self.threads[thread_id] = []


# =============================================================================
# Core Tools
# =============================================================================

# Shared storage
MEMORIES: list[dict] = []
NOTES: list[dict] = []


def _memory_tool(agent: Agent) -> Tool:
    """Memory: store and recall facts."""

    def fn(params: dict, _agent: Agent) -> dict:
        action = params["action"]

        if action == "store":
            memory = {"content": params["content"], "ts": datetime.now().isoformat()}
            MEMORIES.append(memory)
            return {"stored": True}

        elif action == "search":
            query = params.get("query", "").lower()
            matches = [m for m in MEMORIES if query in m["content"].lower()] if query else MEMORIES
            return {"memories": matches[-10:]}

        elif action == "list":
            return {"memories": MEMORIES[-20:]}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="memory",
        description="Store or search memories. Actions: store (content), search (query), list.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["store", "search", "list"]},
                "content": {"type": "string", "description": "Content to store"},
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["action"]
        },
        fn=fn
    )


def _objective_tool(agent: Agent) -> Tool:
    """Objective management: spawn, list, check background work."""

    def fn(params: dict, ag: Agent) -> dict:
        action = params["action"]

        if action == "spawn":
            obj_id = str(uuid.uuid4())[:8]
            obj = Objective(
                id=obj_id,
                goal=params["goal"],
                schedule=params.get("schedule")
            )
            ag.objectives[obj_id] = obj
            # Start in background
            asyncio.create_task(ag.run_objective(obj_id))
            return {
                "spawned": obj_id,
                "goal": obj.goal,
                "schedule": obj.schedule,
                "message": "Objective started in background. Chat can continue."
            }

        elif action == "list":
            return {"objectives": [
                {"id": o.id, "goal": o.goal, "status": o.status,
                 "schedule": o.schedule, "result": o.result[:200] if o.result else None}
                for o in ag.objectives.values()
            ]}

        elif action == "check":
            obj_id = params["id"]
            obj = ag.objectives.get(obj_id)
            if not obj:
                return {"error": f"Objective {obj_id} not found"}
            return {
                "id": obj.id, "goal": obj.goal, "status": obj.status,
                "result": obj.result, "error": obj.error
            }

        elif action == "cancel":
            obj_id = params["id"]
            obj = ag.objectives.get(obj_id)
            if obj:
                obj.status = "cancelled"
                return {"cancelled": obj_id}
            return {"error": f"Objective {obj_id} not found"}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="objective",
        description="""Manage background objectives (async work).

Actions:
- spawn: Create new objective (goal, optional schedule like 'hourly', 'daily', 'every 5 minutes')
- list: See all objectives and their status
- check: Get details of specific objective (id)
- cancel: Stop an objective (id)

Use spawn for complex/time-consuming tasks. The objective runs in background while chat continues.""",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["spawn", "list", "check", "cancel"]},
                "goal": {"type": "string", "description": "What to accomplish (for spawn)"},
                "schedule": {"type": "string", "description": "Recurring schedule (for spawn)"},
                "id": {"type": "string", "description": "Objective ID (for check/cancel)"}
            },
            "required": ["action"]
        },
        fn=fn
    )


def _notes_tool(agent: Agent) -> Tool:
    """Notes: simple todo list (passive, unlike objectives)."""

    def fn(params: dict, _agent: Agent) -> dict:
        action = params["action"]

        if action == "add":
            note = {
                "id": len(NOTES),
                "text": params["text"],
                "done": False,
                "created": datetime.now().isoformat()
            }
            NOTES.append(note)
            return {"added": note}

        elif action == "list":
            return {"notes": NOTES}

        elif action == "complete":
            note_id = params["id"]
            if 0 <= note_id < len(NOTES):
                NOTES[note_id]["done"] = True
                return {"completed": NOTES[note_id]}
            return {"error": f"Note {note_id} not found"}

        elif action == "remove":
            note_id = params["id"]
            if 0 <= note_id < len(NOTES):
                removed = NOTES.pop(note_id)
                for i, note in enumerate(NOTES):
                    note["id"] = i
                return {"removed": removed}
            return {"error": f"Note {note_id} not found"}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="notes",
        description="Simple notes/todos. Actions: add (text), list, complete (id), remove (id).",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "remove"]},
                "text": {"type": "string", "description": "Note text (for add)"},
                "id": {"type": "integer", "description": "Note ID (for complete/remove)"}
            },
            "required": ["action"]
        },
        fn=fn
    )


# =============================================================================
# CLI
# =============================================================================

async def main_async():
    """Async REPL with background objective support."""
    print("Agent with Background Objectives")
    print("=" * 40)
    print("Chat continues while objectives work in background.")
    print("Type 'quit' to exit.\n")

    agent = Agent()

    # Start scheduler for recurring objectives
    scheduler_task = asyncio.create_task(agent.run_scheduler())

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            user_input = user_input.strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            response = await agent.run_async(user_input)
            print(f"Assistant: {response}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        scheduler_task.cancel()


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
