"""
Unified Message Abstraction Agent

The most elegant systems have a single unifying abstraction.
For an AI assistant, that abstraction is: everything is a message in a conversation.

- User input → message
- Assistant response → message
- Tool execution → message
- Memory retrieval → message
- Task state → message

This means the entire system reduces to: a loop that processes messages and decides what to do next.
"""

import json
from datetime import datetime
from typing import Callable

import anthropic


# =============================================================================
# Core Data Model: Message, Tool, Thread
# =============================================================================

class Tool:
    """
    A tool is a capability the agent can use.

    Memory? It's a tool that reads/writes to storage.
    Tasks? It's a tool that manages a list.
    Learning new tools? It's a tool that registers new tools.
    """

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
    The minimal agent: a loop that processes messages and decides what to do next.

    ┌─────────────────────────┐
    │         LOOP            │
    │   input → LLM → action  │
    │          ↑         │    │
    │          └─────────┘    │
    └───────────┬─────────────┘
                ↓
          ┌──────────┐
          │  Tools   │
          └──────────┘
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514", load_tools: bool = True):
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools: dict[str, Tool] = {}
        self.threads: dict[str, list] = {"default": []}

        # Bootstrap with core tools
        self.register(memory_tool)
        self.register(task_tool)
        self.register(register_tool_tool)

        # Load tools from tools/ folder
        if load_tools:
            self._load_external_tools()

    def register(self, tool: Tool):
        """Register a new tool with the agent."""
        self.tools[tool.name] = tool

    def _load_external_tools(self):
        """Load tools from the tools/ folder."""
        try:
            from tools import get_all_tools
            for tool in get_all_tools(Tool):
                self.register(tool)
        except ImportError:
            pass  # tools/ folder not present or not configured

    def run(self, user_input: str, thread_id: str = "default") -> str:
        """
        Process user input and return a response.

        This is the entire control flow: a single loop.
        """
        thread = self.threads.setdefault(thread_id, [])
        thread.append({"role": "user", "content": user_input})

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=8096,
                system=self._system_prompt(),
                tools=self._tool_schemas(),
                messages=thread
            )

            # Collect assistant message
            thread.append({"role": "assistant", "content": response.content})

            # If no tool use, we're done
            if response.stop_reason == "end_turn":
                return self._extract_text(response)

            # Execute tools and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = self.tools[block.name].execute(block.input, self)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })

            # Tool results become the next "user" message (API convention)
            thread.append({"role": "user", "content": tool_results})

    def _system_prompt(self) -> str:
        return """You are a helpful assistant with access to powerful tools.

Core capabilities:
- Memory: store important facts, retrieve when relevant
- Tasks: create, update, complete tasks as requested
- Register new tools: extend your capabilities at runtime

Web & Email (if configured):
- web_search: Quick DuckDuckGo searches (no API key needed)
- browse: Full browser automation - can fill forms, click, extract data
- get_agent_email, send_email, check_inbox: Email via AgentMail

Secrets management:
- get_secret, store_secret: Secure API key storage
- request_api_key: Ask user for missing keys

You can autonomously obtain API keys by browsing to service signup pages,
using your email for verification, and extracting keys from dashboards."""

    def _tool_schemas(self) -> list:
        return [t.schema for t in self.tools.values()]

    def _extract_text(self, response) -> str:
        return "".join(b.text for b in response.content if hasattr(b, "text"))

    def get_thread(self, thread_id: str = "default") -> list:
        """Get the message history for a thread."""
        return self.threads.get(thread_id, [])

    def clear_thread(self, thread_id: str = "default"):
        """Clear a thread's message history."""
        self.threads[thread_id] = []


# =============================================================================
# Core Tools: Memory, Tasks, Register Tool
# =============================================================================

# Memory storage (append-only log)
MEMORIES = []

def memory_fn(params: dict, agent: Agent) -> dict:
    """
    Memory: append-only log + search.

    The elegance is in not over-engineering this:
    - Append-only log: every memory is immutable, timestamped
    - Simple search: keyword match (swap for embeddings in production)
    - LLM does the organizing: give it raw memories and let it synthesize
    """
    action = params["action"]

    if action == "store":
        memory = {
            "content": params["content"],
            "timestamp": datetime.now().isoformat()
        }
        MEMORIES.append(memory)
        return {"stored": True, "memory": memory}

    elif action == "search":
        query = params.get("query", "").lower()
        if query:
            # Simple keyword match; swap for embeddings in production
            matches = [m for m in MEMORIES if query in m["content"].lower()]
        else:
            matches = MEMORIES
        return {"memories": matches[-10:]}  # Return recent matches

    elif action == "list":
        return {"memories": MEMORIES[-20:]}  # Return recent memories

    return {"error": f"Unknown action: {action}"}


memory_tool = Tool(
    name="memory",
    description="Store or search memories. Use 'store' to remember important facts, 'search' to recall by keyword, 'list' to see recent memories.",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["store", "search", "list"],
                "description": "The action to perform"
            },
            "content": {
                "type": "string",
                "description": "The content to store (required for 'store' action)"
            },
            "query": {
                "type": "string",
                "description": "The search query (required for 'search' action)"
            }
        },
        "required": ["action"]
    },
    fn=memory_fn
)


# Task storage
TASKS = []

def task_fn(params: dict, agent: Agent) -> dict:
    """
    Tasks: simple CRUD on a list.

    Tasks are just messages with a task-management tool.
    """
    action = params["action"]

    if action == "add":
        task = {
            "id": len(TASKS),
            "text": params["text"],
            "done": False,
            "created": datetime.now().isoformat()
        }
        TASKS.append(task)
        return {"added": task}

    elif action == "list":
        return {"tasks": TASKS}

    elif action == "complete":
        task_id = params["id"]
        if 0 <= task_id < len(TASKS):
            TASKS[task_id]["done"] = True
            TASKS[task_id]["completed"] = datetime.now().isoformat()
            return {"completed": TASKS[task_id]}
        return {"error": f"Task {task_id} not found"}

    elif action == "remove":
        task_id = params["id"]
        if 0 <= task_id < len(TASKS):
            removed = TASKS.pop(task_id)
            # Re-index remaining tasks
            for i, task in enumerate(TASKS):
                task["id"] = i
            return {"removed": removed}
        return {"error": f"Task {task_id} not found"}

    return {"error": f"Unknown action: {action}"}


task_tool = Tool(
    name="tasks",
    description="Manage tasks: add (with 'text'), list, complete (with 'id'), or remove (with 'id').",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "complete", "remove"],
                "description": "The action to perform"
            },
            "text": {
                "type": "string",
                "description": "The task text (required for 'add' action)"
            },
            "id": {
                "type": "integer",
                "description": "The task ID (required for 'complete' and 'remove' actions)"
            }
        },
        "required": ["action"]
    },
    fn=task_fn
)


def register_tool_fn(params: dict, agent: Agent) -> dict:
    """
    Meta-tool: register new tools at runtime.

    Learning new tools? It's a tool that registers new tools.

    WARNING: In production, this needs proper sandboxing.
    Here we demonstrate the pattern.
    """
    code = params["code"]
    tool_var_name = params["tool_var_name"]

    # Create a namespace for the new tool
    namespace = {
        "Tool": Tool,
        "datetime": datetime,
        "json": json,
    }

    try:
        exec(code, namespace)
        new_tool = namespace[tool_var_name]
        agent.register(new_tool)
        return {"registered": new_tool.name, "description": new_tool.schema["description"]}
    except Exception as e:
        return {"error": str(e)}


register_tool_tool = Tool(
    name="register_tool",
    description="Register a new tool by providing Python code that defines a Tool instance. The code should create a Tool() and assign it to a variable.",
    parameters={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code defining the tool function and Tool instance"
            },
            "tool_var_name": {
                "type": "string",
                "description": "Variable name of the Tool instance in the code"
            }
        },
        "required": ["code", "tool_var_name"]
    },
    fn=register_tool_fn
)


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Simple REPL for interacting with the agent."""
    print("Unified Message Abstraction Agent")
    print("=" * 40)
    print("Everything is a message. Type 'quit' to exit.\n")

    agent = Agent()

    while True:
        try:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            response = agent.run(user_input)
            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
