# Unified Message Abstraction Agent

> The most elegant systems have a single unifying abstraction.
> For an AI assistant, that abstraction is: **everything is a message in a conversation.**

## Core Philosophy

```
User input       → message
Assistant response → message
Tool execution   → message
Memory retrieval → message
Task state       → message
```

This means the entire system reduces to: **a loop that processes messages and decides what to do next.**

## Architecture

### The Minimal Loop

```
┌─────────────────────────────────────────────┐
│                   LOOP                       │
│                                             │
│   input → LLM → action → execute → output   │
│              ↑                 │             │
│              └─────────────────┘             │
└─────────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    ┌───────┐   ┌────────┐   ┌────────┐
    │ Tools │   │ Memory │   │ Tasks  │
    └───────┘   └────────┘   └────────┘
```

**The insight:** Tools, Memory, and Tasks are all just Tools.
- Memory is a tool that reads/writes to storage
- Tasks is a tool that manages a list
- This collapses to:

```
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
```

### The Data Model

Three entities. That's it.

```python
Message {
  id: string
  role: "user" | "assistant" | "tool"
  content: string
  metadata: object
  timestamp: datetime
}

Tool {
  name: string
  description: string
  parameters: json_schema
  execute: function
}

Thread {
  id: string
  messages: Message[]
}
```

**Memory?** It's messages with a retrieval tool.
**Tasks?** It's messages with a task-management tool.
**Learning new tools?** It's a tool that registers new tools.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd babyagi_3

# Install dependency
pip install anthropic

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

## Usage

### Interactive REPL

```bash
python agent.py
```

```
Unified Message Abstraction Agent
========================================
Everything is a message. Type 'quit' to exit.

You: Hello, remember that my favorite color is blue.
Assistant: I've stored that your favorite color is blue.

You: What's my favorite color?
Assistant: Your favorite color is blue.

You: Add a task: review the Q1 report
Assistant: I've added the task "review the Q1 report" to your list.

You: What are my tasks?
Assistant: You have 1 task:
- [0] review the Q1 report (pending)

You: Mark the first task as complete
Assistant: Done! Task 0 "review the Q1 report" is now complete.
```

### Programmatic Usage

```python
from agent import Agent

agent = Agent()

# Basic conversation
agent.run("Hello, remember that my favorite color is blue.")
agent.run("What's my favorite color?")

# Task management
agent.run("Add a task: review the Q1 report")
agent.run("What are my tasks?")
agent.run("Mark the first task as complete")

# Multiple threads (separate conversations)
agent.run("Remember: project deadline is Friday", thread_id="work")
agent.run("Remember: buy groceries", thread_id="personal")

# Teaching new tools
agent.run("""
Create a tool called 'calculator' that takes two numbers and an operation,
and returns the result.
""")
```

## Core Tools

### Memory Tool

Append-only log with search capability.

```python
# Store a memory
agent.run("Remember that the server IP is 192.168.1.100")

# Search memories
agent.run("What's the server IP?")

# List recent memories
agent.run("Show me my recent memories")
```

**Actions:**
- `store` - Save content to memory with timestamp
- `search` - Find memories by keyword
- `list` - Show recent memories

### Task Tool

Simple CRUD operations on a task list.

```python
# Add tasks
agent.run("Add a task: write documentation")
agent.run("Add a task: fix the login bug")

# List tasks
agent.run("What are my tasks?")

# Complete a task
agent.run("Complete task 0")

# Remove a task
agent.run("Remove task 1")
```

**Actions:**
- `add` - Create a new task
- `list` - Show all tasks
- `complete` - Mark a task as done
- `remove` - Delete a task

### Register Tool Tool

Meta-tool for runtime extensibility.

```python
agent.run("""
Create a tool that converts temperatures between Celsius and Fahrenheit.
It should take a value and a direction (c_to_f or f_to_c).
""")

# Now use the new tool
agent.run("Convert 100 Celsius to Fahrenheit")
```

## On "Organized Memory"

The elegance is in not over-engineering this. The approach:

1. **Append-only log** — every memory is immutable, timestamped
2. **Semantic search** — keyword match (swap for embeddings in production)
3. **Summarization on read** — the LLM itself summarizes relevant memories at query time

You don't need a complex knowledge graph. **The LLM is the organizer.** Give it raw memories and let it synthesize.

For long-term organization, periodically run a "reflection" pass:

```python
agent.run("Review your recent memories and create a summary of key facts about the user.")
```

## Extending the System

### Adding Custom Tools

```python
from agent import Agent, Tool

def my_tool_fn(params, agent):
    # Your logic here
    return {"result": "success"}

my_tool = Tool(
    name="my_tool",
    description="Description of what the tool does",
    parameters={
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "param2": {"type": "integer", "description": "Second parameter"}
        },
        "required": ["param1"]
    },
    fn=my_tool_fn
)

agent = Agent()
agent.register(my_tool)
```

### Using a Different Model

```python
agent = Agent(model="claude-opus-4-20250514")
```

### Persistent Storage

Swap the in-memory lists for a database:

```python
# In production, replace MEMORIES and TASKS with:
# - SQLite for simplicity
# - PostgreSQL for scale
# - Redis for speed
# - Vector DB for semantic search
```

## Design Principles

### What Makes This Elegant

| Principle | Implementation |
|-----------|----------------|
| **Single loop** | One control flow for everything |
| **Tools are data** | Same structure handles memory, tasks, extensibility |
| **LLM does the hard work** | Routing, synthesis, organization |
| **No frameworks** | Just Python and the API |
| **Extensible by design** | New capabilities = new tools |

### The Entire System

~200 lines of code. Everything else is tools.

```
agent.py
├── Tool class (15 lines)
├── Agent class (60 lines)
├── memory_tool (30 lines)
├── task_tool (35 lines)
├── register_tool_tool (25 lines)
└── CLI (20 lines)
```

## Advanced Patterns

### Multi-Thread Conversations

```python
# Work context
agent.run("I'm working on the authentication system", thread_id="work")
agent.run("What am I working on?", thread_id="work")

# Personal context (separate memory)
agent.run("I need to call mom", thread_id="personal")
```

### Reflection and Consolidation

```python
# Periodically consolidate memories
agent.run("""
Review all my memories and tasks. Create a summary of:
1. Key facts about me
2. Important ongoing projects
3. Patterns you've noticed
Store this summary as a new memory.
""")
```

### Tool Composition

The LLM naturally composes tools:

```python
agent.run("""
Add a task for each memory I've stored today.
""")
# Agent will: search memories → add task for each
```

## API Reference

### Agent

```python
class Agent:
    def __init__(self, model: str = "claude-sonnet-4-20250514")
    def register(self, tool: Tool) -> None
    def run(self, user_input: str, thread_id: str = "default") -> str
    def get_thread(self, thread_id: str = "default") -> list
    def clear_thread(self, thread_id: str = "default") -> None
```

### Tool

```python
class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,  # JSON Schema
        fn: Callable[[dict, Agent], dict]
    )
    def execute(self, params: dict, agent: Agent) -> dict
```

## Philosophy

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

This agent embodies that principle. It's not about what features you can add—it's about finding the minimal abstraction that makes features emerge naturally.

**Everything is a message. Everything is a tool. That's the whole system.**

## License

MIT
