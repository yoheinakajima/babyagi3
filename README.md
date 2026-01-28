# BabyAGI 3

> The most elegant systems have a single unifying abstraction.
> For an AI assistant, that abstraction is: **everything is a message in a conversation.**

## Core Philosophy

```
User input        → message
Assistant response → message
Tool execution    → message
Memory retrieval  → message
Task state        → message
Background work   → message
```

This means the entire system reduces to: **a loop that processes messages and decides what to do next.**

## What's New in v0.2.0

- **Background Objectives** — Async work that runs while chat continues
- **Recurring Tasks** — Schedule objectives hourly, daily, or custom intervals
- **e2b Sandbox** — Safe code execution for dynamically created tools
- **Tool Validation** — Prevents malformed tool registrations
- **External Tools** — Web search, browser automation, email, secrets management
- **API Server** — FastAPI server mode with full REST API

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
- Objectives is a tool that spawns background work
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

Four core entities:

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

Objective {
  id: string
  description: string
  status: "pending" | "running" | "completed" | "failed"
  thread_id: string
  recurring: string | None  # "hourly", "daily", "every N minutes"
}
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd babyagi_3

# Install dependencies (choose one)
pip install anthropic e2b-code-interpreter fastapi pydantic uvicorn
# Or using uv:
uv sync

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

### Optional Dependencies

```bash
# For email functionality
export AGENTMAIL_API_KEY="your-agentmail-key"

# For secure secrets storage
pip install keyring keyrings.cryptfile

# For web browsing (auto-installed when needed)
pip install browser-use langchain-anthropic
```

## Usage

### Interactive CLI

```bash
python agent.py
# or
python main.py
```

```
BabyAGI Agent
========================================
Everything is a message. Type 'quit' to exit.

You: Hello, remember that my favorite color is blue.
Assistant: I've stored that your favorite color is blue.

You: What's my favorite color?
Assistant: Your favorite color is blue.

You: Add a task: review the Q1 report
Assistant: I've added the task "review the Q1 report" to your list.

You: Research the latest AI papers and summarize them for me
Assistant: I'll start that as a background objective. You can continue chatting while I work on it.
```

### API Server

```bash
# Start on default port 8000
python main.py serve

# Start on custom port
python main.py serve 8080

# Or directly with uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000
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

# Background objectives
agent.run("Research competitor products and create a report")
# Chat continues while objective runs in background

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
- `store` — Save content to memory with timestamp
- `search` — Find memories by keyword
- `list` — Show recent memories

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
- `add` — Create a new task
- `list` — Show all tasks
- `complete` — Mark a task as done
- `remove` — Delete a task

### Objectives Tool

Background work that runs asynchronously while chat continues.

```python
# Start a one-time objective
agent.run("Research the latest Claude API features and summarize them")

# Start a recurring objective
agent.run("Every day, check my inbox and summarize important emails")

# Check objective status
agent.run("What objectives are running?")
```

**Actions:**
- `create` — Start a new background objective
- `list` — Show all objectives and their status
- `check` — Get status of a specific objective

**Scheduling options:**
- `"hourly"` — Run every hour
- `"daily"` — Run every day
- `"every N minutes"` — Run at custom intervals

### Notes Tool

Quick note-taking without search overhead.

```python
agent.run("Note: API rate limit is 100 requests per minute")
agent.run("Show my notes")
```

### Register Tool Tool

Meta-tool for runtime extensibility. Dynamically created tools run in a secure e2b sandbox.

```python
agent.run("""
Create a tool that converts temperatures between Celsius and Fahrenheit.
It should take a value and a direction (c_to_f or f_to_c).
""")

# Now use the new tool
agent.run("Convert 100 Celsius to Fahrenheit")
```

Tools that require external packages are automatically sandboxed for safety.

## External Tools

BabyAGI includes optional external tools for web, email, and secrets management.

### Web Tools

```python
# Search the web (uses DuckDuckGo, no API key needed)
agent.run("Search for 'latest machine learning papers 2024'")

# Browse a webpage with AI
agent.run("Go to example.com and extract the main content")

# Fetch and parse a URL
agent.run("Fetch https://api.example.com/status and show me the response")
```

### Email Tools

Requires `AGENTMAIL_API_KEY` environment variable.

```python
# Get agent's email address
agent.run("What's my email address?")

# Send an email
agent.run("Send an email to user@example.com with the project summary")

# Check inbox
agent.run("Check my inbox for new messages")

# Wait for verification emails
agent.run("Sign up for service X and wait for the verification email")
```

### Secrets Management

Securely store and retrieve API keys.

```python
# Store a secret
agent.run("Store my OpenAI API key: sk-...")

# List stored secrets (values are masked)
agent.run("What API keys do I have stored?")

# Secrets are automatically retrieved when needed
agent.run("Use my GitHub token to check my repositories")
```

## API Server Endpoints

When running in server mode (`python main.py serve`), the following endpoints are available:

### POST /message

Send a message to the agent.

```bash
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello!", "thread_id": "main"}'
```

### GET /objectives

List all background objectives.

```bash
curl http://localhost:8000/objectives
```

### GET /objectives/{id}

Get status of a specific objective.

```bash
curl http://localhost:8000/objectives/obj-123
```

### GET /threads/{id}

Retrieve message history for a thread.

```bash
curl http://localhost:8000/threads/main
```

### DELETE /threads/{id}

Clear a conversation thread.

```bash
curl -X DELETE http://localhost:8000/threads/main
```

### GET /health

Health check endpoint.

```bash
curl http://localhost:8000/health
```

## Extending the System

### Adding Custom Tools (Class-based)

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

### Adding Custom Tools (Decorator-based)

```python
from tools import tool

@tool
def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle.

    Args:
        length: The length of the rectangle
        width: The width of the rectangle

    Returns:
        The area of the rectangle
    """
    return length * width

# The decorator automatically generates the JSON schema from type hints
agent.register(calculate_area)
```

### Using a Different Model

```python
agent = Agent(model="claude-opus-4-20250514")
```

### Disabling External Tools

```python
agent = Agent(load_tools=False)  # Only core tools loaded
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
| **Tools are data** | Same structure handles memory, tasks, objectives |
| **LLM does the hard work** | Routing, synthesis, organization |
| **No frameworks** | Just Python and the API |
| **Extensible by design** | New capabilities = new tools |
| **Safe extensibility** | Dynamic tools run in sandboxed environment |

### The Entire System

~200 lines of core code. Everything else is tools.

```
agent.py
├── Tool class (dataclass)
├── Agent class (main loop)
├── Objective class (background work)
├── Core tools (memory, tasks, objectives, notes, register)
└── CLI interface

tools/
├── __init__.py (decorator framework)
├── sandbox.py (e2b code execution)
├── web.py (search, browse, fetch)
├── email.py (AgentMail integration)
└── secrets.py (secure key storage)

server.py (FastAPI endpoints)
main.py (entry point)
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

### Background Objectives

```python
# Long-running research task
agent.run("""
Research the top 10 AI frameworks released this year.
Compare their features, performance, and community adoption.
Create a detailed report.
""")

# The agent continues to respond while the objective runs
agent.run("While that's running, can you help me with something else?")
```

### Recurring Tasks

```python
# Daily summary
agent.run("Every day at 9am, summarize my unread emails")

# Hourly check
agent.run("Every hour, check the server status and alert me if it's down")

# Custom interval
agent.run("Every 30 minutes, fetch the latest stock prices")
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

### Auto-Signup Workflow

```python
# Automatically sign up for a service
agent.run("""
Sign up for an account at example.com using my agent email.
Wait for the verification email and complete the signup.
""")
```

## On "Organized Memory"

The elegance is in not over-engineering this. The approach:

1. **Append-only log** — every memory is immutable, timestamped
2. **Semantic search** — keyword match (swap for embeddings in production)
3. **Summarization on read** — the LLM itself summarizes relevant memories at query time

You don't need a complex knowledge graph. **The LLM is the organizer.** Give it raw memories and let it synthesize.

## API Reference

### Agent

```python
class Agent:
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        load_tools: bool = True
    )
    def register(self, tool: Tool | ToolLike) -> None
    def run(self, user_input: str, thread_id: str = "default") -> str
    def get_thread(self, thread_id: str = "default") -> list
    def clear_thread(self, thread_id: str = "default") -> None
```

### Tool

```python
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    fn: Callable[[dict, Agent], dict]

    def execute(self, params: dict, agent: Agent) -> dict
```

### Objective

```python
@dataclass
class Objective:
    id: str
    description: str
    status: str  # "pending", "running", "completed", "failed"
    thread_id: str
    result: str | None
    recurring: str | None  # "hourly", "daily", "every N minutes"
    last_run: datetime | None
```

### Tool Decorator

```python
from tools import tool

@tool
def my_function(param1: str, param2: int = 0) -> dict:
    """Description of the function.

    Args:
        param1: Description of param1
        param2: Description of param2 (optional)
    """
    return {"result": "value"}

# Or with explicit name/description
@tool(name="custom_name", description="Custom description")
def another_function(x: float) -> float:
    return x * 2
```

## Requirements

- Python >= 3.12
- anthropic >= 0.76.0
- e2b-code-interpreter >= 1.0.0
- fastapi >= 0.115.0
- pydantic >= 2.12.5
- uvicorn >= 0.32.0

## Philosophy

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

This agent embodies that principle. It's not about what features you can add—it's about finding the minimal abstraction that makes features emerge naturally.

**Everything is a message. Everything is a tool. That's the whole system.**

## License

MIT
