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
Channel input     → message
```

This means the entire system reduces to: **a loop that processes messages and decides what to do next.**

## What's New in v0.3.0

- **Graph-Based Memory** — SQLite-backed persistent memory with entity extraction, relationship tracking, and semantic search
- **Smart Tool Selection** — Intelligent context-aware tool filtering to manage growing tool inventories
- **Tool Self-Improvement** — Dynamically created tools persist across restarts with usage statistics and health tracking
- **Objective Concurrency Control** — Max 5 concurrent objectives with semaphore-based limiting
- **Priority Queue** — Objectives support priority 1-10 (lower = higher); higher priority runs first
- **Retry with Backoff** — Failed objectives retry automatically with exponential backoff (2s, 4s, 8s)
- **Budget Controls** — Set `budget_usd` or `token_limit` per objective to cap costs
- **True Cancellation** — Cancel signals propagate to running objectives immediately
- **Metrics & Cost Tracking** — Automatic tracking of API usage, costs, and tool performance
- **Thread Repair** — Automatic repair of corrupted message threads from failed tool executions

### Previous (v0.2.0)

- **Skills System** — Learn new behaviors from SKILL.md files with safety scanning
- **Composio Integration** — Connect to 250+ apps (Slack, GitHub, Notion) via OAuth
- **Multi-Channel Architecture** — Receive messages from CLI, email, voice, and more
- **Full-Featured Scheduler** — Cron expressions, intervals, one-time tasks with persistence
- **Styled CLI Output** — Color-coded messages (user/agent/system) with verbose modes
- **Event System** — UI-ready event emitter for tool calls, objectives, and tasks
- **Unified Communication** — Send messages via any channel with `send_message` tool
- **Owner vs External** — Context-aware responses based on message source
- **YAML Configuration** — Easy channel setup with environment variable substitution
- **Tool Health System** — Automatic detection of available tools and missing dependencies
- **Extensible Channels** — Add new channels by implementing simple listener/sender patterns
- **Tool Persistence** — Dynamically created tools survive restarts with health tracking
- **Credential Storage** — Secure storage for accounts and payment methods with keyring integration

### Previous (v0.2.0)

- **Background Objectives** — Async work that runs while chat continues
- **e2b Sandbox** — Safe code execution for dynamically created tools
- **External Tools** — Web search, browser automation, email, secrets management
- **API Server** — FastAPI server mode with full REST API

## Architecture

### Multi-Channel Design

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Listeners                              │
│         (Diverse - each channel is different)                    │
│                                                                  │
│   CLI              Email             Voice            Future...  │
│   - Terminal       - Poll inbox      - Wake word      - SMS      │
│   - Always on      - Auto-reply      - STT/TTS        - WhatsApp │
│                                                                  │
│         All call: agent.run_async(content, context={...})       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AGENT                                   │
│                                                                  │
│   Threads     Memory     Objectives     Scheduler     Tools      │
│      │          │            │              │           │        │
│      └──────────┴────────────┴──────────────┴───────────┘        │
│                              │                                   │
│               ┌──────────────┴──────────────┐                    │
│               │     send_message tool       │                    │
│               └──────────────┬──────────────┘                    │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: Senders                               │
│            (Unified - one tool, many channels)                   │
│                                                                  │
│   send_message(channel="email", to="owner", content="Done!")    │
│   send_message(channel="sms", to="+1...", content="Alert!")     │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** Input and output are separate concerns.
- **Listeners** are diverse (polling, webhooks, streaming) — no forced interface
- **Senders** are unified via `send_message` tool — agent uses same tool for any channel

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
    │ Tools │   │ Memory │   │Schedule│
    └───────┘   └────────┘   └────────┘
```

**The insight:** Tools, Memory, Scheduling, and Notes are all just Tools.
- Memory is a tool that reads/writes to storage
- Notes is a tool that manages a simple todo list
- Objectives is a tool that spawns background work
- Schedule is a tool that manages time-based automation

### The Data Model

Core entities:

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
  packages: list[str]  # Required packages
  env: list[str]       # Required env vars
}

ToolDefinition {
  id: string
  name: string
  description: string
  tool_type: "executable" | "skill" | "composio"

  # For executable tools
  source_code: string | None
  parameters: json_schema
  packages: list[str]

  # For skills
  skill_content: string | None  # SKILL.md markdown

  # For composio tools
  composio_app: string | None   # "SLACK", "GITHUB"
  composio_action: string | None

  # Dependencies
  depends_on: list[str]  # Other tools/skills

  # Stats
  usage_count: int
  success_rate: float
}

Thread {
  id: string
  messages: Message[]
}

Objective {
  id: string
  goal: string
  status: "pending" | "running" | "completed" | "failed" | "cancelled"
  thread_id: string
  schedule: string | None
  result: string | None
  error: string | None
  # Priority and resource controls
  priority: int  # 1-10, lower = higher priority (default 5)
  retry_count: int  # Current retry attempt
  max_retries: int  # Max retry attempts (default 3)
  last_error: string | None  # Error from last failed attempt
  # Budget tracking
  budget_usd: float | None  # Max cost allowed
  spent_usd: float  # Total cost so far
  token_limit: int | None  # Max tokens allowed
  tokens_used: int  # Total tokens used
}

Schedule {
  kind: "at" | "every" | "cron"
  at: string | None      # ISO timestamp for one-time
  every: string | None   # Interval: "5m", "2h", "1d"
  cron: string | None    # Cron expression
  tz: string | None      # Timezone
}

ScheduledTask {
  id: string
  name: string
  goal: string
  schedule: Schedule
  enabled: bool
  next_run_at: string
  last_run_at: string
  run_count: int
}

ToolDefinition {
  id: string
  name: string
  description: string
  source_code: string | None     # Python code for dynamic tools
  parameters: json_schema
  packages: list[str]            # Required packages
  env: list[str]                 # Required env vars
  category: string               # "core", "builtin", "custom"
  is_enabled: bool
  is_dynamic: bool               # False for built-in tools

  # Execution tracking
  usage_count: int
  success_count: int
  error_count: int
  avg_duration_ms: float
  last_used_at: datetime | None
  last_error: string | None
}

Credential {
  id: string
  credential_type: string        # "account", "credit_card", "api_key"
  service: string                # "github.com", "stripe.com"

  # Account fields
  username: string | None
  email: string | None
  password_ref: string | None    # Reference to keyring

  # Credit card fields
  card_last_four: string | None
  card_type: string | None       # "visa", "mastercard", etc.
  card_expiry: string | None
  card_ref: string | None        # Reference to keyring
  billing_name: string | None
  billing_address: string | None

  # Metadata
  notes: string | None
  last_used_at: datetime | None
}
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd babyagi_3

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install anthropic e2b-code-interpreter fastapi pydantic uvicorn \
    duckduckgo-search httpx beautifulsoup4 agentmail keyring \
    keyrings-cryptfile ddgs croniter pyyaml

# Set your API key
export ANTHROPIC_API_KEY="your-api-key"
```

### Optional Dependencies

```bash
# For email functionality
export AGENTMAIL_API_KEY="your-agentmail-key"

# For browser automation
export BROWSER_USE_API_KEY="your-browser-use-key"

# For dynamic tool sandboxing
export E2B_API_KEY="your-e2b-key"

# For Composio integrations (250+ apps)
pip install composio-core
composio login  # or set COMPOSIO_API_KEY

# For voice channel (install separately)
pip install sounddevice numpy openai-whisper pyttsx3
```

## Usage

### Interactive CLI

```bash
python main.py          # Run CLI only (default)
python main.py cli      # Explicit CLI mode
```

```
BabyAGI v0.3.0
========================================

You: Hello, remember that my favorite color is blue.
Assistant: I've stored that your favorite color is blue.

You: What's my favorite color?
Assistant: Your favorite color is blue.
```

### Styled Output & Verbose Mode

The CLI features color-coded output for better readability:
- **User messages**: Green
- **Agent responses**: Cyan
- **System messages**: Blue
- **Verbose/debug**: Yellow (light) / Gray (deep)

**Verbose Levels:**

| Level | Shows |
|-------|-------|
| `off` | Only user/agent messages (default) |
| `light` | Key operations: tool names, task starts, objective status |
| `deep` | Everything: tool inputs/outputs, timing, full details |

**Enable verbose output:**

```bash
# Environment variable
BABYAGI_VERBOSE=light python main.py

# Or in config.yaml
verbose: light  # off, light, or deep
```

**Runtime toggle:**

```
You: /verbose light
Verbose output: light (key operations)

You: What's in my memory?
  [tool] memory
  [tool] memory done (12ms)
Assistant: You have 3 memories stored...

You: /verbose deep
Verbose output: deep (everything)

You: Store that I like coffee
  [tool] memory
    input: {action="store", content="User likes coffee"}
  [tool] memory done (8ms)
    result: {stored=true}
Assistant: I've stored that you like coffee.

You: /verbose off
Verbose output: off
```

### Multi-Channel Mode

Run all enabled channels concurrently:

```bash
python main.py channels
```

```
BabyAGI v0.3.0 - Multi-Channel Mode
========================================

Active channels: cli, email
Press Ctrl+C to stop

You: Research X and email me the results when done
Assistant: I'll start that as a background objective and email you when complete.
```

The agent can now:
- Receive your CLI input
- Monitor your email inbox
- Respond to emails automatically
- Send you messages on any channel

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

# Notes management
agent.run("Add a note: review the Q1 report")
agent.run("What are my notes?")
agent.run("Mark note 0 as complete")

# Multiple threads (separate conversations)
agent.run("Remember: project deadline is Friday", thread_id="work")
agent.run("Remember: buy groceries", thread_id="personal")

# Background objectives
agent.run("Research competitor products and create a report")
# Chat continues while objective runs in background

# Scheduling
agent.run("Remind me in 30 minutes to take a break")
agent.run("Check server status every 5 minutes")
agent.run("Send me a daily summary at 9am")

# Teaching new tools
agent.run("""
Create a tool called 'calculator' that takes two numbers and an operation,
and returns the result.
""")
```

## Core Tools

### Memory Tool

Persistent memory with semantic search, entity extraction, and knowledge graph capabilities.

```python
# Store a memory
agent.run("Remember that the server IP is 192.168.1.100")

# Search memories
agent.run("What's the server IP?")

# List recent memories
agent.run("Show me my recent memories")

# Find entities in the knowledge graph
agent.run("What do I know about John Smith?")

# Get a summary of what you know about a topic
agent.run("Summarize what you know about our cloud infrastructure")
```

**Actions:**
- `store` — Save content to memory with timestamp
- `search` — Semantic search over memories (returns last 10 matches)
- `list` — Show recent memories (last 20)
- `find_entity` — Search entities in the knowledge graph (type: person/org/concept)
- `find_relationship` — Find connections between entities
- `get_summary` — Get pre-computed summary by key (e.g., 'root', 'entity:uuid')
- `get_context` — Get assembled context for the current conversation
- `deep_search` — Thorough agentic search for complex queries
- `list_tools` — List all registered tools with usage stats and health status
- `tool_stats` — Get aggregate statistics about tool usage
- `problematic_tools` — Find tools with high error rates
- `get_tool` — Get detailed info about a specific tool (tool_name)

**Convenience Methods (for dynamic tools):**

Dynamic tools can access memory directly via agent methods:

```python
# In a dynamic tool function
def my_tool(params: dict, agent) -> dict:
    # Search memory
    result = agent.memory_recall("server IP")
    # Returns: {"memories": [{"content": "...", "timestamp": "...", "channel": "..."}]}

    # Store to memory
    result = agent.memory_store("New fact discovered")
    # Returns: {"stored": True, "event_id": "..."}

    return {"success": True}
```

### Notes Tool

Simple todo list for passive tracking (unlike active objectives).

```python
# Add notes
agent.run("Add a note: write documentation")
agent.run("Add a note: fix the login bug")

# List notes
agent.run("What are my notes?")

# Complete a note
agent.run("Complete note 0")

# Remove a note
agent.run("Remove note 1")
```

**Actions:**
- `add` — Create a new note (text)
- `list` — Show all notes
- `complete` — Mark a note as done (id)
- `remove` — Delete a note (id)

### Objectives Tool

Background work that runs asynchronously while chat continues. Features priority queuing, automatic retry with exponential backoff, budget controls, and true cancellation.

```python
# Start a one-time objective
agent.run("Research the latest Claude API features and summarize them")

# Start with priority (1=highest, 10=lowest)
agent.run("Urgent: Research competitor announcement immediately")  # priority=1

# Start with budget limit
agent.run("Research market trends, but limit to $0.50")  # budget_usd=0.50

# Start with token limit
agent.run("Generate a summary, max 10000 tokens")  # token_limit=10000

# Check objective status (includes cost tracking)
agent.run("What objectives are running?")

# Cancel an objective (immediately signals running objective to stop)
agent.run("Cancel objective abc123")
```

**Actions:**
- `spawn` — Start a new background objective with options:
  - `goal` — What to accomplish (required)
  - `priority` — 1-10, lower = higher priority (default: 5)
  - `schedule` — Recurring schedule (optional)
  - `budget_usd` — Maximum cost allowed (stops if exceeded)
  - `token_limit` — Maximum tokens allowed (stops if exceeded)
  - `max_retries` — Retry attempts on failure (default: 3)
- `list` — See all objectives sorted by priority, with status and cost tracking
- `check` — Get full details including retry count, spent_usd, tokens_used
- `cancel` — Stop an objective (sends cancellation signal to running objectives)

**Concurrency & Retry:**
- Max 5 objectives run simultaneously; others queue by priority
- Failed objectives retry automatically with exponential backoff (2s, 4s, 8s)
- Budget exceeded or max retries reached → objective fails permanently

### Schedule Tool

Full-featured time-based task automation with persistence.

```python
# One-time reminders
agent.run("Remind me in 30 minutes to call mom")
agent.run("Schedule a task at 3pm to send the report")

# Recurring tasks
agent.run("Check my email every hour")
agent.run("Send me a daily summary at 9am")
agent.run("Run health checks every 5 minutes")

# Cron expressions
agent.run("Run backups at midnight on weekdays")
```

**Actions:**
- `add` — Create scheduled task (name, goal, spec)
- `list` — View all scheduled tasks
- `get` — Get task details (id)
- `update` — Modify task (id, name?, goal?, spec?, enabled?)
- `remove` — Delete task (id)
- `run` — Execute task now (id, force=True)
- `history` — View execution history (id, limit=10)

**Schedule Specification (string shortcuts):**
- `"in 5m"` or `"in 2h"` — one-time, relative
- `"every 5m"` or `"every 1h"` — recurring interval
- `"daily at 9:00"` — daily at specific time
- `"weekdays at 9am"` — weekdays only
- `"hourly"` or `"daily"` — simple patterns
- `"0 9 * * 1-5"` — raw cron expression

**Schedule Specification (full control):**
```json
{"kind": "at", "at": "2024-01-15T09:00:00", "tz": "America/New_York"}
{"kind": "every", "every": "5m"}
{"kind": "cron", "cron": "0 9 * * 1-5", "tz": "America/New_York"}
```

**Persistence:** Tasks are stored in `~/.babyagi/scheduler/tasks.json` and survive restarts.

### Register Tool Tool

Meta-tool for runtime extensibility. Dynamically created tools run in a secure e2b sandbox when they require external packages.

```python
agent.run("""
Create a tool that converts temperatures between Celsius and Fahrenheit.
It should take a value and a direction (c_to_f or f_to_c).
""")

# Now use the new tool
agent.run("Convert 100 Celsius to Fahrenheit")
```

Tools that require external packages are automatically detected and sandboxed for safety.

#### Tool Persistence

Dynamically created tools are **automatically persisted** to the SQLite database and survive agent restarts:

- **Storage**: Tools are saved in the `tool_definitions` table with full source code
- **Auto-reload**: On startup, all persisted tools are automatically re-registered
- **Smart execution**: Pure Python tools run locally (fast), tools with external packages use e2b sandbox (safe)

```python
# Create a tool - it's persisted automatically
agent.run("Create a calculator tool that adds two numbers")

# Restart the agent...
agent = Agent()

# Tool is still available!
agent.run("Add 5 and 3")  # Works immediately
```

#### Tool Health Tracking

Every tool execution is tracked with detailed metrics:

```python
ToolDefinition {
    name: str                    # Tool name
    description: str             # What it does
    source_code: str             # Python code (for dynamic tools)
    parameters: dict             # JSON schema

    # Execution statistics
    usage_count: int             # Total executions
    success_count: int           # Successful runs
    error_count: int             # Failed runs
    avg_duration_ms: float       # Average execution time
    last_used_at: datetime       # Last execution timestamp
    last_error: str              # Most recent error message

    # Health indicators
    success_rate: float          # Computed: success_count / usage_count
    is_healthy: bool             # True if error rate < 50%
}
```

**Monitoring tool health:**

```python
# The agent can check tool health
agent.run("Which of my tools have been failing?")
agent.run("Show me usage statistics for my tools")

# Programmatic access
from memory import MemoryStore
store = MemoryStore()
unhealthy = store.get_unhealthy_tools()  # Tools with high error rates
problematic = store.get_problematic_tools()  # Tools needing attention
```

### Skills Tool

Skills are behavioral instructions stored as SKILL.md files. Unlike executable tools that run code, skills provide guidance that the AI follows.

```python
# Acquire a skill from a URL or file
agent.run("Learn the code review skill from https://example.com/skills/code-review/SKILL.md")

# Acquire with auto-enable if safe
agent.run("Acquire skill from ~/my-skills/email-triage.md and enable it if safe")

# Check skill readiness
agent.run("Check if the email-triage skill has all required tools")

# Activate a skill to get instructions
agent.run("Activate the code review skill")
```

**SKILL.md Format:**

```markdown
---
name: code-review
description: Guidelines for thorough code reviews
---

# Code Review Skill

When reviewing code, follow these steps:
1. Check for security vulnerabilities
2. Verify error handling
3. Look for performance issues
4. Ensure tests are adequate
```

**Safety Scanning:**
- Skills are scanned for dangerous patterns before activation
- Score >= 80: Safe, can auto-enable
- Score < 80: Requires manual review
- Critical patterns (eval, exec, prompt injection): Auto-rejected

### Composio Integration

Composio provides 250+ OAuth app integrations (Slack, GitHub, Notion, Gmail, etc.). Once connected, Composio actions become tools in your toolkit.

```python
# List available apps
agent.run("Show me what Composio apps are available")

# Connect to an app (starts OAuth flow)
agent.run("Connect to Slack via Composio")

# Enable app tools
agent.run("Enable all GitHub actions from Composio")

# Enable specific actions only
agent.run("Enable only SLACK_SEND_MESSAGE from Composio")

# Check connection status
agent.run("What Composio integrations do I have enabled?")

# Use Composio tools (after enabling)
agent.run("Send a message to #general on Slack saying 'Hello from BabyAGI!'")
```

**Setup:**

```bash
# Install Composio
pip install composio-core

# Login to Composio
composio login

# Set API key (alternative)
export COMPOSIO_API_KEY="your-key"
```

**Composio Actions:**
- `list_apps` — Show all available apps
- `list_actions` — Show actions for a specific app
- `connect` — Start OAuth flow to authorize an app
- `enable` — Register Composio actions as tools
- `disable` — Remove tools (keeps Composio auth)
- `status` — Show what's connected and enabled

### Workflow Tools

Workflows combine multiple tools and skills into reusable automation.

```python
# Create a workflow
agent.run("""
Create a workflow called 'daily_standup' that:
1. Activates the standup format skill
2. Gathers yesterday's completed tasks
3. Posts the summary to Slack #standup
""")

# Workflows can declare dependencies
agent.run("""
Create workflow 'pr_review' depending on:
- skill_code_review
- github_create_review
""")
```

**Workflow Features:**
- Dependencies are verified before creation
- Generated as executable Python tools
- Can combine skills with executable tools
- Persisted for reuse across sessions

### Send Message Tool

Send messages across any configured channel:

```python
# The agent can use send_message to communicate on any channel
agent.run("Send me an email when the research is done")
agent.run("Text me if there's an urgent issue")

# Cross-channel examples:
# - Receive email → respond via email AND text you
# - Background task completes → email results to you
# - External person emails → consult with you via CLI
```

## External Tools

BabyAGI includes optional external tools for web, email, and secrets management.

### Web Tools

```python
# Search the web (uses DuckDuckGo, no API key needed)
agent.run("Search for 'latest machine learning papers 2024'")

# Browse a webpage with AI (requires BROWSER_USE_API_KEY)
agent.run("Go to example.com and extract the main content")

# Fetch and parse a URL
agent.run("Fetch https://api.example.com/status and show me the response")

# Auto-signup workflow (requires BROWSER_USE_API_KEY and AGENTMAIL_API_KEY)
agent.run("Sign up for service X and get the API key")
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

BabyAGI provides a **two-layer security model** for storing sensitive data:

#### Layer 1: API Keys and Secrets

Securely store and retrieve API keys using system keyring with automatic fallback to encrypted file storage.

```python
# Store a secret
agent.run("Store my OpenAI API key: sk-...")

# List stored secrets (values are masked)
agent.run("What API keys do I have stored?")

# Secrets are automatically retrieved when needed
agent.run("Use my GitHub token to check my repositories")
```

**Storage backends (checked in order):**
1. **Environment variables** — checked first for immediate access
2. **System keyring** — OS-level encryption (macOS Keychain, Windows Credential Locker, Linux Secret Service)
3. **Encrypted file fallback** — uses `keyrings.cryptfile` if system keyring unavailable

**Security features:**
- Values are always masked in output (e.g., `sk-x...2024`)
- Batch validation with `check_required_secrets()`
- Interactive prompts via `request_api_key()` when keys are missing

#### Layer 2: Credential Storage (Accounts & Payment Methods)

Store complete account credentials and payment methods with a split storage architecture:

- **Metadata** (in SQLite) — service names, usernames, emails, card last-4, billing info
- **Sensitive data** (in Keyring only) — passwords, full card numbers, CVVs

```python
# Store an account
agent.run("Store my login for github.com: username is 'myuser', password is 'secret123'")

# Store a credit card
agent.run("Store my Visa card ending in 4242 for online purchases")

# List all credentials (sensitive data masked)
agent.run("What accounts do I have stored?")

# Search credentials
agent.run("Find my credentials for anything related to AWS")

# Retrieve with secrets (explicit request required)
agent.run("Get my github.com password - I need to log in")
```

**Credential types supported:**

| Type | Metadata Stored | Secrets in Keyring |
|------|-----------------|-------------------|
| `account` | service, username, email | password |
| `credit_card` | last 4 digits, card type, expiry, billing address | full card number, CVV |
| `api_key` | service name, description | API key value |

See the `Credential` model in [The Data Model](#the-data-model) section for the full schema.

**Security guarantees:**
- Passwords are **never** stored in the database
- Card numbers only stored as last 4 digits in DB — full number in keyring
- All sensitive data encrypted at rest via OS keyring
- Explicit `include_secrets=True` flag required to retrieve sensitive data
- Card types auto-detected from number patterns (Visa, MasterCard, Amex, Discover, UnionPay)

## Configuration

BabyAGI uses a `config.yaml` file with environment variable substitution.

### Basic Configuration

```yaml
# config.yaml
owner:
  id: "your-name"
  email: "${OWNER_EMAIL}"
  contacts:
    email: "${OWNER_EMAIL}"
    # sms: "${OWNER_PHONE}"

# Verbose output: off, light, or deep (can also use BABYAGI_VERBOSE env var)
verbose: off

channels:
  cli:
    enabled: true

  email:
    enabled: true
    poll_interval: 60  # seconds between inbox checks
    api_key: "${AGENTMAIL_API_KEY}"
    inbox_id: "${AGENTMAIL_INBOX_ID}"

  voice:
    enabled: false
    wake_word: "hey assistant"
    stt_provider: "whisper"    # or "openai"
    tts_provider: "pyttsx3"    # or "openai"
    whisper_model: "base"      # tiny, base, small, medium, large
    sample_rate: 16000
    max_duration: 10           # max recording seconds

agent:
  model: "claude-sonnet-4-20250514"
  # model: "claude-opus-4-20250514"  # For more complex tasks

  # Agent identity
  name: "${AGENT_NAME:Assistant}"
  description: "${AGENT_DESCRIPTION:a helpful AI assistant}"
  objective: "Help my owner with tasks..."

  # Behavior settings
  behavior:
    spending:
      require_approval: true
      auto_approve_limit: 0.0  # USD
    external_policy:
      respond_to_unknown: true
      consult_owner_threshold: "medium"  # low, medium, high
    accounts:
      use_agent_email: true
      check_existing_first: true

# Memory configuration (optional)
memory:
  enabled: true  # Set to false to disable persistent memory
  path: "~/.babyagi/memory"  # SQLite database location
  background_extraction: true  # Extract entities/relationships in background
  extraction_interval: 60  # Seconds between extraction runs
```

### Environment Variables

```bash
# Required
export ANTHROPIC_API_KEY="your-anthropic-key"

# For email channel
export AGENTMAIL_API_KEY="your-agentmail-key"
export OWNER_EMAIL="you@example.com"

# Optional - for enhanced features
export AGENTMAIL_INBOX_ID="your-inbox-id"  # Auto-created if not set
export BROWSER_USE_API_KEY="your-browser-use-key"  # Browser automation
export E2B_API_KEY="your-e2b-key"  # Sandbox for dynamic tools
export OWNER_ID="your-name"  # Owner identifier
```

## API Server Endpoints

When running in server mode (`python main.py serve`), the following endpoints are available:

### POST /message

Send a message to the agent.

```bash
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello!", "thread_id": "main"}'

# Async mode - returns immediately, processes in background
curl -X POST http://localhost:8000/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Research AI trends", "thread_id": "main", "async_mode": true}'
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

Returns:
```json
{
  "status": "ok",
  "objectives_count": 2,
  "threads_count": 3,
  "tools": ["memory", "objective", "notes", "schedule", "register_tool", "send_message", ...]
}
```

## Extending the System

### Adding a New Channel

To add a new channel (e.g., Telegram), create a listener and sender:

**1. Create Listener** (`listeners/telegram.py`):

```python
async def run_telegram_listener(agent, config: dict = None):
    """Listen for Telegram messages."""
    bot = TelegramBot(config["token"])

    async for update in bot.get_updates():
        is_owner = update.from_user.id == config["owner_id"]

        response = await agent.run_async(
            content=update.message.text,
            thread_id=f"telegram:{update.chat.id}",
            context={
                "channel": "telegram",
                "is_owner": is_owner,
                "chat_id": update.chat.id,
            }
        )

        # Auto-reply for owner
        if is_owner and response:
            await bot.send_message(update.chat.id, response)
```

**2. Create Sender** (`senders/telegram.py`):

```python
class TelegramSender:
    name = "telegram"
    capabilities = ["images", "documents"]

    def __init__(self, token: str):
        self.bot = TelegramBot(token)

    async def send(self, to: str, content: str, **kwargs) -> dict:
        await self.bot.send_message(to, content)
        return {"sent": True, "channel": "telegram"}
```

**3. Register in `main.py`**:

```python
if is_channel_enabled(config, "telegram"):
    agent.register_sender("telegram", TelegramSender(config["token"]))
    tasks.append(run_telegram_listener(agent, config))
```

**4. Add to config**:

```yaml
channels:
  telegram:
    enabled: true
    token: "${TELEGRAM_BOT_TOKEN}"
    owner_id: 123456789
```

**Total: ~80 lines for a new channel.**

### Building a UI (Event System)

The agent emits events for all key operations, making it easy to build UIs or logging systems:

**Available Events:**
- `tool_start` — Tool execution starting: `{name, input}`
- `tool_end` — Tool execution completed: `{name, result, duration_ms}`
- `tool_registered` — New tool registered: `{name, description, parameters, source_code}`
- `tool_disabled` — Tool disabled: `{name, reason}`
- `objective_start` — Background objective starting: `{id, goal, priority, attempt}`
- `objective_end` — Background objective completed: `{id, status, result, spent_usd, tokens_used}`
- `objective_retry` — Objective retrying after failure: `{id, attempt, max_retries, delay_seconds, error}`
- `objective_usage` — Token/cost update for objective: `{id, input_tokens, output_tokens, cost_usd, total_spent_usd, total_tokens_used}`
- `task_start` — Scheduled task starting: `{id, name, goal}`
- `task_end` — Scheduled task completed: `{id, status, duration_ms}`
- `thread_repaired` — Corrupted thread auto-repaired: `{thread_id, repaired}`
- `agent_response` — Agent finished generating response

**Example: WebSocket UI**

```python
from agent import Agent

class WebSocketUI:
    def __init__(self, agent: Agent, websocket):
        self.ws = websocket

        # Subscribe to agent events
        agent.on("tool_start", self.on_tool_start)
        agent.on("tool_end", self.on_tool_end)
        agent.on("objective_start", self.on_objective)
        agent.on("*", self.on_any)  # Wildcard: all events

    async def on_tool_start(self, event):
        await self.ws.send_json({
            "type": "tool_start",
            "tool": event["name"],
            "input": event.get("input")
        })

    async def on_tool_end(self, event):
        await self.ws.send_json({
            "type": "tool_end",
            "tool": event["name"],
            "duration_ms": event.get("duration_ms")
        })

    async def on_objective(self, event):
        await self.ws.send_json({
            "type": "objective",
            "id": event["id"],
            "goal": event["goal"]
        })

    async def on_any(self, event):
        # Log all events
        print(f"Event: {event.get('_event')}")
```

**Example: File Logger**

```python
import json
from datetime import datetime

def setup_file_logger(agent: Agent, log_path: str):
    def log_event(event):
        with open(log_path, "a") as f:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "event": event.get("_event"),
                "data": {k: v for k, v in event.items() if k != "_event"}
            }
            f.write(json.dumps(entry) + "\n")

    agent.on("*", log_event)
```

The CLI's verbose output is itself just an event subscriber—see `listeners/cli.py` for the reference implementation.

### Adding Custom Tools (Class-based)

```python
from agent import Agent, Tool

def my_tool_fn(params: dict, agent):
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

With package and environment requirements:

```python
@tool(packages=["requests"], env=["API_KEY"])
def fetch_data(endpoint: str) -> dict:
    """Fetch data from an API endpoint."""
    import requests
    return requests.get(endpoint).json()
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

BabyAGI uses SQLite for all persistent data by default:

- **Tool definitions** — Dynamic tools stored in `tool_definitions` table
- **Credentials** — Account metadata in `credentials` table (secrets in keyring)
- **Scheduled tasks** — Stored in `~/.babyagi/scheduler/tasks.json`
- **Memory/Events** — Stored in SQLite with the memory system

For production scaling, you can swap SQLite for PostgreSQL or add Redis caching.

## Tool Health System

BabyAGI includes an automatic tool health checking system that detects:
- Which tools are ready to use
- Which tools need API keys configured
- Which tools are missing required packages

```python
from tools import check_tool_health, get_health_summary

# Get detailed health status
health = check_tool_health()
# Returns: {
#   "ready": ["memory", "objective", "notes", "web_search", ...],
#   "needs_setup": [{"name": "browse", "missing": ["BROWSER_USE_API_KEY"], "reason": "api_keys"}],
#   "unavailable": [{"name": "voice", "missing": ["sounddevice"], "reason": "packages"}],
#   "summary": {"total_ready": 8, "needs_api_keys": 2, "needs_packages": 1}
# }

# Get human-readable summary (used in AI greeting)
summary = get_health_summary()
# Returns: "Ready: memory, objective, notes, web_search\nNeed API keys: browse(BROWSER_USE_API_KEY)"
```

## Design Principles

### What Makes This Elegant

| Principle | Implementation |
|-----------|----------------|
| **Single loop** | One control flow for everything |
| **Tools are data** | Same structure handles memory, tasks, objectives, scheduling |
| **Three tool types** | Executable (code), Skills (behavior), Composio (integrations) |
| **LLM does the hard work** | Routing, synthesis, organization |
| **No frameworks** | Just Python and the API |
| **Extensible by design** | New capabilities = new tools, skills, or Composio apps |
| **Safe extensibility** | Dynamic tools sandboxed, skills safety-scanned |
| **Graceful degradation** | Missing packages/APIs degrade functionality but don't crash |
| **Persistence** | Tools, skills, and scheduled tasks survive restarts |
| **Secure by default** | Credentials split between DB metadata and keyring secrets |

### The Entire System

~300 lines of core code. Everything else is tools, memory, and channels.

```
agent.py
├── Tool class (tool definition with health checks)
├── Agent class (main loop + channel support + EventEmitter)
├── Objective class (background work with priority/budget)
├── Core tools (memory, objectives, notes, schedule, register, send_message)
├── Skill/Composio loading (three tool types on startup)
└── Sender registration

scheduler.py
├── Schedule class (at, every, cron support)
├── ScheduledTask class (task with execution tracking)
├── SchedulerStore class (JSON persistence)
└── Scheduler class (execution engine)

memory/
├── __init__.py (Memory facade)
├── store.py (SQLite backend with vector search)
├── models.py (Event, Entity, Edge, Topic, etc.)
├── extraction.py (NLP entity/relationship extraction)
├── summaries.py (Hierarchical summary management)
├── context.py (Context assembly for prompts)
├── retrieval.py (Quick and deep retrieval)
├── embeddings.py (OpenAI embeddings with caching)
├── integration.py (Agent hooks and enhanced memory tool)
└── tool_context.py (Intelligent tool selection)

metrics/
├── __init__.py
├── clients.py (Instrumented API client)
├── costs.py (Token cost calculation)
├── collector.py (Metrics aggregation)
└── models.py (Metric data models)

utils/
├── __init__.py
├── events.py (EventEmitter mixin)
├── console.py (styled terminal output)
├── collections.py (ThreadSafeList)
└── email_client.py (Email utilities)

listeners/
├── __init__.py
├── cli.py (terminal input + verbose output)
├── email.py (inbox polling)
└── voice.py (speech input)

senders/
├── __init__.py (Sender protocol)
├── cli.py (styled terminal output)
└── email.py (AgentMail output)

tools/
├── __init__.py (decorator framework + health checks)
├── sandbox.py (e2b code execution)
├── skills.py (skills, Composio, workflows)
├── web.py (search, browse, fetch)
├── email.py (AgentMail tools)
├── secrets.py (API key storage)
├── credentials.py (account & payment storage)
├── metrics.py (cost tracking tools)
└── verbose.py (verbosity control)

config.py (YAML loader with env substitution)
config.yaml (channel configuration)
server.py (FastAPI endpoints)
main.py (orchestration + verbose config)
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

# Priority-based execution (urgent work runs first)
agent.run("URGENT: Check if our API is down")  # Gets priority=1
agent.run("When you have time, research market trends")  # Gets priority=7

# Cost-controlled objectives
agent.run("Research competitors but limit cost to $1.00")  # budget_usd=1.00
agent.run("Analyze logs but use max 50000 tokens")  # token_limit=50000

# Check status with cost tracking
agent.run("Show me all objectives with their costs")
# Returns: id, goal, status, priority, spent_usd, tokens_used, retry_count
```

**Resource Controls:**
- Max 5 objectives run concurrently; others queue by priority
- Failed objectives auto-retry with exponential backoff (2s → 4s → 8s)
- Budget/token limits prevent runaway costs
- Cancel immediately stops running objectives

### Scheduled Tasks

```python
# One-time reminder
agent.run("Remind me in 2 hours to take a break")

# Daily summary
agent.run("Every day at 9am, summarize my unread emails")

# Health monitoring
agent.run("Every 5 minutes, check the server status and alert me if it's down")

# Cron-based scheduling
agent.run("Run backups at 2am on weekdays")
```

### Reflection and Consolidation

```python
# Periodically consolidate memories
agent.run("""
Review all my memories and notes. Create a summary of:
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
Add a note for each memory I've stored today.
""")
# Agent will: search memories → add note for each
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

The memory system uses a three-layer architecture:

1. **Raw Event Log** — Every interaction stored immutably in SQLite with timestamps
2. **Extracted Knowledge Graph** — Entities (people, orgs, concepts), relationships, and topics extracted via LLM
3. **Hierarchical Summaries** — Pre-computed summaries at multiple levels for fast context assembly

**Key Features:**

- **Semantic Search** — Vector embeddings (OpenAI text-embedding-3-small) enable meaning-based retrieval
- **Entity Extraction** — Background process identifies people, organizations, concepts from events
- **Relationship Discovery** — Connections between entities tracked (e.g., "John works_at Acme")
- **Topic Tracking** — Conversations categorized by topic for better organization
- **Context Assembly** — Smart context building pulls relevant history, summaries, and entities

**The LLM is still the organizer.** The knowledge graph provides structure, but the LLM synthesizes meaning.

**Storage Location:** `~/.babyagi/memory/memory.db` (configurable via `memory.path` in config.yaml)

**Thread Repair:**

If tool execution fails mid-conversation, the message history can become corrupted (orphaned `tool_use` without matching `tool_result`). The agent automatically repairs this:

```python
# Automatic repair on every run_async call
# Manual repair if needed
result = agent.repair_thread("main")
# Returns: {"repaired": 1, "message": "Repaired 1 orphaned tool_use block(s)"}
```

**Graceful Degradation:**

Memory features degrade gracefully:
- If SQLite is unavailable, falls back to in-memory storage (session only)
- If entity extraction fails, events are still logged
- If summarization fails, raw events are still searchable
- If embeddings API unavailable, keyword search still works

## API Reference

### Agent

```python
class Agent(EventEmitter):
    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        load_tools: bool = True,
        config: dict = None  # Channel/owner configuration
    )

    # Properties
    memory: Memory | None  # SQLite memory system (None if disabled)
    scheduler: Scheduler   # Task scheduler
    tools: dict[str, Tool] # Registered tools
    senders: dict[str, Sender]  # Channel senders

    # Tool registration
    def register(self, tool: Tool | ToolLike, emit_event: bool = False) -> None
    def register_sender(self, channel: str, sender: Sender) -> None

    # Message processing
    def run(self, user_input: str, thread_id: str = "main") -> str
    async def run_async(
        self,
        user_input: str,
        thread_id: str = "main",
        context: dict = None  # Channel context (is_owner, sender, channel)
    ) -> str
    async def run_scheduler(self) -> None  # Start the scheduler loop
    async def run_objective(self, objective_id: str) -> None  # Run objective

    # Thread management
    def get_thread(self, thread_id: str = "main") -> list
    def clear_thread(self, thread_id: str = "main") -> None
    def repair_thread(self, thread_id: str = "main") -> dict  # Fix corrupted threads

    # Memory convenience methods (for dynamic tools)
    def memory_recall(self, query: str) -> dict  # Search memory
    def memory_store(self, content: str) -> dict  # Store to memory

    # Event methods (inherited from EventEmitter)
    def on(self, event: str, handler: Callable) -> Callable  # Subscribe
    def off(self, event: str, handler: Callable = None) -> None  # Unsubscribe
    def emit(self, event: str, data: dict = None) -> None  # Emit event
    def once(self, event: str, handler: Callable) -> Callable  # One-time subscription
```

### Tool

```python
class Tool:
    name: str
    description: str
    parameters: dict  # JSON Schema
    fn: Callable[[dict, Agent], dict]
    packages: list[str]  # Required packages
    env: list[str]       # Required env vars

    def execute(self, params: dict, agent: Agent) -> dict
    def check_health(self) -> dict  # Check requirements
```

### Objective

```python
@dataclass
class Objective:
    id: str
    goal: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    thread_id: str
    schedule: str | None
    result: str | None
    error: str | None
    created: str
    completed: str | None
    # Priority and retry controls
    priority: int = 5  # 1-10, lower = higher priority
    retry_count: int = 0
    max_retries: int = 3
    last_error: str | None = None
    # Budget tracking
    budget_usd: float | None = None  # Max cost allowed
    spent_usd: float = 0.0
    token_limit: int | None = None  # Max tokens allowed
    tokens_used: int = 0
```

### Schedule

```python
@dataclass
class Schedule:
    kind: Literal["at", "every", "cron"]
    at: str | None      # ISO timestamp for one-time
    every: str | None   # Interval: "5m", "2h", "1d"
    cron: str | None    # Cron expression
    tz: str | None      # Timezone

    def next_run(self, after: datetime = None) -> datetime | None
    def human_readable(self) -> str
```

### ScheduledTask

```python
@dataclass
class ScheduledTask:
    id: str
    name: str
    goal: str
    schedule: Schedule
    enabled: bool = True
    next_run_at: str | None
    last_run_at: str | None
    last_status: str | None  # "ok", "error", "skipped"
    run_count: int = 0
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

# With requirements
@tool(packages=["httpx"], env=["API_KEY"])
def api_call(endpoint: str) -> dict:
    """Call an API endpoint."""
    ...

# With custom name/description
@tool(name="custom_name", description="Custom description")
def another_function(x: float) -> float:
    return x * 2
```

### EventEmitter

```python
from utils.events import EventEmitter

class EventEmitter:
    """Mixin class for event-driven communication."""

    def on(self, event: str, handler: Callable) -> Callable
        """Subscribe to an event. Use "*" for all events."""

    def off(self, event: str, handler: Callable = None) -> None
        """Unsubscribe. If handler is None, removes all handlers for event."""

    def emit(self, event: str, data: dict = None) -> None
        """Emit event to all subscribers. data["_event"] is set automatically."""

    def once(self, event: str, handler: Callable) -> Callable
        """Subscribe for a single emission only."""
```

### Console

```python
from utils.console import console, VerboseLevel

# Message output
console.banner("Title", width=40)  # Styled header
console.user("message")            # Green user message
console.agent("message")           # Cyan agent response
console.system("message")          # Blue system info
console.success("message")         # Green success
console.warning("message")         # Yellow warning
console.error("message")           # Red error

# Verbose output (only shown if level is enabled)
console.verbose("message", level=VerboseLevel.LIGHT)
console.verbose("detailed info", level=VerboseLevel.DEEP)

# Verbose level control
console.set_verbose("light")       # "off", "light", "deep"
console.get_verbose()              # Returns VerboseLevel enum

# Convenience methods for event logging
console.tool_start("memory", {"action": "store"})
console.tool_end("memory", {"stored": True}, duration_ms=12)
console.objective_start("abc123", "Research AI trends")
console.objective_end("abc123", "completed")
console.task_start("task1", "Daily summary")
console.task_end("task1", "ok", duration_ms=5000)
```

## Requirements

- Python >= 3.12

### Core Dependencies

```
anthropic >= 0.76.0
e2b-code-interpreter >= 1.0.0
fastapi >= 0.115.0
pydantic >= 2.12.5
uvicorn >= 0.32.0
duckduckgo-search >= 7.0.0
httpx >= 0.27.0
beautifulsoup4 >= 4.12.0
agentmail >= 0.1.0
keyring >= 25.0.0
keyrings-cryptfile >= 1.3.9
ddgs >= 9.10.0
croniter >= 2.0.0
pyyaml >= 6.0
composio-core >= 0.7.21
```

### Optional (for voice channel)

```
sounddevice
numpy
openai-whisper  # Local transcription
pyttsx3         # Local text-to-speech
openai          # Cloud STT/TTS
```

## Philosophy

> "Perfection is achieved, not when there is nothing more to add, but when there is nothing left to take away."
> — Antoine de Saint-Exupéry

This agent embodies that principle. It's not about what features you can add—it's about finding the minimal abstraction that makes features emerge naturally.

**Everything is a message. Everything is a tool. That's the whole system.**

## License

MIT
