# BabyAGI 3

A minimal AI agent you configure once and then interact with entirely through natural language. Tell it to remember things, research topics, send emails, schedule tasks, and learn new skills — it handles the rest.

> **Core philosophy**: Everything is a message in a conversation. The entire system is one loop: `input -> LLM -> action -> execute -> output`.

## Quick Start

```bash
# Install
git clone <repo-url> && cd babyagi_3
uv sync  # or: pip install -e .

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run
python main.py
```

That's it. You now have a persistent AI agent with memory, scheduling, background tasks, and web search.

Verbose mode is on by default — you'll see tool calls, webhook activity, and background operations as they happen. To turn it off, just tell the agent "turn off verbose" or type `/verbose off`.

## Initialization

BabyAGI figures out the right setup path automatically — no manual file management needed.

### How it works

On first run, BabyAGI checks whether it has enough configuration to start:

1. **Already configured?** If `OWNER_NAME` and `OWNER_EMAIL` are set (via environment variables or `config.yaml`), the agent starts immediately — no wizard, no prompts, no extra steps.
2. **Need info?** If the minimum config is missing, an LLM-powered setup conversation guides you through onboarding before the agent starts. You can ask it questions ("what does AgentMail do?", "do I need SendBlue?") and provide info in any order.
3. **Already ran setup before?** The agent remembers and skips straight to startup.

In every case, the agent starts normally after initialization — no restart required.

### What gets collected

At minimum, the setup assistant needs your **name** and **email**. It will also walk you through setting up:

- **Email channel ([AgentMail](https://agentmail.to))** — gives your agent its own email address for sending reports, handling signups, and email communication. Free tier available.
- **SMS/iMessage channel ([SendBlue](https://sendblue.co))** — lets you text your agent from your phone. Needs an API key, secret, and the SendBlue phone number from your dashboard. Also needs your personal phone number so the agent knows which texts are from you.

### Providing API keys

API keys can be provided in several ways. All options persist across restarts unless noted:

1. **Secure prompt during setup (recommended)** — after the setup conversation, a secure prompt asks for each API key. Input is hidden (never echoed) and never sent to the AI. Keys are stored in the system keyring.
2. **Environment variables** — `export SENDBLUE_API_KEY="..."` before running
3. **`.env` file** — create a `.env` file in the project root with `KEY=value` lines (loaded automatically if `python-dotenv` is installed)
4. **Replit secrets** — if running on Replit, add secrets in the Secrets panel (they appear as env vars)
5. **During the setup chat** — you can paste keys directly in the conversation. They work and are persisted to keyring, but they do pass through the LLM API. The secure prompt (option 1) is preferred.
6. **After setup** — tell the agent directly: *"set the SendBlue API key to ..."* or *"update my phone number to +15559876543"*. These are persisted to keyring.

### Updating configuration after setup

You can change any configuration value at any time by telling the agent:

```
You: Update my phone number to +15559876543
You: Set the SendBlue phone number to +15551234567
You: Change the AgentMail API key to am_xxx
You: Update my timezone to America/New_York
```

Changes take effect immediately — no restart needed.

### What gets set up automatically

Once setup finishes, two recurring tasks are scheduled:

- **Daily Stats Report** — compiles tool usage, memory extraction counts, LLM costs by model/source, scheduled task statuses, and errors from the last 24 hours. Emails the report to you.
- **Daily Self-Improvement** — the agent picks one concrete action each day to become more helpful: create a new skill, set up a useful scheduled task, or ask you a question to better understand your needs.

### Skipping the wizard

Set the minimum config and the wizard is bypassed entirely:

```bash
export OWNER_NAME="Alice"
export OWNER_EMAIL="alice@example.com"
python main.py  # starts immediately, no wizard
```

Or put it in `config.yaml`:

```yaml
owner:
  name: "Alice"
  email: "alice@example.com"
```

### Re-running setup

```bash
python main.py init
```

This re-launches the interactive setup wizard, then continues into the agent — no manual file deletion needed.

---

## What You Can Ask For

Everything below is done through natural conversation. The agent decides which tools to use.

### Remember things

```
You: Remember that the production server IP is 10.0.1.42
You: My favorite coffee shop is Blue Bottle on Market Street
You: John Smith's email is john@acme.com, he's the VP of Engineering

You: What's the server IP?
You: What do I know about John Smith?
You: Summarize everything you know about our infrastructure
```

Memory is persistent across restarts. The agent automatically extracts entities (people, companies, concepts), tracks relationships between them, and builds summaries it can recall later.

### Do research in the background

```
You: Research the top 5 CRM platforms and compare pricing
You: While that's running, help me draft an email to the team

You: What objectives are running?
You: Cancel the CRM research
```

Background objectives run asynchronously — you keep chatting while they work. They support priority queuing (1=urgent, 10=low), budget caps (`limit to $0.50`), and automatic retry on failure.

### Schedule tasks

```
You: Remind me in 30 minutes to stretch
You: Check my email every hour and summarize new messages
You: Send me a daily briefing at 9am on weekdays
You: Run server health checks every 5 minutes
```

Schedules support one-time (`in 5m`), recurring (`every 1h`), natural language (`weekdays at 9am`), and raw cron (`0 9 * * 1-5`). Tasks persist across restarts.

### Send messages across channels

```
You: Email alice@example.com with the meeting notes
You: Text me if the server goes down
You: Send the research results to my email when done
```

The agent can send messages via email, SMS/iMessage, or the CLI — depending on which channels you've configured.

### Create new tools on the fly

```
You: Create a tool that converts Celsius to Fahrenheit
You: Convert 100 Celsius

You: Create a tool that fetches stock prices from Yahoo Finance
You: What's AAPL trading at?
```

Dynamic tools are persisted to the database and survive restarts. Tools that need external packages are automatically sandboxed via E2B.

### Learn new skills

```
You: Learn the code review skill from https://example.com/skills/code-review/SKILL.md
You: Activate the code review skill and review this PR
```

Skills are behavioral instructions (SKILL.md files) that guide the agent's approach. They're safety-scanned before activation.

### Manage notes

```
You: Add a note: Review Q1 financials
You: Add a note: Call the dentist
You: What are my notes?
You: Complete note 0
```

### Connect to 250+ apps via Composio

```
You: Connect to Slack via Composio
You: Enable GitHub actions from Composio
You: Send a message to #general on Slack
```

Requires `composio login` or `COMPOSIO_API_KEY`.

---

## Configuration

BabyAGI uses `config.yaml` with environment variable substitution (`${VAR_NAME:default}`).

### Minimal Setup

Just set an API key. Everything else has sensible defaults:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."  # or OPENAI_API_KEY
python main.py
```

### Full Configuration

```yaml
# config.yaml

owner:
  id: "${OWNER_ID:owner}"
  email: "${OWNER_EMAIL:}"
  phone: "${OWNER_PHONE:}"

channels:
  cli:
    enabled: true

  email:
    enabled: true
    poll_interval: 60
    api_key: "${AGENTMAIL_API_KEY:}"

  voice:
    enabled: false
    wake_word: "hey assistant"

  sendblue:
    enabled: true
    api_key: "${SENDBLUE_API_KEY:}"
    api_secret: "${SENDBLUE_API_SECRET:}"

  meeting:
    enabled: true
    api_key: "${RECALL_API_KEY:}"

# LLM models by use case
llm:
  agent_model:
    model: "${AGENT_MODEL:claude-sonnet-4-20250514}"
    max_tokens: 8096
  coding_model:
    model: "${CODING_MODEL:claude-sonnet-4-20250514}"
  fast_model:
    model: "${FAST_MODEL:claude-3-5-haiku-20241022}"
    max_tokens: 1024
  memory_model:
    model: "${MEMORY_MODEL:claude-sonnet-4-20250514}"
    max_tokens: 2048

agent:
  name: "${AGENT_NAME:Assistant}"
  description: "${AGENT_DESCRIPTION:a helpful AI assistant}"
  behavior:
    spending:
      require_approval: true
      auto_approve_limit: 0.0
    external_policy:
      respond_to_unknown: true
      consult_owner_threshold: "medium"

memory:
  enabled: true
  path: "~/.babyagi/memory"
  background_extraction: true
  extraction_interval: 60

verbose: light  # off, light (default), deep
```

### Environment Variables

Set these as environment variables, in a `.env` file, or in Replit secrets.

| Variable | Required | Purpose |
|----------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes (or OPENAI_API_KEY) | LLM provider |
| `OPENAI_API_KEY` | Alternative | OpenAI provider |
| `AGENTMAIL_API_KEY` | For email | AgentMail email channel |
| `OWNER_EMAIL` | For email | Your email for auto-replies |
| `SENDBLUE_API_KEY` | For SMS | SendBlue SMS/iMessage |
| `SENDBLUE_API_SECRET` | For SMS | SendBlue secret |
| `SENDBLUE_PHONE_NUMBER` | For SMS | SendBlue "from" number (from your SendBlue dashboard) |
| `OWNER_PHONE` | For SMS | Your personal phone number (identifies owner texts) |
| `RECALL_API_KEY` | For meetings | Recall.ai meeting bot |
| `BROWSER_USE_API_KEY` | For browsing | Web browser automation |
| `E2B_API_KEY` | For sandboxing | Code execution sandbox |
| `COMPOSIO_API_KEY` | For integrations | 250+ app integrations |
| `AGENT_NAME` | Optional | Agent display name |
| `BABYAGI_VERBOSE` | Optional | `off`, `light` (default), `deep` |

---

## Running Modes

```bash
python main.py              # All enabled channels (default)
python main.py cli          # CLI only
python main.py channels     # All channels, no HTTP server
python main.py serve        # HTTP API server only (port 5000)
python main.py serve 8080   # Custom port
python main.py all          # Server + all channels combined
python main.py all 8080     # Full mode, custom port
python main.py init         # Re-run interactive setup, then start
```

See [RUNNING.md](RUNNING.md) for detailed examples of each mode, including API curl commands.

---

## How Memory Works

Memory is the most distinctive feature. It's not a flat list of facts — it's a three-layer system:

### Layer 1: Event Log

Every interaction (messages, tool calls, results) is stored as an immutable event with timestamps, channel info, and direction. This is the ground truth.

### Layer 2: Knowledge Graph

A background process runs every 60 seconds and uses the LLM to extract:

- **Entities** — People, companies, tools, concepts (e.g., "John Smith / person / VP of Engineering")
- **Relationships** — Connections between entities (e.g., "John Smith works_at Acme Corp")
- **Topics** — Subject clusters from conversations

### Layer 3: Hierarchical Summaries

Pre-computed summaries organized in a tree. Each node tracks staleness and gets refreshed when enough new events accumulate. A special `user_preferences` node (learned from your feedback over time) is always included in the agent's context.

### Self-Improvement

The agent learns from your feedback:

```
You: Actually, I prefer shorter emails — that last one was too long.
```

This creates a `Learning` record tied to the email tool. Next time the agent sends an email, it retrieves that learning and adjusts. Over time, all learnings are summarized into a `user_preferences` summary that's always in the agent's context.

### What you can ask

```
You: What do you know about John Smith?           # Entity lookup
You: How are John and Acme Corp related?           # Relationship query
You: Summarize what you know about our project     # Summary retrieval
You: Search your memory for anything about pricing # Semantic search
You: Show me your recent memories                  # Event listing
```

See [memory/README.md](memory/README.md) for the full memory system documentation.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Listeners                          │
│   CLI          Email         Voice        SMS        API    │
│   stdin        AgentMail     Whisper      SendBlue   HTTP   │
│         All call: agent.run_async(content, context)         │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          v
┌─────────────────────────────────────────────────────────────┐
│                        AGENT                                │
│  Threads    Memory    Objectives    Scheduler    Tools      │
│                          │                                  │
│               ┌──────────┴──────────┐                       │
│               │   send_message      │                       │
│               └──────────┬──────────┘                       │
└──────────────────────────┼──────────────────────────────────┘
                           │
                           v
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT: Senders                           │
│   CLI          Email         SMS          (any channel)     │
└─────────────────────────────────────────────────────────────┘
```

The full system in ~300 lines of core code. Everything else is tools, memory, and channels.

For detailed architecture diagrams (Mermaid), see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Project Structure

```
babyagi_3/
├── main.py              # Entry point (CLI, server, channel modes)
├── agent.py             # Core Agent class, Tool, Objective, main loop
├── config.py            # YAML config loader with env substitution
├── config.yaml          # Configuration file
├── llm_config.py        # Multi-provider LLM support (Anthropic/OpenAI)
├── scheduler.py         # Task scheduling (at/every/cron)
├── server.py            # FastAPI HTTP server
│
├── memory/              # Graph-based persistent memory system
├── tools/               # Tool framework and external tools
├── listeners/           # Input channels (CLI, email, voice, SMS)
├── senders/             # Output channels (CLI, email, SMS)
├── metrics/             # Cost tracking and instrumentation
├── utils/               # EventEmitter, console styling, thread-safe collections
└── docs/                # Design documents
```

Each folder has its own README with file-by-file documentation.

---

## Documentation Index

| Document | Contents |
|----------|----------|
| **[README.md](README.md)** (this file) | Setup, configuration, what you can ask, how memory works |
| **[RUNNING.md](RUNNING.md)** | Manual running examples for every mode |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System architecture with Mermaid diagrams |
| **[MODELS.md](MODELS.md)** | Complete data model reference |
| **[memory/README.md](memory/README.md)** | Memory system internals |
| **[tools/README.md](tools/README.md)** | Tool system and @tool decorator |
| **[listeners/README.md](listeners/README.md)** | Input channel details |
| **[senders/README.md](senders/README.md)** | Output channel details |
| **[utils/README.md](utils/README.md)** | EventEmitter, console, utilities |
| **[metrics/README.md](metrics/README.md)** | Cost tracking system |

---

## Extending

### Add a new channel (~80 lines)

1. Create `listeners/<channel>.py` — async function that polls and calls `agent.run_async()`
2. Create `senders/<channel>.py` — class implementing the `Sender` protocol
3. Register both in `main.py`
4. Add config to `config.yaml`

### Add a custom tool (~10 lines)

```python
from tools import tool

@tool
def my_tool(query: str, limit: int = 10) -> dict:
    """Does something useful."""
    return {"result": "..."}

agent.register(my_tool)
```

### Use a different model

```python
agent = Agent(model="claude-opus-4-20250514")
```

Or configure per use case in `config.yaml` under `llm:`.

---

## Requirements

- Python >= 3.12
- One of: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

Core dependencies: `anthropic`, `litellm`, `openai`, `fastapi`, `uvicorn`, `httpx`, `beautifulsoup4`, `duckduckgo-search`, `agentmail`, `keyring`, `croniter`, `pyyaml`, `e2b-code-interpreter`, `composio`.

Optional (voice): `sounddevice`, `numpy`, `openai-whisper`, `pyttsx3`.

## License

MIT
