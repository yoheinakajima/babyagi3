# Running BabyAGI - Manual Examples

This document shows how to run BabyAGI in every supported mode, with concrete examples of what you can ask the agent to do.

> For configuration, see [README.md](README.md).
> For architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Prerequisites

```bash
# 1. Install dependencies
uv sync
# or: pip install -e .

# 2. Set at minimum one LLM provider key
export ANTHROPIC_API_KEY="sk-ant-..."
# or: export OPENAI_API_KEY="sk-..."
```

---

## Mode 1: CLI Only (Default)

```bash
python main.py
# or explicitly:
python main.py cli
```

This starts a terminal REPL. Type messages and the agent responds inline.

### Example Session: Memory

```
You: Remember that the production server IP is 10.0.1.42
Assistant: I've stored that. Your production server IP is 10.0.1.42.

You: What's the server IP?
Assistant: Your production server IP is 10.0.1.42.
```

### Example Session: Notes

```
You: Add a note: Review Q1 financials
Assistant: Note added.

You: Add a note: Call the dentist
Assistant: Note added.

You: What are my notes?
Assistant: Your notes:
  0. Review Q1 financials
  1. Call the dentist

You: Complete note 0
Assistant: Marked "Review Q1 financials" as complete.
```

### Example Session: Background Objectives

```
You: Research the top 5 AI frameworks and compare their features
Assistant: I'll start that as a background objective. You can keep chatting.

You: What objectives are running?
Assistant: 1 objective running:
  - obj_abc123: "Research the top 5 AI frameworks..." (running, $0.03 spent)

You: How about we talk about something else while that runs?
Assistant: Sure! What would you like to discuss?
```

### Example Session: Scheduling

```
You: Remind me in 30 minutes to stretch
Assistant: Scheduled: "stretch reminder" will run in 30 minutes.

You: Check my email every hour and summarize new messages
Assistant: Scheduled: "email check" will run every 1 hour.

You: Show my scheduled tasks
Assistant: 2 tasks:
  1. "stretch reminder" - runs at 2:45 PM (one-time)
  2. "email check" - runs every 1h (next: 3:15 PM)
```

### Example Session: Dynamic Tool Creation

```
You: Create a tool that converts Celsius to Fahrenheit
Assistant: I've created the "celsius_to_fahrenheit" tool. It takes a temperature
in Celsius and returns the Fahrenheit equivalent.

You: Convert 100 Celsius
Assistant: 100°C = 212°F
```

### Example Session: Verbose Mode

```
You: /verbose light
Verbose output: light (key operations)

You: What do you remember about me?
  [tool] memory
  [tool] memory done (15ms)
Assistant: Here's what I know about you...

You: /verbose deep
Verbose output: deep (everything)

You: Store that I like dark roast coffee
  [tool] memory
    input: {action="store", content="User likes dark roast coffee"}
  [tool] memory done (8ms)
    result: {stored=true, event_id="evt_..."}
Assistant: Stored! You like dark roast coffee.

You: /verbose off
```

---

## Mode 2: Multi-Channel

```bash
python main.py channels
```

Runs all enabled channels concurrently. The CLI is always included alongside email, voice, SMS, etc.

### What this enables

- You can chat via CLI while the agent monitors your email inbox
- The agent can email you results from background objectives
- External people can email the agent and get responses
- The scheduler runs in the background

### Configuration required

Edit `config.yaml` to enable channels, or set environment variables:

```bash
# For email
export AGENTMAIL_API_KEY="your-key"
export OWNER_EMAIL="you@example.com"

# For SMS/iMessage
export SENDBLUE_API_KEY="your-key"
export SENDBLUE_API_SECRET="your-secret"
export OWNER_PHONE="+1234567890"
```

### Example: Cross-channel communication

```
You: Research competitor pricing and email me the results
Assistant: I'll research that in the background and email you when done.
[Background objective starts]
[... researches ...]
[Agent sends email to owner with results]
```

---

## Mode 3: API Server

```bash
python main.py serve         # Default port 5000
python main.py serve 8080    # Custom port
```

Starts a FastAPI HTTP server. No CLI input — interact via REST API.

### Send a message

```bash
curl -X POST http://localhost:5000/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello! Remember that my name is Alice.", "thread_id": "main"}'
```

Response:
```json
{
  "response": "Hello Alice! I'll remember your name.",
  "thread_id": "main"
}
```

### Async mode (non-blocking)

```bash
curl -X POST http://localhost:5000/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Research AI trends", "thread_id": "main", "async_mode": true}'
```

Response:
```json
{
  "response": "",
  "thread_id": "main",
  "queued": true
}
```

### List objectives

```bash
curl http://localhost:5000/objectives
```

### Get thread history

```bash
curl http://localhost:5000/threads/main
```

### Health check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "ok",
  "objectives_count": 1,
  "threads_count": 2,
  "tools": ["memory", "objective", "notes", "schedule", ...]
}
```

### Clear a thread

```bash
curl -X DELETE http://localhost:5000/threads/main
```

---

## Mode 4: Full Mode (Server + Channels)

```bash
python main.py all            # Default port 5000
python main.py all 8080       # Custom port
```

Runs everything: API server, CLI, email listener, SMS webhooks, scheduler. The server and all listeners share the same agent instance.

### What this adds over mode 2

- REST API endpoints for external integrations
- SendBlue SMS webhooks at `/webhooks/sendblue`
- Meeting bot webhooks at `/webhooks/recall`

---

## Programmatic Usage

```python
from agent import Agent

agent = Agent()

# Synchronous (blocks until response)
response = agent.run("What's 2 + 2?")
print(response)  # "4"

# With a specific thread
agent.run("I'm working on Project X", thread_id="work")
agent.run("What am I working on?", thread_id="work")
# -> "You're working on Project X"

# Async
import asyncio

async def main():
    agent = Agent()
    response = await agent.run_async("Hello!")
    print(response)

asyncio.run(main())
```

### Using without external tools

```python
agent = Agent(load_tools=False)  # Only core tools (memory, notes, etc.)
```

### Using a different model

```python
agent = Agent(model="claude-opus-4-20250514")
```

### Subscribing to events

```python
agent = Agent()

@agent.on("tool_start")
def on_tool(event):
    print(f"Tool: {event['name']}")

@agent.on("objective_end")
def on_obj(event):
    print(f"Objective {event['id']}: {event['status']}")

agent.run("Search the web for Python tutorials")
```

---

## Environment Variables Reference

### Required (one of)

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Claude API key |
| `OPENAI_API_KEY` | OpenAI API key (alternative provider) |

### Optional — Channels

| Variable | Description |
|----------|-------------|
| `AGENTMAIL_API_KEY` | Email via AgentMail |
| `AGENTMAIL_INBOX_ID` | Specific inbox (auto-created if not set) |
| `OWNER_EMAIL` | Owner's email for auto-replies |
| `SENDBLUE_API_KEY` | SMS/iMessage via SendBlue |
| `SENDBLUE_API_SECRET` | SendBlue API secret |
| `OWNER_PHONE` | Owner's phone number |
| `RECALL_API_KEY` | Meeting transcription via Recall.ai |

### Optional — Tools

| Variable | Description |
|----------|-------------|
| `BROWSER_USE_API_KEY` | Web browsing automation |
| `E2B_API_KEY` | Code execution sandbox |
| `COMPOSIO_API_KEY` | 250+ app integrations |

### Optional — Agent Identity

| Variable | Description |
|----------|-------------|
| `AGENT_NAME` | Agent's display name (default: "Assistant") |
| `AGENT_DESCRIPTION` | Agent's self-description |
| `OWNER_ID` | Owner's identifier |
| `BABYAGI_VERBOSE` | Verbosity: `off`, `light`, `deep` |

---

## Data Storage Locations

| Data | Location |
|------|----------|
| Memory database | `~/.babyagi/memory/memory.db` |
| Scheduled tasks | `~/.babyagi/scheduler/tasks.json` |
| Task run history | `~/.babyagi/scheduler/runs/<task_id>.jsonl` |
| Secrets | System keyring (or `~/.local/share/python_keyring/`) |

---

## Related Docs

- [README.md](README.md) — Configuration and agent interaction guide
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture
- [MODELS.md](MODELS.md) — Data model reference
