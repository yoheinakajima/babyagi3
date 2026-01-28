# Unified Message Abstraction Agent

## Overview
A minimal AI assistant built on a single unifying abstraction: everything is a message in a conversation. The entire system is a loop that processes messages and decides what to do next.

## Project Structure
```
.
├── agent.py          # Main application (Agent class, Tools, CLI)
├── README.md         # Documentation
├── pyproject.toml    # Python dependencies
└── replit.md         # Project metadata (this file)
```

## Tech Stack
- Python 3.12
- Anthropic Claude API (claude-sonnet-4-20250514)

## Dependencies
- anthropic (Python package)

## Required Environment Variables
- `ANTHROPIC_API_KEY` - API key for Anthropic Claude

## Running the Application
```bash
python agent.py
```

This starts an interactive REPL where you can chat with the agent.

## Core Features
- **Memory Tool**: Store and search memories (append-only log)
- **Task Tool**: Simple CRUD for task management
- **Register Tool**: Runtime tool registration for extensibility

## Architecture
The agent follows a minimal design:
1. Single loop: input → LLM → action → execute → output
2. Tools are data: memory, tasks, and extensibility are all tools
3. ~200 lines of core code
