# BabyAGI v0.3.0

## Overview
BabyAGI is an AI assistant with background objectives, multi-channel support (CLI, email, voice), and an integrated SQLite-based memory system with embeddings for events, entities, relationships, topics, and tasks.

## Project Structure
```
.
├── main.py           # Entry point with channel modes
├── agent.py          # Core Agent class with tool execution
├── scheduler.py      # Background task scheduling
├── config.yaml       # Configuration settings
├── listeners/
│   └── cli.py        # CLI REPL listener
├── memory/           # SQLite-based memory with embeddings
│   ├── __init__.py
│   └── store.py
├── tools/            # External tool implementations
├── utils/
│   └── console.py    # Console output helpers
└── replit.md         # Project metadata (this file)
```

## Tech Stack
- Python 3.12
- Anthropic Claude API (claude-sonnet-4-20250514)
- SQLite for memory storage (~/.babyagi/memory/memory.db)
- Embeddings for semantic search

## Dependencies
- anthropic
- httpx
- pyyaml
- numpy
- Additional tool-specific packages

## Required Environment Variables
- `ANTHROPIC_API_KEY` - API key for Anthropic Claude

## Running the Application
```bash
python main.py
```

This starts an interactive CLI where you can chat with the agent.

## Core Features
- **Memory System**: SQLite-based storage with embeddings for events, entities, relationships, topics
- **Scheduler**: Background task scheduling with cron/interval support
- **Multi-channel**: CLI, email, and voice channel support
- **Extensible Tools**: Dynamic tool loading and registration

## Architecture
1. Main entry point (`main.py`) initializes Agent and starts CLI listener
2. CLI listener displays greeting, then starts scheduler in background
3. Scheduler runs scheduled tasks asynchronously without blocking CLI
4. Memory persists to ~/.babyagi/memory/memory.db
5. Scheduled tasks persist to ~/.babyagi/scheduler/tasks.json

## Known Issues & Fixes
- **Scheduler startup blocking**: Fixed by starting scheduler AFTER greeting displays (in cli.py with `start_scheduler=True` parameter). Previously, overdue scheduled tasks would block the event loop during startup.

## Recent Changes
- 2026-01-30: Fixed startup hang caused by scheduler executing overdue tasks before greeting display
- Restructured CLI listener to accept `start_scheduler` parameter
- All console output uses stderr with flush=True for workflow log visibility
