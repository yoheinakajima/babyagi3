# memory/ - Graph-Based Persistent Memory System

The memory system provides BabyAGI with persistent, structured recall across conversations. It uses a three-layer architecture: raw event logging, an extracted knowledge graph, and pre-computed hierarchical summaries.

**Storage**: SQLite database at `~/.babyagi/memory/memory.db` (configurable via `memory.path` in `config.yaml`).

## Architecture

```
User/Agent Interactions
        |
        v
  [Event Logging]  ──>  events table (immutable log)
        |
        v (background)
  [Extraction Pipeline]  ──>  entities, edges, topics tables
        |
        v (periodic)
  [Summary Manager]  ──>  summaries table (hierarchical tree)
        |
        v (on demand)
  [Context Assembly]  ──>  system prompt section
```

## Files

| File | Purpose |
|------|---------|
| [`__init__.py`](__init__.py) | `Memory` facade class that ties all subsystems together |
| [`models.py`](models.py) | Dataclasses: `Event`, `Entity`, `Edge`, `Topic`, `SummaryNode`, `Learning`, etc. |
| [`store.py`](store.py) | SQLite backend — table creation, CRUD, vector search, retention policy |
| [`context.py`](context.py) | Assembles relevant memory into a system prompt section for the LLM |
| [`extraction.py`](extraction.py) | Background NLP pipeline — extracts entities, relationships, topics from events |
| [`retrieval.py`](retrieval.py) | `QuickRetrieval` (filter-based) and `DeepRetrievalAgent` (agentic search) |
| [`summaries.py`](summaries.py) | Hierarchical summary tree — leaf/rollup summaries, stale tracking, refresh |
| [`embeddings.py`](embeddings.py) | Vector embedding generation (OpenAI/Voyage) with caching |
| [`integration.py`](integration.py) | Agent event hooks — logs conversations, connects memory to the event system |
| [`learning.py`](learning.py) | Self-improvement: `FeedbackExtractor`, `ObjectiveEvaluator`, `PreferenceSummarizer` |
| [`tool_context.py`](tool_context.py) | Intelligent tool selection — filters tools based on conversation context |

## Three Layers

### 1. Raw Event Log

Every interaction is stored as an immutable `Event` with timestamps, channel info, and direction (inbound/outbound/internal). Events are the ground truth.

### 2. Extracted Knowledge Graph

A background pipeline runs periodically (default: every 60 seconds) and uses the LLM to extract:

- **Entities** — People, organizations, tools, concepts (with aliases)
- **Edges** — Relationships between entities (e.g., "works_at", "invested_in") with strength scores
- **Topics** — Subject areas and keyword clusters

### 3. Hierarchical Summaries

Pre-computed summaries organized in a tree structure. Each node tracks staleness so it can be refreshed when enough new events accumulate. A special `user_preferences` node is always included in the agent's context.

## Self-Improvement System

The learning subsystem (`learning.py`) enables the agent to improve over time:

- **FeedbackExtractor** — Detects corrections and preferences in user messages
- **ObjectiveEvaluator** — Self-evaluates completed background objectives
- **LearningRetriever** — Fetches relevant past learnings via vector similarity
- **PreferenceSummarizer** — Aggregates learnings into a user preferences summary

See [`docs/DESIGN_SELF_IMPROVEMENT.md`](../docs/DESIGN_SELF_IMPROVEMENT.md) for the full design document.

## Usage

```python
from memory import Memory

memory = Memory(store_path="~/.babyagi/memory")

# Log an event
event = memory.log_event(
    content="User prefers short emails",
    channel="cli",
    direction="inbound",
    is_owner=True,
)

# Search events semantically
results = memory.search_events("email preferences", limit=5)

# Find entities in the knowledge graph
people = memory.find_entities(query="John", type="person")

# Get pre-computed context for the LLM
context = memory.assemble_context(event)
```

## Graceful Degradation

The memory system degrades gracefully when components are unavailable:

- **SQLite unavailable** -> Falls back to in-memory storage (session only)
- **Extraction fails** -> Events are still logged; graph just doesn't update
- **Embeddings API unavailable** -> Keyword search still works
- **Summarization fails** -> Raw events remain searchable

## Related Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Memory system diagrams
- [MODELS.md](../MODELS.md) — Full data model reference
- [docs/DESIGN_SELF_IMPROVEMENT.md](../docs/DESIGN_SELF_IMPROVEMENT.md) — Learning system design
