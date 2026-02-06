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

The learning subsystem (`learning.py`) enables the agent to improve over time by capturing feedback, self-evaluating objectives, storing learnings, and retrieving them when relevant.

### How It Works

```
User Message ──> FeedbackExtractor ──> Learning record
                                            |
Objective Ends ──> ObjectiveEvaluator ──> Learning record(s)
                                            |
                                            v
                                     learnings table
                                            |
                        ┌───────────────────┼────────────────────┐
                        v                   v                    v
              LearningRetriever    PreferenceSummarizer    Vector Search
              (tool/objective)     (user_preferences node)  (similarity)
                        |                   |
                        v                   v
                  context.learnings   context.user_preferences
                        |                   |
                        └───────┬───────────┘
                                v
                          System Prompt
```

### Learning Model

Each learning is stored as a `Learning` record with:

- **Source**: `source_type` (user_feedback, self_evaluation, observation) and link to the originating event
- **Content**: The insight text plus a vector embedding for similarity search
- **Classification**: `sentiment` (positive/negative/neutral), `confidence` (0-1), and `category` (general, owner_profile, agent_self, tool_feedback)
- **Associations**: Optional `tool_id`, `topic_ids`, `objective_type`, `entity_ids`
- **Actionable insight**: `applies_when` (condition) and `recommendation` (what to do differently)
- **Usage tracking**: `times_applied` counter and `last_applied_at` timestamp

### Components

| Class | Purpose |
|-------|---------|
| **FeedbackExtractor** | Analyzes incoming owner messages for corrections, preferences, praise, or complaints about prior work. Uses LLM to detect feedback and creates a `Learning` tied to the relevant tool or objective type. |
| **ObjectiveEvaluator** | When a background objective completes or fails, evaluates the approach: were tools used well? Was the approach efficient? Generates learnings from the self-evaluation. |
| **LearningRetriever** | Fetches relevant learnings for context assembly — by tool name, by objective (vector similarity), by objective type, by category, or recent. Tracks when learnings are applied. |
| **PreferenceSummarizer** | Aggregates all learnings (up to 50, prioritizing recent) into a concise `user_preferences` summary organized by category: owner profile, agent rules, communication preferences, and tool/work preferences. |

### Integration

The system hooks into the agent event loop via `integration.py`:

- **`message_received`** — Runs `FeedbackExtractor` on owner messages, creates learnings, increments staleness on the `user_preferences` summary node
- **`objective_end`** — Runs `ObjectiveEvaluator` on completed/failed objectives, stores resulting learnings, triggers preference re-summarization

### Context Assembly

When building the system prompt (`context.py`):

1. **User preferences** (budget: 300 tokens) — Always included from the `user_preferences` summary node
2. **Relevant learnings** (budget: 200 tokens) — Included based on what the agent is doing:
   - If using a tool → retrieves tool-specific learnings (prioritizing corrections)
   - If starting an objective → retrieves similar past objective learnings via vector search

### Example Flow

```
User: "Actually, keep emails under 3 paragraphs. That last one was too long."

→ FeedbackExtractor detects:
  - feedback_type: "preference"
  - about_tool: "send_message"
  - sentiment: "negative"
  - recommendation: "Keep emails under 3 paragraphs"

→ Learning created and stored with embedding

→ user_preferences summary refreshed:
  "- Keep emails concise, under 3 paragraphs"

→ Next time send_message is used, the learning appears in context
```

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
