# Data Models Reference

This document describes all the data models used in BabyAGI. Models are defined as Python `@dataclass` classes.

> For architecture diagrams, see [ARCHITECTURE.md](ARCHITECTURE.md).
> For running examples, see [RUNNING.md](RUNNING.md).

---

## Agent Core Models

Defined in [`agent.py`](agent.py).

### Objective

Background work that runs asynchronously while the main conversation continues.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `goal` | `str` | required | What to accomplish |
| `status` | `str` | `"pending"` | `pending`, `running`, `completed`, `failed`, `cancelled` |
| `thread_id` | `str` | auto | Dedicated conversation thread for this objective |
| `schedule` | `str \| None` | `None` | Recurring schedule expression |
| `result` | `str \| None` | `None` | Final result text |
| `error` | `str \| None` | `None` | Error message if failed |
| `created` | `str` | auto | ISO timestamp |
| `completed` | `str \| None` | `None` | ISO timestamp when finished |
| `priority` | `int` | `5` | 1-10, lower = higher priority |
| `retry_count` | `int` | `0` | Current retry attempt |
| `max_retries` | `int` | `3` | Max retry attempts |
| `last_error` | `str \| None` | `None` | Error from last failed attempt |
| `budget_usd` | `float \| None` | `None` | Maximum cost allowed |
| `spent_usd` | `float` | `0.0` | Total cost so far |
| `token_limit` | `int \| None` | `None` | Maximum tokens allowed |
| `tokens_used` | `int` | `0` | Total tokens used |

### Tool

A callable function exposed to the LLM.

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique tool name |
| `description` | `str` | What the tool does (shown to LLM) |
| `parameters` | `dict` | JSON Schema for input |
| `fn` | `Callable` | Function to execute: `(params: dict, agent: Agent) -> dict` |
| `packages` | `list[str]` | Required Python packages |
| `env` | `list[str]` | Required environment variables |

---

## Scheduler Models

Defined in [`scheduler.py`](scheduler.py).

### Schedule

A time specification for when a task should run.

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `str` | `"at"` (one-time), `"every"` (interval), `"cron"` (expression) |
| `at` | `str \| None` | ISO timestamp for one-time schedule |
| `every` | `str \| None` | Interval string: `"5m"`, `"2h"`, `"1d"` |
| `cron` | `str \| None` | Cron expression: `"0 9 * * 1-5"` |
| `tz` | `str \| None` | Timezone: `"America/New_York"` |

### ScheduledTask

A task with its schedule and execution history.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `name` | `str` | required | Human-readable name |
| `goal` | `str` | required | What to execute (sent to agent) |
| `schedule` | `Schedule` | required | When to run |
| `enabled` | `bool` | `True` | Whether task is active |
| `next_run_at` | `str \| None` | auto | Next scheduled execution |
| `last_run_at` | `str \| None` | `None` | Last execution time |
| `last_status` | `str \| None` | `None` | `"ok"`, `"error"`, `"skipped"` |
| `run_count` | `int` | `0` | Total executions |

---

## Memory Models

Defined in [`memory/models.py`](memory/models.py).

### Event

An immutable log entry representing something that happened.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `timestamp` | `datetime` | auto | When it happened |
| `channel` | `str \| None` | `None` | `"email"`, `"sms"`, `"cli"`, `"api"` |
| `direction` | `str` | required | `"inbound"`, `"outbound"`, `"internal"` |
| `event_type` | `str` | required | `"message"`, `"tool_call"`, `"tool_result"`, etc. |
| `task_id` | `str \| None` | `None` | Associated objective/task |
| `tool_id` | `str \| None` | `None` | Associated tool |
| `person_id` | `str \| None` | `None` | Links to Entity |
| `is_owner` | `bool` | `False` | Whether from the agent's owner |
| `parent_event_id` | `str \| None` | `None` | For threading events |
| `conversation_id` | `str \| None` | `None` | Conversation group |
| `content` | `str` | `""` | The event content |
| `content_embedding` | `list[float] \| None` | `None` | Vector for semantic search |
| `extraction_status` | `str` | `"pending"` | `"pending"`, `"processing"`, `"complete"`, `"failed"` |

### Entity

A person, organization, tool, concept, or other named thing in the knowledge graph.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `name` | `str` | required | Canonical name |
| `type` | `str` | required | `"person"`, `"org"`, `"tool"`, `"concept"` |
| `type_raw` | `str` | required | Original type text (e.g., "venture capitalist") |
| `aliases` | `list[str]` | `[]` | Alternative names |
| `description` | `str \| None` | `None` | Summary description |
| `is_owner` | `bool` | `False` | Whether this is the agent's owner |
| `is_self` | `bool` | `False` | Whether this is the agent itself |
| `event_count` | `int` | `0` | Number of associated events |
| `first_seen` | `datetime` | auto | First mention |
| `last_seen` | `datetime` | auto | Most recent mention |

### Edge

A relationship between two entities.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `source_entity_id` | `str` | required | Source entity |
| `target_entity_id` | `str` | required | Target entity |
| `relation` | `str` | required | Free-form: "invested in", "works at" |
| `relation_type` | `str \| None` | `None` | Clustered: "financial", "professional" |
| `is_current` | `bool` | `True` | Whether relationship is current |
| `strength` | `float` | `0.5` | 0-1 relationship strength |

### Topic

An extracted theme or subject area.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `label` | `str` | required | Topic name |
| `description` | `str \| None` | `None` | Topic description |
| `keywords` | `list[str]` | `[]` | Associated keywords |
| `event_count` | `int` | `0` | Associated events |
| `entity_count` | `int` | `0` | Associated entities |

### SummaryNode

A node in the hierarchical summary tree.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `node_type` | `str` | required | `"root"`, `"channel"`, `"entity"`, `"topic"`, `"user_preferences"` |
| `key` | `str` | required | Unique key: `"root"`, `"entity:{uuid}"`, `"channel:email"` |
| `label` | `str` | required | Human-readable label |
| `parent_id` | `str \| None` | `None` | Parent node in tree |
| `summary` | `str` | `""` | The pre-computed summary text |
| `events_since_update` | `int` | `0` | Staleness counter |

### Learning

A piece of learned knowledge from feedback or self-evaluation.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `source_type` | `str` | required | `"user_feedback"`, `"self_evaluation"`, `"observation"` |
| `source_event_id` | `str \| None` | `None` | Triggering event |
| `content` | `str` | required | The learning/insight |
| `sentiment` | `str` | `"neutral"` | `"positive"`, `"negative"`, `"neutral"` |
| `confidence` | `float` | `0.5` | 0-1 confidence score |
| `category` | `str` | `"general"` | `"general"`, `"owner_profile"`, `"agent_self"`, `"tool_feedback"` |
| `tool_id` | `str \| None` | `None` | If about a specific tool |
| `topic_ids` | `list[str]` | `[]` | Related topics |
| `objective_type` | `str \| None` | `None` | e.g., "research", "code", "email" |
| `recommendation` | `str \| None` | `None` | What to do differently |
| `times_applied` | `int` | `0` | How often used in context |

### Fact

A fact triplet (subject-predicate-object) extracted from any source.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `subject_entity_id` | `str` | required | Links to Entity |
| `predicate` | `str` | required | The verb/relationship |
| `object_entity_id` | `str \| None` | `None` | If object is an entity |
| `object_value` | `str \| None` | `None` | If object is a literal value |
| `fact_type` | `str` | `"relation"` | `relation`, `attribute`, `event`, `state`, `metric` |
| `fact_text` | `str` | `""` | Full natural language sentence |
| `source_type` | `str` | `"conversation"` | `conversation`, `document`, `tool`, `observation` |
| `confidence` | `float` | `0.8` | Extraction confidence |
| `strength` | `float` | `0.5` | Accumulated over repeated mentions |
| `is_current` | `bool` | `True` | Whether still valid |

### ToolDefinition

A persisted tool definition — enables tools to survive restarts.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `name` | `str` | required | Tool name (unique) |
| `description` | `str` | required | What it does |
| `tool_type` | `str` | `"executable"` | `"executable"`, `"skill"`, `"composio"` |
| `source_code` | `str \| None` | `None` | Python code for dynamic tools |
| `parameters` | `dict` | `{}` | JSON schema for input |
| `packages` | `list[str]` | `[]` | Required packages |
| `skill_content` | `str \| None` | `None` | SKILL.md content (for skills) |
| `composio_app` | `str \| None` | `None` | e.g., "SLACK" (for composio) |
| `composio_action` | `str \| None` | `None` | e.g., "SLACK_SEND_MESSAGE" |
| `depends_on` | `list[str]` | `[]` | Tools/skills this depends on |
| `is_enabled` | `bool` | `True` | Whether active |
| `is_dynamic` | `bool` | `True` | False for built-in tools |
| `usage_count` | `int` | `0` | Total executions |
| `success_count` | `int` | `0` | Successful runs |
| `error_count` | `int` | `0` | Failed runs |
| `avg_duration_ms` | `float` | `0.0` | Average execution time |
| `last_error` | `str \| None` | `None` | Most recent error |

Computed properties:
- `success_rate` — `success_count / usage_count * 100`
- `is_healthy` — `True` if `success_rate >= 50%` (or insufficient data)

### Credential

Securely stored credential — user account or payment method.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | `str` | auto | Unique identifier |
| `credential_type` | `str` | required | `"account"`, `"credit_card"` |
| `service` | `str` | required | e.g., "github.com" |
| `username` | `str \| None` | `None` | For accounts |
| `email` | `str \| None` | `None` | For accounts |
| `password_ref` | `str \| None` | `None` | Keyring reference (never stored in DB) |
| `card_last_four` | `str \| None` | `None` | For credit cards |
| `card_type` | `str \| None` | `None` | "visa", "mastercard", etc. |
| `card_expiry` | `str \| None` | `None` | "MM/YY" |
| `billing_name` | `str \| None` | `None` | For credit cards |

---

## Metrics Models

Defined in [`metrics/models.py`](metrics/models.py).

### LLMCallMetric

Record of a single LLM API call.

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier |
| `timestamp` | `datetime` | When the call was made |
| `source` | `str` | `"agent"`, `"extraction"`, `"summary"`, `"retrieval"` |
| `model` | `str` | Model ID (e.g., `"claude-sonnet-4-20250514"`) |
| `input_tokens` | `int` | Tokens sent |
| `output_tokens` | `int` | Tokens received |
| `cost_usd` | `float` | Calculated cost |
| `duration_ms` | `int` | Latency |

### EmbeddingCallMetric

Record of a single embedding API call.

| Field | Type | Description |
|-------|------|-------------|
| `provider` | `str` | `"openai"`, `"voyage"` |
| `model` | `str` | e.g., `"text-embedding-3-small"` |
| `text_count` | `int` | Number of texts embedded |
| `cost_usd` | `float` | Calculated cost |
| `cached` | `bool` | Whether result was from cache |

### SessionMetrics

Aggregated metrics for a session or thread.

| Field | Type | Description |
|-------|------|-------------|
| `thread_id` | `str` | Which thread |
| `llm_call_count` | `int` | Total LLM calls |
| `total_input_tokens` | `int` | Total input tokens |
| `total_output_tokens` | `int` | Total output tokens |
| `total_llm_cost_usd` | `float` | Total LLM cost |
| `tool_call_count` | `int` | Total tool calls |
| `cost_by_source` | `dict[str, float]` | Cost breakdown by source |

### MetricsSummary

High-level summary across all sessions.

| Field | Type | Description |
|-------|------|-------------|
| `total_llm_calls` | `int` | Grand total |
| `total_tokens` | `int` | Grand total |
| `total_llm_cost_usd` | `float` | Grand total LLM cost |
| `total_embedding_cost_usd` | `float` | Grand total embedding cost |
| `cost_by_model` | `dict[str, float]` | Breakdown by model |
| `cost_by_source` | `dict[str, float]` | Breakdown by source |

---

## Context Models

Defined in [`memory/models.py`](memory/models.py).

### AssembledContext

The context assembled for each agent invocation, formatted into the system prompt.

| Field | Type | Description |
|-------|------|-------------|
| `identity` | `dict` | Agent name, description, owner info |
| `state` | `dict` | Current mood, focus, active topics |
| `knowledge` | `str` | Pre-computed summary of what the agent knows |
| `recent` | `dict` | Recent activity summary |
| `channel` | `dict \| None` | Current channel context |
| `counterparty` | `dict \| None` | Info about the person being talked to |
| `user_preferences` | `str` | Summarized preferences (always included) |
| `learnings` | `list[dict]` | Context-specific learnings |

Methods:
- `to_dict()` — Serialize to dictionary
- `to_prompt()` — Format as a system prompt section for the LLM

---

## Sender Protocol

Defined in [`senders/__init__.py`](senders/__init__.py).

### Sender (Protocol)

```python
class Sender(Protocol):
    name: str
    capabilities: list[str]  # e.g., ["attachments", "images"]

    async def send(self, to: str, content: str, **kwargs) -> dict:
        ...
```

---

## Related Docs

- [README.md](README.md) — Main project documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture diagrams
- [memory/README.md](memory/README.md) — Memory system details
