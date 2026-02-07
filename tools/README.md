# tools/ - Tool System

The tools subsystem provides all of BabyAGI's external capabilities. Tools are registered via a `@tool` decorator that auto-generates JSON schemas from type hints, or as class-based `Tool` instances defined in `agent.py`.

## How Tools Work

1. The `@tool` decorator inspects a function's type hints and docstring
2. It generates a JSON schema compatible with the Claude tool-use API
3. At startup, all tool modules are imported, triggering registration
4. The agent sends the tool schemas to Claude alongside conversation messages
5. When Claude returns a `tool_use` block, the agent executes the matching function
6. The result is fed back as a `tool_result` message

## Files

| File | Purpose |
|------|---------|
| [`__init__.py`](__init__.py) | `@tool` decorator, JSON schema generation, health check system |
| [`web.py`](web.py) | `web_search` (DuckDuckGo), `browse` (Browser Use Cloud), `fetch_url` |
| [`email.py`](email.py) | `send_email`, `get_email_inbox` via AgentMail API |
| [`sandbox.py`](sandbox.py) | `execute_code` — E2B cloud sandbox for safe code execution |
| [`secrets.py`](secrets.py) | `get_secret`, `store_secret`, `request_api_key`, `update_config` — API key storage via system keyring |
| [`credentials.py`](credentials.py) | `store_credential`, `get_credential`, `list_credentials` — account/payment storage |
| [`skills.py`](skills.py) | `learn_skill`, `acquire_skill`, `activate_skill`, `enable_composio` |
| [`meeting.py`](meeting.py) | `join_meeting`, `meeting_status` — Recall.ai meeting transcription |
| [`research_agent.py`](research_agent.py) | `research_task`, `research_status` — delegated research objectives |
| [`research.py`](research.py) | Internal research tools: `data_collection`, `batch_next`, `checkpoint`, etc. |
| [`metrics.py`](metrics.py) | `get_cost_summary` — session cost tracking |
| [`verbose.py`](verbose.py) | `set_verbose` — runtime verbosity control |
| [`files/`](files/) | File processing, creation, and storage subsystem |

## Core Tools (defined in agent.py)

These tools are built directly into the agent rather than using the `@tool` decorator:

| Tool | Actions | Purpose |
|------|---------|---------|
| `memory` | store, search, list, find_entity, find_relationship, get_summary, deep_search, list_tools, tool_stats | Persistent knowledge management |
| `objective` | spawn, list, check, cancel | Background task management |
| `notes` | add, list, complete, remove | Simple todo list |
| `schedule` | add, list, get, update, remove, run, history | Time-based task automation |
| `register_tool` | (direct) | Dynamic tool creation at runtime |
| `send_message` | (direct) | Send output to any channel (email, SMS, CLI) |

## The @tool Decorator

```python
from tools import tool

# Simple usage — schema auto-generated from type hints
@tool
def my_tool(query: str, limit: int = 10) -> dict:
    """Search for something.

    Args:
        query: What to search for
        limit: Max results to return
    """
    return {"results": [...]}

# With dependency metadata
@tool(packages=["pandas"], env=["DATA_API_KEY"])
def analyze_data(dataset: str) -> dict:
    """Analyze a dataset."""
    import pandas as pd
    ...
```

The decorator extracts:
- **Name** from the function name (or `name=` override)
- **Description** from the docstring's first paragraph
- **Parameter schemas** from type hints (`str` -> `string`, `int` -> `integer`, etc.)
- **Parameter descriptions** from the `Args:` section of the docstring
- **Required vs optional** from whether parameters have defaults

## Health Check System

Every tool can declare its requirements. The health system checks these at startup:

```python
from tools import check_tool_health, get_health_summary

health = check_tool_health()
# -> {"ready": [...], "needs_setup": [...], "unavailable": [...]}

summary = get_health_summary()
# -> "Ready: memory, objective, notes, web_search\nNeed API keys: browse(BROWSER_USE_API_KEY)"
```

## tools/files/ Subfolder

| File | Purpose |
|------|---------|
| [`files/__init__.py`](files/__init__.py) | Package init |
| [`files/processor.py`](files/processor.py) | File reading, parsing, content extraction |
| [`files/creators.py`](files/creators.py) | File generation (reports, exports) |
| [`files/storage.py`](files/storage.py) | File persistence and retrieval |

## Dynamic Tool Creation

The `register_tool` core tool allows the agent to create new tools at runtime. These are:
- Persisted to SQLite (`tool_definitions` table) and survive restarts
- Automatically sandboxed via E2B if they require external packages
- Tracked with usage statistics (success rate, avg duration, error count)

## Related Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Tool system architecture diagram
- [RUNNING.md](../RUNNING.md) — Examples of using tools from the CLI
- [MODELS.md](../MODELS.md) — `ToolDefinition` data model
