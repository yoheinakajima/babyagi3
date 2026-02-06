# utils/ - Shared Utilities

Small, reusable modules used across the codebase.

## Files

| File | Purpose |
|------|---------|
| [`__init__.py`](__init__.py) | Re-exports `EventEmitter`, `console`, `VerboseLevel` |
| [`events.py`](events.py) | `EventEmitter` mixin — lightweight pub/sub for decoupled communication |
| [`console.py`](console.py) | Styled terminal output — color-coded messages, verbose levels, event logging |
| [`collections.py`](collections.py) | `ThreadSafeList` — list wrapper with `RLock` for concurrent access |
| [`email_client.py`](email_client.py) | Email utilities shared between listeners and senders |

## EventEmitter

The `EventEmitter` is a mixin class that the `Agent` inherits from. It provides a simple pub/sub pattern:

```python
from utils.events import EventEmitter

class Agent(EventEmitter):
    ...

# Subscribe
agent.on("tool_start", lambda e: print(f"Running {e['name']}"))
agent.on("*", lambda e: log(e))  # Wildcard: all events

# Emit
agent.emit("tool_start", {"name": "memory", "input": {...}})

# One-time subscription
agent.once("objective_end", handle_completion)

# Unsubscribe
agent.off("tool_start", handler)
```

## Console

Color-coded terminal output with three verbose levels:

| Level | Shows |
|-------|-------|
| `off` | Only user/agent messages |
| `light` | Key operations: tool names, task starts |
| `deep` | Everything: tool inputs/outputs, full details |

```python
from utils.console import console, VerboseLevel

console.user("Hello")       # Green
console.agent("Response")   # Cyan
console.system("Info")      # Blue
console.error("Failed")     # Red

console.set_verbose("light")
console.verbose("Detail", level=VerboseLevel.LIGHT)
```

## Related Docs

- [ARCHITECTURE.md](../ARCHITECTURE.md) — Event system diagram
