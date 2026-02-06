# senders/ - Output Channels

Senders handle outbound communication from the agent. Unlike listeners (which are diverse async functions), senders follow a strict protocol defined in `__init__.py`.

## The Sender Protocol

```python
@runtime_checkable
class Sender(Protocol):
    name: str                          # Channel identifier
    capabilities: list[str]            # e.g., ["attachments", "images"]

    async def send(self, to: str, content: str, **kwargs) -> dict:
        # to: email address, phone number, or "owner"
        # Returns {"sent": True, ...} or {"error": "..."}
        ...
```

## Files

| File | Purpose | Requires |
|------|---------|----------|
| [`__init__.py`](__init__.py) | `Sender` protocol definition | — |
| [`cli.py`](cli.py) | Prints styled output to the terminal | Always available |
| [`email.py`](email.py) | Sends email via AgentMail API, supports HTML and attachments | `AGENTMAIL_API_KEY` |
| [`sendblue.py`](sendblue.py) | Sends SMS/iMessage via SendBlue API | `SENDBLUE_API_KEY`, `SENDBLUE_API_SECRET` |

## How Senders Are Used

The agent communicates outbound **only** through the `send_message` tool. This tool looks up the appropriate sender from `agent.senders` and calls `sender.send()`.

```
Agent decides to send a message
    -> calls send_message(channel="email", to="user@example.com", content="...")
    -> agent.senders["email"].send("user@example.com", "...")
    -> EmailSender posts to AgentMail API
```

## Registration

Senders are registered in `main.py` via `_register_senders()`:

```python
from senders.cli import CLISender
from senders.email import EmailSender

agent.register_sender("cli", CLISender())
agent.register_sender("email", EmailSender(config))
```

## Adding a New Sender

1. Create `senders/<channel>.py` implementing the `Sender` protocol
2. Register it in `main.py`'s `_register_senders()` function
3. The agent can now use `send_message(channel="<channel>", ...)` to reach it

## Related Docs

- [`listeners/README.md`](../listeners/README.md) — Input channels (the other half)
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Multi-channel architecture diagram
