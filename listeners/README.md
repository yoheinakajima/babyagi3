# listeners/ - Input Channels

Listeners are async functions that receive messages from external sources and route them to the agent. Each listener follows the same pattern: wait for input, call `agent.run_async()`, and handle the response.

There is no base class — just a convention:

```python
async def run_<channel>_listener(agent, config: dict = None):
    while True:
        message = await poll_for_input()
        response = await agent.run_async(
            content=message.text,
            thread_id=f"<channel>:{message.sender}",
            context={
                "channel": "<channel>",
                "is_owner": <bool>,
                "sender": "<identifier>",
            },
        )
```

## Files

| File | Purpose | Requires |
|------|---------|----------|
| [`__init__.py`](__init__.py) | Re-exports all listeners | — |
| [`cli.py`](cli.py) | Terminal REPL — reads stdin, displays styled output, handles `/verbose` command | Always available |
| [`email.py`](email.py) | Polls AgentMail inbox on an interval, creates per-sender threads | `AGENTMAIL_API_KEY` |
| [`voice.py`](voice.py) | Listens for wake word, records audio, transcribes with Whisper, plays TTS response | `sounddevice`, `openai-whisper`, `pyttsx3` |
| [`sendblue.py`](sendblue.py) | Polls SendBlue API for new SMS/iMessage messages | `SENDBLUE_API_KEY`, `SENDBLUE_API_SECRET` |

## Context Object

Every listener passes a `context` dict to `agent.run_async()`:

| Field | Type | Purpose |
|-------|------|---------|
| `channel` | `str` | Which channel this came from (`"cli"`, `"email"`, `"sendblue"`, etc.) |
| `is_owner` | `bool` | Whether the message is from the agent's owner |
| `sender` | `str` | Identifier for the sender (email address, phone number, etc.) |

The agent uses `is_owner` to decide how to respond. Owner messages get full access; external messages follow the `external_policy` behavior settings in `config.yaml`.

## Adding a New Listener

1. Create `listeners/<channel>.py` with an `async def run_<channel>_listener(agent, config)` function
2. Register it in `main.py` under the `run_all_channels()` function
3. Add the channel to `config.yaml` under `channels:`
4. Optionally create a matching sender in `senders/`

See the [Extending the System](../README.md#extending-the-system) section in the main README for a complete example.

## Related Docs

- [`senders/README.md`](../senders/README.md) — Output channels (the other half)
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Multi-channel architecture diagram
