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

## Voice Listener

The voice listener provides hands-free speech interaction with the agent. It records audio from your microphone, transcribes it to text, sends it to the agent, and speaks the response back.

### Setup

1. Install voice dependencies:

```bash
pip install babyagi[voice]
# Or manually: pip install sounddevice numpy openai-whisper pyttsx3 scipy
```

2. Enable voice in `config.yaml`:

```yaml
channels:
  voice:
    enabled: true
```

3. Run with voice enabled:

```bash
python main.py              # Runs all enabled channels including voice
python main.py channels     # Same, explicit channel mode
python main.py all          # Voice + API server + all channels
```

### How It Works

```
Microphone → Record (sounddevice) → Transcribe (Whisper) → Agent → TTS → Speaker
```

1. **Recording**: Listens on your default microphone. Recording starts immediately and stops automatically when silence is detected (2 seconds of quiet after speech). Maximum recording duration is 10 seconds.
2. **Transcription**: Converts speech to text using either local Whisper or the OpenAI Whisper API.
3. **Agent processing**: The transcribed text is sent to the agent as an owner message on the `voice` channel.
4. **Text-to-speech**: The agent's response is spoken aloud using either pyttsx3 (local, no API key) or OpenAI TTS.
5. **Repeat**: The listener loops back to step 1.

### Configuration

All voice settings go under `channels.voice` in `config.yaml`:

```yaml
channels:
  voice:
    enabled: false              # Set to true to enable
    wake_word: "hey assistant"  # Not yet implemented
    stt_provider: "whisper"     # "whisper" (local) or "openai" (cloud API)
    tts_provider: "pyttsx3"     # "pyttsx3" (local) or "openai" (cloud API)
    whisper_model: "base"       # tiny, base, small, medium, large
    sample_rate: 16000          # Audio sample rate in Hz
    max_duration: 10            # Max recording length in seconds
    silence_threshold: 2.0      # Seconds of silence before recording stops
    min_duration: 0.5           # Minimum speech duration to process
    energy_threshold: 0.01      # RMS energy level below which audio counts as silence
    speech_rate: 175            # Words per minute for pyttsx3
    openai_voice: "alloy"       # Voice for OpenAI TTS (alloy/echo/fable/onyx/nova/shimmer)
```

### Provider Options

**Speech-to-text (stt_provider)**:
- `"whisper"` (default) — Runs OpenAI's Whisper model locally. No API key needed but requires more CPU/RAM. Model size (`whisper_model`) trades accuracy for speed.
- `"openai"` — Uses the OpenAI Whisper API. Requires `OPENAI_API_KEY`. Faster on low-power machines.

**Text-to-speech (tts_provider)**:
- `"pyttsx3"` (default) — Local TTS using system speech engines. No API key needed. Works on all platforms.
- `"openai"` — Uses the OpenAI TTS API for higher-quality voices. Requires `OPENAI_API_KEY` and a system audio player (see below).

### Platform Support

Recording and local TTS (pyttsx3) work on all platforms. OpenAI TTS playback requires a system audio player:

| Platform | Player | Install |
|----------|--------|---------|
| macOS | `afplay` | Built-in |
| Linux (PulseAudio) | `paplay` | `sudo apt install pulseaudio-utils` |
| Linux (ALSA) | `aplay` | `sudo apt install alsa-utils` |
| Linux (FFmpeg) | `ffplay` | `sudo apt install ffmpeg` |
| Windows | PowerShell | Built-in |

If no player is found, the listener logs a warning and skips playback. Switch to `tts_provider: "pyttsx3"` for guaranteed cross-platform local playback.

### Silence Detection

The listener uses energy-based silence detection. It records audio in 100ms chunks and calculates the RMS energy of each chunk. When speech is detected (energy above `energy_threshold`), it continues recording. Once the energy drops below the threshold for `silence_threshold` seconds, recording stops automatically. This avoids the old behavior of always waiting the full `max_duration`.

If your environment is noisy, increase `energy_threshold`. If the listener cuts off speech too early, increase `silence_threshold` or decrease `energy_threshold`.

### Console Output

```
[Voice] Listening... (Ctrl+C to stop)

[Voice] Recording... (speak now, stops on silence)
[Voice] You: What's on my schedule today?
[Voice] Assistant: You have a meeting at 2pm with the engineering team.
```

## Related Docs

- [`senders/README.md`](../senders/README.md) — Output channels (the other half)
- [ARCHITECTURE.md](../ARCHITECTURE.md) — Multi-channel architecture diagram
