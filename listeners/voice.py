"""
Voice Listener - Speech input/output.

Provides voice interaction with the agent.
This is a basic implementation - extend with wake word detection,
better STT/TTS providers, streaming, etc.

Requires:
    pip install sounddevice numpy openai-whisper pyttsx3
    # Or use cloud STT/TTS services
"""

import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


async def run_voice_listener(agent, config: dict = None):
    """Run the voice listener.

    Args:
        agent: The Agent instance
        config: Configuration dict with:
            - wake_word: Wake word to listen for (default: None = always listen)
            - silence_threshold: Seconds of silence to end recording (default: 2)
            - stt_provider: "whisper" (local) or "openai" (cloud)
            - tts_provider: "pyttsx3" (local) or "openai" (cloud)
    """
    config = config or {}

    # Check for required packages
    try:
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        logger.warning("Voice listener disabled: sounddevice/numpy not installed")
        logger.info("Install with: pip install sounddevice numpy")
        return

    # Session ID for this voice session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread_id = f"voice:{session_id}"

    logger.info(f"Voice listener started (session: {session_id})")
    print("\n[Voice] Listening... (Ctrl+C to stop)\n")

    while True:
        try:
            # Record audio
            audio = await _record_audio(config)
            if audio is None:
                continue

            # Transcribe
            text = await _transcribe(audio, config)
            if not text or not text.strip():
                continue

            print(f"[Voice] You: {text}")

            # Process through agent
            response = await agent.run_async(
                text,
                thread_id=thread_id,
                context={
                    "channel": "voice",
                    "is_owner": True,
                    "session_id": session_id,
                }
            )

            print(f"[Voice] Assistant: {response}")

            # Speak response
            await _speak(response, config)

        except asyncio.CancelledError:
            logger.info("Voice listener stopped")
            break
        except Exception as e:
            logger.error(f"Voice listener error: {e}")
            await asyncio.sleep(1)


async def _record_audio(config: dict) -> bytes | None:
    """Record audio until silence detected.

    Returns audio data as bytes, or None if no speech detected.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return None

    sample_rate = config.get("sample_rate", 16000)
    silence_threshold = config.get("silence_threshold", 2)
    min_duration = config.get("min_duration", 0.5)

    # Simple recording - record for a fixed duration
    # TODO: Implement silence detection for automatic stop
    duration = config.get("max_duration", 10)

    try:
        print("[Voice] Recording...")
        audio = await asyncio.to_thread(
            sd.rec,
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        await asyncio.to_thread(sd.wait)

        # Convert to bytes
        return audio.tobytes()

    except Exception as e:
        logger.error(f"Recording error: {e}")
        return None


async def _transcribe(audio: bytes, config: dict) -> str | None:
    """Transcribe audio to text.

    Uses local Whisper by default, or OpenAI API if configured.
    """
    provider = config.get("stt_provider", "whisper")

    if provider == "openai":
        return await _transcribe_openai(audio, config)
    else:
        return await _transcribe_whisper(audio, config)


async def _transcribe_whisper(audio: bytes, config: dict) -> str | None:
    """Transcribe using local Whisper model."""
    try:
        import whisper
        import numpy as np
        import tempfile
        import scipy.io.wavfile as wav
    except ImportError:
        logger.warning("Whisper not installed. Install with: pip install openai-whisper")
        return None

    try:
        # Load model (cached after first load)
        model_name = config.get("whisper_model", "base")
        model = whisper.load_model(model_name)

        # Convert bytes back to numpy array
        sample_rate = config.get("sample_rate", 16000)
        audio_np = np.frombuffer(audio, dtype=np.float32)

        # Save to temp file (Whisper expects a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, audio_np)
            result = await asyncio.to_thread(model.transcribe, f.name)

        return result.get("text", "")

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return None


async def _transcribe_openai(audio: bytes, config: dict) -> str | None:
    """Transcribe using OpenAI Whisper API."""
    try:
        import openai
        import tempfile
        import numpy as np
        import scipy.io.wavfile as wav
    except ImportError:
        logger.warning("OpenAI package not installed")
        return None

    try:
        client = openai.OpenAI()

        # Save audio to temp file
        sample_rate = config.get("sample_rate", 16000)
        audio_np = np.frombuffer(audio, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, sample_rate, audio_np)

            with open(f.name, "rb") as audio_file:
                result = await asyncio.to_thread(
                    client.audio.transcriptions.create,
                    model="whisper-1",
                    file=audio_file
                )

        return result.text

    except Exception as e:
        logger.error(f"OpenAI transcription error: {e}")
        return None


async def _speak(text: str, config: dict):
    """Convert text to speech and play it.

    Uses pyttsx3 by default, or OpenAI TTS if configured.
    """
    provider = config.get("tts_provider", "pyttsx3")

    if provider == "openai":
        await _speak_openai(text, config)
    else:
        await _speak_pyttsx3(text, config)


async def _speak_pyttsx3(text: str, config: dict):
    """Speak using pyttsx3 (local TTS)."""
    try:
        import pyttsx3
    except ImportError:
        logger.warning("pyttsx3 not installed. Install with: pip install pyttsx3")
        return

    try:
        engine = pyttsx3.init()
        rate = config.get("speech_rate", 175)
        engine.setProperty("rate", rate)

        await asyncio.to_thread(engine.say, text)
        await asyncio.to_thread(engine.runAndWait)

    except Exception as e:
        logger.error(f"TTS error: {e}")


async def _speak_openai(text: str, config: dict):
    """Speak using OpenAI TTS API."""
    try:
        import openai
        import tempfile
        import subprocess
    except ImportError:
        logger.warning("OpenAI package not installed")
        return

    try:
        client = openai.OpenAI()
        voice = config.get("openai_voice", "alloy")

        response = await asyncio.to_thread(
            client.audio.speech.create,
            model="tts-1",
            voice=voice,
            input=text
        )

        # Save and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            # Play with system player
            await asyncio.to_thread(
                subprocess.run,
                ["afplay", f.name],  # macOS
                capture_output=True
            )

    except Exception as e:
        logger.error(f"OpenAI TTS error: {e}")
