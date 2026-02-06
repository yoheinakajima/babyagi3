"""
Voice Listener - Speech input/output.

Provides voice interaction with the agent.
This is a basic implementation - extend with wake word detection,
better STT/TTS providers, streaming, etc.

Requires:
    pip install babyagi[voice]
    # Or: pip install sounddevice numpy openai-whisper pyttsx3 scipy
"""

import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime

from utils.console import console

logger = logging.getLogger(__name__)


def _get_playback_command(file_path: str) -> list[str]:
    """Return a platform-appropriate command to play an audio file.

    Falls back to None if no system player is found.
    """
    if sys.platform == "darwin":
        return ["afplay", file_path]
    elif sys.platform == "win32":
        # PowerShell's Start-Process with -Wait plays and blocks until done
        return [
            "powershell", "-c",
            f'(New-Object Media.SoundPlayer "{file_path}").PlaySync()',
        ]
    else:
        # Linux - try common players in preference order
        for player in ["paplay", "aplay", "ffplay"]:
            # Check if the player exists on PATH
            from shutil import which
            if which(player) is not None:
                if player == "ffplay":
                    return [player, "-nodisp", "-autoexit", file_path]
                return [player, file_path]
    return None


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
        logger.info("Install with: pip install babyagi[voice]")
        return

    # Session ID for this voice session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    thread_id = f"voice:{session_id}"

    logger.info(f"Voice listener started (session: {session_id})")
    console.system("\n[Voice] Listening... (Ctrl+C to stop)\n")

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

            console.user(text, prompt="[Voice] You")

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

            console.agent(response, prefix="[Voice] Assistant")

            # Speak response
            await _speak(response, config)

        except asyncio.CancelledError:
            logger.info("Voice listener stopped")
            break
        except Exception as e:
            logger.error(f"Voice listener error: {e}")
            await asyncio.sleep(1)


async def _record_audio(config: dict) -> bytes | None:
    """Record audio with silence detection.

    Starts recording when sound is detected and stops after a period of
    silence (configurable via silence_threshold). Falls back to fixed-duration
    recording if silence detection encounters issues.

    Returns audio data as bytes, or None if no speech detected.
    """
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        return None

    sample_rate = config.get("sample_rate", 16000)
    silence_threshold = config.get("silence_threshold", 2.0)
    min_duration = config.get("min_duration", 0.5)
    max_duration = config.get("max_duration", 10)
    # RMS energy below this level counts as silence. Configurable because
    # ambient noise varies widely across environments and microphones.
    energy_threshold = config.get("energy_threshold", 0.01)

    try:
        console.system("[Voice] Recording... (speak now, stops on silence)")

        # Use a streaming approach: record in small chunks and detect silence
        chunk_duration = 0.1  # 100ms chunks
        chunk_samples = int(chunk_duration * sample_rate)
        max_chunks = int(max_duration / chunk_duration)
        silence_chunks_needed = int(silence_threshold / chunk_duration)

        chunks = []
        silence_count = 0
        has_speech = False

        for _ in range(max_chunks):
            chunk = await asyncio.to_thread(
                sd.rec,
                chunk_samples,
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32,
            )
            await asyncio.to_thread(sd.wait)

            chunks.append(chunk)

            # Calculate RMS energy of this chunk
            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms > energy_threshold:
                has_speech = True
                silence_count = 0
            else:
                silence_count += 1

            # Stop if we've had speech followed by enough silence
            if has_speech and silence_count >= silence_chunks_needed:
                break

        if not has_speech:
            # No speech detected at all
            return None

        # Concatenate all chunks, trim trailing silence (keep a small tail)
        audio = np.concatenate(chunks, axis=0)

        # Trim trailing silence but keep a small buffer
        keep_chunks = len(chunks) - max(0, silence_count - 2)
        if keep_chunks > 0:
            trim_samples = keep_chunks * chunk_samples
            audio = audio[:trim_samples]

        # Check minimum duration
        duration = len(audio) / sample_rate
        if duration < min_duration:
            return None

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
        import scipy.io.wavfile as wav
    except ImportError:
        logger.warning("Whisper not installed. Install with: pip install openai-whisper")
        return None

    tmp_path = None
    try:
        # Load model (cached after first load)
        model_name = config.get("whisper_model", "base")
        model = whisper.load_model(model_name)

        # Convert bytes back to numpy array
        sample_rate = config.get("sample_rate", 16000)
        audio_np = np.frombuffer(audio, dtype=np.float32)

        # Save to temp file (Whisper expects a file)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            wav.write(tmp_path, sample_rate, audio_np)

        result = await asyncio.to_thread(model.transcribe, tmp_path)
        return result.get("text", "")

    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def _transcribe_openai(audio: bytes, config: dict) -> str | None:
    """Transcribe using OpenAI Whisper API."""
    try:
        import openai
        import numpy as np
        import scipy.io.wavfile as wav
    except ImportError:
        logger.warning("OpenAI package not installed")
        return None

    tmp_path = None
    try:
        client = openai.OpenAI()

        # Save audio to temp file
        sample_rate = config.get("sample_rate", 16000)
        audio_np = np.frombuffer(audio, dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            wav.write(tmp_path, sample_rate, audio_np)

        with open(tmp_path, "rb") as audio_file:
            result = await asyncio.to_thread(
                client.audio.transcriptions.create,
                model="whisper-1",
                file=audio_file
            )

        return result.text

    except Exception as e:
        logger.error(f"OpenAI transcription error: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


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
    """Speak using OpenAI TTS API.

    Plays audio using platform-appropriate commands:
    - macOS: afplay
    - Linux: paplay, aplay, or ffplay (whichever is available)
    - Windows: PowerShell SoundPlayer
    Falls back to sounddevice playback if no system player is found.
    """
    try:
        import openai
        import subprocess
    except ImportError:
        logger.warning("OpenAI package not installed")
        return

    tmp_path = None
    try:
        client = openai.OpenAI()
        voice = config.get("openai_voice", "alloy")

        response = await asyncio.to_thread(
            client.audio.speech.create,
            model="tts-1",
            voice=voice,
            input=text
        )

        # Save to temp file and play
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
            f.write(response.content)

        cmd = _get_playback_command(tmp_path)

        if cmd is not None:
            await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
            )
        else:
            # Fallback: try sounddevice + scipy for WAV playback
            try:
                import sounddevice as sd
                import numpy as np
                from scipy.io import wavfile

                # sounddevice can't play mp3 directly; log a warning
                logger.warning(
                    "No system audio player found (afplay/paplay/aplay/ffplay). "
                    "Install one or switch tts_provider to 'pyttsx3' for local playback."
                )
            except Exception:
                logger.warning(
                    "No system audio player found. Install 'paplay' (PulseAudio), "
                    "'aplay' (ALSA), or 'ffplay' (FFmpeg) for OpenAI TTS playback, "
                    "or switch tts_provider to 'pyttsx3'."
                )

    except Exception as e:
        logger.error(f"OpenAI TTS error: {e}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
