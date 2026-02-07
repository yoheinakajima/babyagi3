import base64
import os

from tools import tool, tool_error


@tool(env=["ELEVENLABS_API_KEY"], packages=["httpx"])
def elevenlabs_text_to_speech(text: str, voice_id: str = "EXAVITQu4vr4xnSDxMaL", model_id: str = "eleven_multilingual_v2") -> dict:
    """Generate speech with ElevenLabs and return audio as base64."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return tool_error("ELEVENLABS_API_KEY is not set")

    try:
        import httpx
    except ImportError:
        return tool_error("httpx not installed", fix="pip install httpx")

    try:
        with httpx.Client(timeout=60) as client:
            response = client.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={
                    "xi-api-key": api_key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                },
                json={"text": text, "model_id": model_id},
            )
    except Exception as exc:
        return tool_error(f"Request failed: {exc}")

    if response.status_code >= 400:
        return tool_error(
            f"Request failed with status {response.status_code}",
            details=response.text,
        )

    return {
        "status_code": response.status_code,
        "audio_base64": base64.b64encode(response.content).decode("utf-8"),
        "mime_type": "audio/mpeg",
    }
