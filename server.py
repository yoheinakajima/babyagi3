"""
API Server for the Agent

Provides HTTP endpoints for:
- Receiving messages (webhooks)
- SendBlue SMS/iMessage webhooks
- Checking objective status
- Managing threads

This enables external systems to interact with the agent via HTTP.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Request, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from agent import Agent, Objective
from utils.console import console

logger = logging.getLogger(__name__)


# =============================================================================
# Shared Agent Instance
# =============================================================================

agent = Agent()
scheduler_task: asyncio.Task | None = None


def _register_server_senders():
    """Register senders for webhook replies."""
    # SendBlue sender (for auto-reply to webhook messages)
    if os.environ.get("SENDBLUE_API_KEY") and os.environ.get("SENDBLUE_API_SECRET"):
        try:
            from senders.sendblue import SendBlueSender
            sendblue_config = {
                "api_key": os.environ.get("SENDBLUE_API_KEY"),
                "api_secret": os.environ.get("SENDBLUE_API_SECRET"),
                "from_number": os.environ.get("SENDBLUE_PHONE_NUMBER"),
            }
            agent.register_sender("sendblue", SendBlueSender(sendblue_config))
            logger.debug("SendBlue sender registered for webhooks")
        except Exception as e:
            logger.error(f"Failed to register SendBlue sender: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, register senders, clean up on shutdown."""
    global scheduler_task
    _register_server_senders()
    scheduler_task = asyncio.create_task(agent.run_scheduler())
    logger.debug("Server started with SendBlue webhook at /webhooks/sendblue")
    yield
    if scheduler_task:
        scheduler_task.cancel()


app = FastAPI(
    title="Agent API",
    description="HTTP interface for the agent with background objectives",
    lifespan=lifespan
)


# =============================================================================
# Request/Response Models
# =============================================================================

class MessageRequest(BaseModel):
    """Incoming message (user input or webhook payload)."""
    content: str
    thread_id: str = "main"
    async_mode: bool = False  # If true, return immediately and process in background


class MessageResponse(BaseModel):
    """Response to a message."""
    response: str | None = None
    thread_id: str
    queued: bool = False  # True if processing in background


class ObjectiveResponse(BaseModel):
    """Objective details."""
    id: str
    goal: str
    status: str
    schedule: str | None
    result: str | None
    error: str | None


class SendBlueWebhookPayload(BaseModel):
    """SendBlue webhook payload for inbound messages.

    See: https://docs.sendblue.com/getting-started/webhooks/
    """
    message_handle: Optional[str] = None
    from_number: Optional[str] = None
    to_number: Optional[str] = None
    content: Optional[str] = None
    media_url: Optional[str] = None
    date_sent: Optional[str] = None
    date_created: Optional[str] = None
    is_outbound: Optional[bool] = False
    was_downgraded: Optional[bool] = None
    status: Optional[str] = None
    error_code: Optional[int] = None
    error_message: Optional[str] = None
    # Additional fields that may be present
    account_email: Optional[str] = None
    group_id: Optional[str] = None


# =============================================================================
# Recall.ai Webhook Models
# =============================================================================

class RecallTranscriptWord(BaseModel):
    """A word in the transcript."""
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    confidence: Optional[float] = None


class RecallTranscriptData(BaseModel):
    """Real-time transcript data from Recall.ai."""
    bot_id: Optional[str] = None
    transcript: Optional[dict] = None
    # Transcript contains: speaker, speaker_id, words[], is_final


class RecallBotStatusChange(BaseModel):
    """Bot status change webhook from Recall.ai."""
    event: Optional[str] = None  # "bot.status_change"
    data: Optional[dict] = None
    # data contains: bot_id, status, status_changes[]


class RecallTranscriptDone(BaseModel):
    """Transcript done webhook from Recall.ai."""
    event: Optional[str] = None  # "transcript.done"
    data: Optional[dict] = None
    # data contains: bot_id, transcript_id


# =============================================================================
# Endpoints
# =============================================================================

@app.post("/message", response_model=MessageResponse)
async def receive_message(req: MessageRequest, background_tasks: BackgroundTasks):
    """
    Receive a message and process it.

    This is the main endpoint for:
    - User chat messages
    - Webhooks from external services
    - Automated triggers

    If async_mode=true, queues the message and returns immediately.
    """
    if req.async_mode:
        background_tasks.add_task(agent.run_async, req.content, req.thread_id)
        return MessageResponse(thread_id=req.thread_id, queued=True)

    response = await agent.run_async(req.content, req.thread_id)
    return MessageResponse(response=response, thread_id=req.thread_id)


@app.get("/objectives")
async def list_objectives() -> list[ObjectiveResponse]:
    """List all objectives and their status."""
    return [
        ObjectiveResponse(
            id=obj.id,
            goal=obj.goal,
            status=obj.status,
            schedule=obj.schedule,
            result=obj.result,
            error=obj.error
        )
        for obj in agent.objectives.values()
    ]


@app.get("/objectives/{objective_id}")
async def get_objective(objective_id: str) -> ObjectiveResponse:
    """Get details of a specific objective."""
    obj = agent.objectives.get(objective_id)
    if not obj:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Objective not found")

    return ObjectiveResponse(
        id=obj.id,
        goal=obj.goal,
        status=obj.status,
        schedule=obj.schedule,
        result=obj.result,
        error=obj.error
    )


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get message history for a thread."""
    return {"thread_id": thread_id, "messages": agent.get_thread(thread_id)}


@app.delete("/threads/{thread_id}")
async def clear_thread(thread_id: str):
    """Clear a thread's message history."""
    agent.clear_thread(thread_id)
    return {"cleared": thread_id}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "objectives_count": len(agent.objectives),
        "threads_count": len(agent.threads),
        "tools": list(agent.tools.keys())
    }


# =============================================================================
# SendBlue Webhook
# =============================================================================

# Track processed message IDs to prevent duplicates
_sendblue_processed_ids: set[str] = set()


def _normalize_phone(phone: str) -> str:
    """Normalize phone number for comparison."""
    if not phone:
        return ""
    cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    if not cleaned.startswith("+"):
        if cleaned.startswith("1") and len(cleaned) == 11:
            cleaned = "+" + cleaned
        elif len(cleaned) == 10:
            cleaned = "+1" + cleaned
    return cleaned.lower()


@app.post("/webhooks/sendblue")
async def sendblue_webhook(
    payload: SendBlueWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Receive inbound SMS/iMessage from SendBlue.

    Configure this URL in your SendBlue dashboard:
    https://your-domain.com/webhooks/sendblue

    SendBlue will POST to this endpoint when messages are received.
    """
    global _sendblue_processed_ids

    # Log the incoming webhook
    console.activity("sendblue", f"inbound from {payload.from_number}")

    # Get message ID for deduplication
    msg_id = payload.message_handle or f"{payload.from_number}:{payload.date_sent}"

    # Skip if already processed (webhooks can be sent multiple times)
    if msg_id in _sendblue_processed_ids:
        logger.debug(f"SendBlue webhook: skipping duplicate message {msg_id}")
        return {"status": "ok", "message": "duplicate"}

    # Skip outbound messages (this webhook is for inbound)
    if payload.is_outbound:
        logger.debug(f"SendBlue webhook: skipping outbound message {msg_id}")
        return {"status": "ok", "message": "outbound_ignored"}

    # Skip empty messages
    if not payload.content and not payload.media_url:
        logger.debug(f"SendBlue webhook: skipping empty message {msg_id}")
        return {"status": "ok", "message": "empty_ignored"}

    # Mark as processed
    _sendblue_processed_ids.add(msg_id)

    # Limit processed IDs cache size (keep last 1000)
    if len(_sendblue_processed_ids) > 1000:
        _sendblue_processed_ids = set(list(_sendblue_processed_ids)[-500:])

    # Get owner phone for comparison
    owner_phone = os.environ.get("OWNER_PHONE", "")
    owner_phone_normalized = _normalize_phone(owner_phone)
    from_number_normalized = _normalize_phone(payload.from_number or "")

    # Determine if owner
    is_owner = bool(
        owner_phone_normalized and
        from_number_normalized == owner_phone_normalized
    )

    # Build thread ID
    if is_owner:
        thread_id = "sendblue:owner"
    else:
        thread_id = f"sendblue:{from_number_normalized}"

    # Build context
    context = {
        "channel": "sendblue",
        "is_owner": is_owner,
        "sender": payload.from_number,
        "message_id": msg_id,
    }

    # Format input with message context
    sender_type = "Owner" if is_owner else "External"
    message_input = f"[Text from {sender_type}: {payload.from_number}]\n\n{payload.content or ''}"

    if payload.media_url:
        message_input += f"\n\n[Media attached: {payload.media_url}]"

    logger.debug(f"Processing SendBlue message from {payload.from_number} (owner={is_owner})")

    # Process in background to return 200 quickly (SendBlue requires fast response)
    async def process_and_reply():
        try:
            response_text = await agent.run_async(
                message_input,
                thread_id=thread_id,
                context=context
            )

            # Auto-reply for owner messages
            if is_owner and response_text:
                if "sendblue" in agent.senders:
                    await agent.senders["sendblue"].send(
                        to=payload.from_number,
                        content=response_text
                    )
                    console.activity("sendblue", f"replied to {payload.from_number}")
                else:
                    logger.warning("SendBlue sender not registered, cannot auto-reply")

        except Exception as e:
            logger.error(f"Error processing SendBlue message: {e}")

    background_tasks.add_task(process_and_reply)

    return {"status": "ok", "message": "processing"}


# =============================================================================
# Recall.ai Webhooks
# =============================================================================

@app.websocket("/webhooks/recall/realtime")
async def recall_realtime_websocket(websocket: WebSocket):
    """
    Receive real-time transcript data from Recall.ai via WebSocket.

    Recall.ai connects to this endpoint and streams transcript chunks
    as JSON messages during the meeting.
    """
    await websocket.accept()
    console.activity("recall", "realtime WebSocket connected")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except Exception as e:
                logger.error(f"Failed to parse Recall realtime WS message: {e}")
                continue

            event_type = payload.get("event", "")
            data = payload.get("data", {})

            if event_type in ("transcript.data", "transcript.partial_data"):
                transcript_data = data.get("data", {})
                bot_info = data.get("bot", {})
                bot_id = bot_info.get("id", "")

                participant = transcript_data.get("participant", {})
                words = transcript_data.get("words", [])
                speaker = participant.get("name", "unknown")
                text = " ".join(w.get("text", "") for w in words)

                if bot_id and text:
                    logger.debug(f"Recall realtime [{event_type}] bot={bot_id}: {speaker}: {text[:100]}")
                    try:
                        from tools.meeting import get_meeting_processor
                        processor = get_meeting_processor()
                        segment = {
                            "speaker": speaker,
                            "speaker_id": participant.get("id"),
                            "words": words,
                            "is_final": event_type == "transcript.data",
                        }
                        processor.add_transcript_segment(bot_id, segment)
                    except Exception as e:
                        logger.error(f"Error processing realtime transcript: {e}")
            else:
                logger.debug(f"Recall realtime WS event: {event_type}")

    except WebSocketDisconnect:
        console.activity("recall", "realtime WebSocket disconnected")
    except Exception as e:
        logger.error(f"Recall realtime WebSocket error: {e}")


@app.post("/webhooks/recall/realtime")
async def recall_realtime_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Fallback POST handler for real-time transcript data from Recall.ai.
    Kept for backward compatibility if webhook type is used.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse Recall realtime webhook: {e}")
        return {"status": "error", "message": "Invalid JSON"}

    bot_id = payload.get("bot_id")
    transcript = payload.get("transcript", {})

    if not bot_id or not transcript:
        return {"status": "ok", "message": "no_data"}

    logger.debug(f"Recall realtime transcript for bot {bot_id}: {transcript.get('speaker', 'unknown')}")

    async def process_realtime():
        try:
            from tools.meeting import get_meeting_processor
            processor = get_meeting_processor()
            processor.add_transcript_segment(bot_id, transcript)
        except Exception as e:
            logger.error(f"Error processing realtime transcript: {e}")

    background_tasks.add_task(process_realtime)

    return {"status": "ok"}


@app.post("/webhooks/recall/status")
async def recall_status_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    Receive bot status change events from Recall.ai.

    Status flow: ready → joining_call → in_waiting_room → in_call_recording → call_ended → done

    Configure this URL in your Recall.ai dashboard webhook settings.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse Recall status webhook: {e}")
        return {"status": "error", "message": "Invalid JSON"}

    event_type = payload.get("event")
    data = payload.get("data", {})
    bot_id = data.get("bot_id")
    status = data.get("status", {}).get("code")

    bot_short = (bot_id[:8] + "…") if bot_id else "unknown"
    console.activity("recall", f"bot {bot_short} → {status or 'unknown'}")

    # Handle meeting end - trigger post-meeting processing
    if status in ("call_ended", "done", "analysis_done"):
        async def process_meeting_end():
            try:
                from tools.meeting import get_meeting_processor, RecallClient

                processor = get_meeting_processor()
                client = RecallClient()

                # Fetch full transcript
                transcript_data = await client.get_bot_transcript(bot_id)
                if "error" in transcript_data:
                    logger.error(f"Failed to fetch transcript for bot {bot_id}: {transcript_data}")
                    return

                # Get bot details for metadata
                bot_data = await client.get_bot(bot_id)
                meeting_metadata = {
                    "title": bot_data.get("meeting_metadata", {}).get("title", "Meeting"),
                    "platform": bot_data.get("meeting_url", "").split("/")[2] if bot_data.get("meeting_url") else None,
                    "duration_minutes": 0,  # Calculate from timestamps if available
                }

                # Process the completed meeting
                summary = await processor.process_completed_meeting(
                    bot_id=bot_id,
                    transcript_data=transcript_data,
                    meeting_metadata=meeting_metadata,
                )

                # Notify owner
                owner_phone = os.environ.get("OWNER_PHONE", "")
                if owner_phone and "sendblue" in agent.senders:
                    notification = f"Meeting ended: {meeting_metadata.get('title', 'Meeting')}\n\n"
                    notification += f"Summary: {summary.summary[:300]}..."
                    if summary.action_items:
                        notification += f"\n\nAction items: {len(summary.action_items)}"

                    try:
                        await agent.senders["sendblue"].send(
                            to=owner_phone,
                            content=notification,
                        )
                    except Exception as e:
                        logger.error(f"Failed to send meeting notification: {e}")

                console.activity("recall", f"meeting processed: {meeting_metadata.get('title', 'Meeting')}")

            except Exception as e:
                logger.error(f"Error processing meeting end for bot {bot_id}: {e}")

        background_tasks.add_task(process_meeting_end)

    return {"status": "ok"}


@app.post("/webhooks/recall")
async def recall_general_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
):
    """
    General Recall.ai webhook endpoint.

    Handles multiple event types and routes to appropriate handlers.
    Configure this URL in your Recall.ai dashboard as the main webhook.
    """
    try:
        payload = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse Recall webhook: {e}")
        return {"status": "error", "message": "Invalid JSON"}

    event_type = payload.get("event", "")
    console.activity("recall", f"webhook: {event_type}")

    # Route to appropriate handler
    if event_type == "bot.status_change":
        return await recall_status_webhook(request, background_tasks)
    elif event_type in ("transcript.data", "transcript.partial_data"):
        # Reconstruct request-like object for realtime handler
        return await recall_realtime_webhook(request, background_tasks)
    elif event_type == "transcript.done":
        # Transcript done is similar to bot status done
        data = payload.get("data", {})
        bot_id = data.get("bot_id")
        bot_short = (bot_id[:8] + "…") if bot_id else "unknown"
        console.activity("recall", f"transcript done for bot {bot_short}")
        # The status webhook will handle the actual processing when bot status changes to done

    return {"status": "ok", "event": event_type}


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
