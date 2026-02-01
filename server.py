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
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, Request, Header, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from agent import Agent, Objective

logger = logging.getLogger(__name__)


# =============================================================================
# API Key Authentication
# =============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """Verify the API key is valid. Returns the key if valid, raises 401 otherwise."""
    expected = os.environ.get("AGENT_API_KEY")
    if not expected:
        raise HTTPException(status_code=500, detail="Server API key not configured")
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


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
            logger.info("SendBlue sender registered for webhooks")
        except Exception as e:
            logger.error(f"Failed to register SendBlue sender: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, register senders, clean up on shutdown."""
    global scheduler_task
    _register_server_senders()
    scheduler_task = asyncio.create_task(agent.run_scheduler())
    logger.info("Server started with SendBlue webhook at /webhooks/sendblue/{secret}")
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
# Endpoints
# =============================================================================

@app.post("/message", response_model=MessageResponse)
async def receive_message(
    req: MessageRequest,
    background_tasks: BackgroundTasks,
    _: str = Depends(require_api_key),
):
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
async def list_objectives(_: str = Depends(require_api_key)) -> list[ObjectiveResponse]:
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
async def get_objective(objective_id: str, _: str = Depends(require_api_key)) -> ObjectiveResponse:
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
async def get_thread(thread_id: str, _: str = Depends(require_api_key)):
    """Get message history for a thread."""
    return {"thread_id": thread_id, "messages": agent.get_thread(thread_id)}


@app.delete("/threads/{thread_id}")
async def clear_thread(thread_id: str, _: str = Depends(require_api_key)):
    """Clear a thread's message history."""
    agent.clear_thread(thread_id)
    return {"cleared": thread_id}


@app.get("/health")
async def health(_: str = Depends(require_api_key)):
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


@app.post("/webhooks/sendblue/{webhook_secret}")
async def sendblue_webhook(
    webhook_secret: str,
    payload: SendBlueWebhookPayload,
    background_tasks: BackgroundTasks,
    request: Request,
):
    """
    Receive inbound SMS/iMessage from SendBlue.

    Configure this URL in your SendBlue dashboard:
    https://your-domain.com/webhooks/sendblue/YOUR_SECRET_HERE

    Generate a secret with: python -c "import secrets; print(secrets.token_urlsafe(32))"

    SendBlue will POST to this endpoint when messages are received.
    """
    # Verify webhook secret
    expected_secret = os.environ.get("SENDBLUE_WEBHOOK_SECRET")
    if not expected_secret or webhook_secret != expected_secret:
        raise HTTPException(status_code=404)  # 404 to not reveal endpoint exists
    global _sendblue_processed_ids

    # Log the incoming webhook
    logger.info(f"SendBlue webhook received: from={payload.from_number}, content_preview={payload.content[:50] if payload.content else 'empty'}...")

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

    logger.info(f"Processing SendBlue message from {payload.from_number} (owner={is_owner})")

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
                    logger.info(f"Replied to {payload.from_number}")
                else:
                    logger.warning("SendBlue sender not registered, cannot auto-reply")

        except Exception as e:
            logger.error(f"Error processing SendBlue message: {e}")

    background_tasks.add_task(process_and_reply)

    return {"status": "ok", "message": "processing"}


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
