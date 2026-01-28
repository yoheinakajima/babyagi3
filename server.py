"""
API Server for the Agent

Provides HTTP endpoints for:
- Receiving messages (webhooks)
- Checking objective status
- Managing threads

This enables external systems to interact with the agent via HTTP.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

from agent import Agent, Objective


# =============================================================================
# Shared Agent Instance
# =============================================================================

agent = Agent()
scheduler_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, clean up on shutdown."""
    global scheduler_task
    scheduler_task = asyncio.create_task(agent.run_scheduler())
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
# Run
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
