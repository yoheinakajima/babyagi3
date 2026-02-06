"""
Unified Message Abstraction Agent with Background Objectives

Core insight: Everything is still a message, but some messages trigger background work.

- User input → message
- Assistant response → message
- Tool execution → message
- Background objective → messages in its own thread

The elegance: objectives are just agent runs in separate threads.
Chat continues while objectives work. Objectives can spawn sub-objectives.
"""

import asyncio
import heapq
import inspect
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

import anthropic

from metrics import (
    InstrumentedAsyncAnthropic,
    AsyncLiteLLMAnthropicAdapter,
    MetricsCollector,
    set_event_emitter,
    track_source,
    get_model_for_use_case,
)
from llm_config import get_llm_config, init_llm_config
from scheduler import (
    Scheduler, ScheduledTask, Schedule, SchedulerStore,
    create_task, parse_schedule, RunRecord
)
from utils.events import EventEmitter
from utils.console import console, VerboseLevel
from utils.collections import ThreadSafeList


def json_serialize(obj):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# =============================================================================
# Tool Validation
# =============================================================================

class ToolValidationError(Exception):
    """Raised when a tool fails validation during registration."""
    pass


class BudgetExceededException(Exception):
    """Raised when an objective exceeds its allocated budget."""
    pass


class ObjectiveCancelledException(Exception):
    """Raised when an objective is cancelled during execution."""
    pass


@runtime_checkable
class ToolLike(Protocol):
    """Protocol for duck-typed tool validation.

    Any object with these attributes can be converted to a Tool.
    """
    name: str
    fn: Callable
    schema: dict


# =============================================================================
# Core Data Model
# =============================================================================

@dataclass
class Objective:
    """
    An objective is background work with its own conversation thread.

    Simple objectives: "search for X and summarize"
    Complex objectives: "research Y, create a report, email it to Z"

    The agent handles complexity naturally through its loop.
    """
    id: str
    goal: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    thread_id: str = ""
    schedule: str | None = None  # cron expression for recurring
    result: str | None = None
    error: str | None = None
    created: str = ""
    completed: str | None = None
    # Priority: 1=highest, 10=lowest (default 5)
    priority: int = 5
    # Retry configuration
    retry_count: int = 0
    max_retries: int = 3
    last_error: str | None = None
    # Error history for retry awareness
    error_history: list = field(default_factory=list)
    # Budget tracking
    budget_usd: float | None = None  # Max cost allowed (None = unlimited)
    spent_usd: float = 0.0
    token_limit: int | None = None  # Max tokens allowed (None = unlimited)
    tokens_used: int = 0

    def __post_init__(self):
        if not self.thread_id:
            self.thread_id = f"objective_{self.id}"
        if not self.created:
            self.created = datetime.now().isoformat()


class Tool:
    """A capability the agent can use."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict,
        fn: Callable,
        packages: list[str] = None,
        env: list[str] = None
    ):
        self.name = name
        self.fn = fn
        self.packages = packages or []
        self.env = env or []
        self.schema = {
            "name": name,
            "description": description,
            "input_schema": parameters
        }

    def execute(self, params: dict, agent: "Agent"):
        return self.fn(params, agent)

    def check_health(self) -> dict:
        """
        Check if this tool's requirements are satisfied.

        Returns:
            {
                "ready": True/False,
                "missing_packages": [...],
                "missing_env": [...],
            }
        """
        from tools import check_requirements
        return check_requirements(self.packages, self.env)


class Agent(EventEmitter):
    """
    The agent: a loop that processes messages, with support for background objectives.

    ┌─────────────────────────────────────────┐
    │              Agent                       │
    │  ┌─────────┐  ┌────────────┐            │
    │  │ Threads │  │ Objectives │            │
    │  │  (chat) │  │ (bg work)  │            │
    │  └────┬────┘  └─────┬──────┘            │
    │       └──────┬──────┘                   │
    │              ▼                          │
    │         ┌────────┐                      │
    │         │ Tools  │                      │
    │         └────────┘                      │
    └─────────────────────────────────────────┘

    Multi-Channel Architecture:
    - Listeners (input): Receive messages from CLI, email, voice, etc.
    - Senders (output): Send messages via any channel
    - Context: Each message carries channel info and owner status

    Events emitted:
    - tool_start: {"name": str, "input": dict}
    - tool_end: {"name": str, "result": any, "duration_ms": int}
    - objective_start: {"id": str, "goal": str}
    - objective_end: {"id": str, "status": str, "result": str}
    - task_start: {"id": str, "name": str, "goal": str}
    - task_end: {"id": str, "status": str, "duration_ms": int}
    """

    def __init__(self, model: str = None, load_tools: bool = True, config: dict = None, use_litellm: bool = True):
        self.__init_events__()  # Initialize event system

        # Initialize LLM configuration from config
        if config:
            init_llm_config(config)

        # Get model from config if not provided
        llm_config = get_llm_config()
        self.model = model or llm_config.agent_model.model_id

        # Select client based on configuration
        # use_litellm=True (default): Use LiteLLM for multi-provider support (OpenAI, Anthropic, etc.)
        # use_litellm=False: Use direct Anthropic client (requires ANTHROPIC_API_KEY)
        if use_litellm:
            self.client = AsyncLiteLLMAnthropicAdapter()
        else:
            self.client = InstrumentedAsyncAnthropic()

        set_event_emitter(self)  # Agent is an EventEmitter, metrics emit through it

        self.tools: dict[str, Tool] = {}
        self.threads: dict[str, list] = {"main": []}
        self.objectives: dict[str, Objective] = {}
        self._running_objectives: set[str] = set()
        self._lock = asyncio.Lock()

        # Concurrency control for objectives
        self.MAX_CONCURRENT_OBJECTIVES = 5
        self._concurrency_semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_OBJECTIVES)

        # Cancellation tokens for stopping running objectives
        self._cancellation_tokens: dict[str, asyncio.Event] = {}

        # Priority queue for objectives: (priority, timestamp, obj_id)
        self._objective_queue: list[tuple[int, float, str]] = []

        # Reference to main event loop for tools running in thread pool
        # Set when first async operation runs
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # Per-thread locks for serializing operations on same thread_id
        # Prevents race conditions when multiple sources (scheduler, email, etc.)
        # access the same conversation thread concurrently
        self._thread_locks: dict[str, asyncio.Lock] = {}

        # Activity tracking for priority-aware background tasks.
        # Updated on every interactive message. Background extraction checks this
        # to yield to interactive traffic (prevents extraction from starving chat).
        self._last_interactive_activity: float | None = None

        # Multi-channel support
        self.senders: dict[str, "Sender"] = {}  # Channel senders for output
        self.config = config or {}  # Agent configuration
        self._current_context: dict = {}  # Context for current message

        # Scheduler for recurring and one-time tasks
        self.scheduler = Scheduler(executor=self._execute_scheduled_task)

        # Memory system - try SQLite, gracefully fall back to in-memory
        self.memory = self._initialize_memory()

        # Metrics collector for tracking costs and performance
        self.metrics_collector = MetricsCollector(
            store=self.memory.store if self.memory else None
        )
        self.metrics_collector.attach(self)
        _set_metrics_collector(self.metrics_collector)  # Set global for tool access

        # Register core tools
        self._register_core_tools()

        # Load persisted dynamic tools from database (self-improvement)
        self._load_persisted_tools()

        # Load external tools from /tools directory
        if load_tools:
            self._load_external_tools()

        # Tool context builder for intelligent tool selection
        self._tool_context_builder = self._initialize_tool_context()
        self._current_tool_selection = None  # Cached selection for current turn

    def register_sender(self, channel: str, sender):
        """Register a channel sender for outbound messages.

        Args:
            channel: Channel name (e.g., "email", "sms", "whatsapp")
            sender: Sender instance implementing the Sender protocol
        """
        self.senders[channel] = sender

    def register(
        self,
        tool: Tool | ToolLike,
        emit_event: bool = False,
        source_code: str | None = None,
        tool_var_name: str | None = None,
        category: str = "custom",
        is_dynamic: bool = True,
    ):
        """Register a tool with validation.

        Accepts Tool instances or any object matching ToolLike protocol.
        Duck-typed objects are automatically converted to Tool instances.

        Args:
            tool: The tool to register.
            emit_event: If True, emit a tool_registered event for persistence.
            source_code: Source code for dynamic tools (enables persistence).
            tool_var_name: Variable name in source code.
            category: Tool category for organization.
            is_dynamic: Whether this is a dynamically created tool.

        Raises:
            ToolValidationError: If tool is malformed or missing required attributes.
        """
        validated = self._validate_and_convert(tool)
        self.tools[validated.name] = validated

        # Emit event for persistence (hooks will handle DB storage)
        if emit_event:
            self.emit("tool_registered", {
                "name": validated.name,
                "description": validated.schema.get("description", ""),
                "parameters": validated.schema.get("input_schema", {}),
                "source_code": source_code,
                "packages": getattr(validated, "packages", []),
                "env": getattr(validated, "env", []),
                "tool_var_name": tool_var_name,
                "category": category,
                "is_dynamic": is_dynamic,
            })

    def _validate_and_convert(self, obj) -> Tool:
        """Validate and convert an object to a proper Tool instance.

        This provides defense against:
        - Custom classes missing required attributes
        - Malformed schema dictionaries
        - Missing or invalid tool functions
        """
        # Already a proper Tool instance
        if isinstance(obj, Tool):
            return obj

        # Check for ToolLike protocol (duck typing)
        if isinstance(obj, ToolLike):
            # Validate schema structure
            schema = obj.schema
            if not isinstance(schema, dict):
                raise ToolValidationError(
                    f"Tool '{getattr(obj, 'name', '?')}' has invalid schema: expected dict, got {type(schema).__name__}"
                )
            if "name" not in schema or "input_schema" not in schema:
                raise ToolValidationError(
                    f"Tool '{obj.name}' schema missing required keys. "
                    f"Expected 'name', 'description', 'input_schema'. Got: {list(schema.keys())}"
                )
            # Convert to proper Tool
            return Tool(
                name=obj.name,
                description=schema.get("description", f"Tool: {obj.name}"),
                parameters=schema["input_schema"],
                fn=obj.fn
            )

        # Not a Tool and doesn't match protocol - provide helpful error
        missing = []
        for attr in ("name", "fn", "schema"):
            if not hasattr(obj, attr):
                missing.append(attr)

        if missing:
            raise ToolValidationError(
                f"Invalid tool object (type: {type(obj).__name__}). "
                f"Missing required attributes: {missing}. "
                f"Use the Tool class or ensure your class has 'name', 'fn', and 'schema' attributes."
            )

        # Has attributes but wrong types
        raise ToolValidationError(
            f"Tool '{getattr(obj, 'name', '?')}' has attributes but failed protocol check. "
            f"Verify 'name' is str, 'fn' is callable, and 'schema' is dict."
        )

    def _initialize_memory(self):
        """Initialize the memory system with automatic SQLite setup.

        Falls back gracefully to in-memory storage if SQLite fails.
        Returns Memory instance if successful, None otherwise.
        """
        from pathlib import Path

        memory_config = self.config.get("memory", {})
        if memory_config.get("enabled") is False:
            console.system("Memory: disabled (config)")
            return None

        try:
            from memory import Memory
            from memory.integration import setup_memory_hooks

            store_path = memory_config.get("path", "~/.babyagi/memory")
            db_path = Path(store_path).expanduser() / "memory.db"
            is_new = not db_path.exists()

            console.system(f"Memory: {'initializing' if is_new else 'loading'} SQLite database...")
            memory = Memory(store_path=store_path)
            setup_memory_hooks(self, memory)

            self._log_memory_success(memory, db_path.parent, is_new)
            return memory

        except (ImportError, PermissionError, Exception) as e:
            self._log_memory_fallback(e)
            return None

    def _log_memory_success(self, memory, path, is_new: bool):
        """Log successful memory initialization."""
        if is_new:
            console.success(f"Memory: created at {path}")
        else:
            try:
                count = memory.store._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                console.success(f"Memory: loaded ({count} events)")
            except Exception as e:
                logger.debug("Could not count memory events: %s", e)
                console.success("Memory: loaded (SQLite persistent)")

    def _log_memory_fallback(self, error: Exception):
        """Log memory fallback due to error."""
        if isinstance(error, ImportError):
            console.warning(f"SQLite memory unavailable: {error}")
        elif isinstance(error, PermissionError):
            console.error(f"Cannot access memory directory: {error}")
        else:
            console.error(f"Memory initialization failed: {error}")
        console.system("Memory: using in-memory storage (session only)")

    def _register_core_tools(self):
        """Register the built-in tools."""
        # Use enhanced memory tool if SQLite memory is available
        if self.memory is not None:
            from memory.integration import create_enhanced_memory_tool
            self.register(create_enhanced_memory_tool(self.memory))
        else:
            self.register(_memory_tool(self))

        self.register(_objective_tool(self))
        self.register(_notes_tool(self))
        self.register(_schedule_tool(self))
        self.register(_register_tool_tool(self))
        self.register(_send_message_tool(self))

        # Register skill and composio management tools
        try:
            from tools.skills import get_skill_tools
            for tool in get_skill_tools(self, Tool):
                self.register(tool)
        except ImportError:
            pass  # Skills module not available

    def _load_persisted_tools(self):
        """Load dynamically-created tools from the database on startup.

        This enables self-improvement: tools the agent creates persist
        across restarts and are automatically reloaded.

        Handles three tool types:
        - "executable": Python code that runs directly
        - "skill": Returns behavioral instructions when called
        - "composio": Thin wrapper that calls Composio library

        Tools that fail to load are disabled rather than crashing.
        """
        if self.memory is None:
            return

        try:
            tool_defs = self.memory.store.get_dynamic_tool_definitions(enabled_only=True)
            if not tool_defs:
                return

            loaded = 0
            failed = 0
            skills_loaded = 0
            composio_loaded = 0

            # Try to import skill helpers
            try:
                from tools.skills import create_skill_tool, create_composio_tool
                skills_available = True
            except ImportError:
                skills_available = False

            # Try to get composio client for composio tools
            composio_client = None
            try:
                from composio import Composio
                composio_client = Composio()
            except (ImportError, Exception) as e:
                logger.debug("Composio not available: %s", e)

            for tool_def in tool_defs:
                try:
                    tool = None
                    tool_type = getattr(tool_def, 'tool_type', 'executable') or 'executable'

                    if tool_type == "skill" and skills_available:
                        # Skill tool - returns instructions
                        tool = create_skill_tool(tool_def, Tool)
                        skills_loaded += 1
                    elif tool_type == "composio" and composio_client:
                        # Composio tool - wraps Composio API
                        tool = create_composio_tool(tool_def, Tool, composio_client)
                        composio_loaded += 1
                    else:
                        # Executable tool - reconstruct from source code
                        tool = self._reconstruct_tool(tool_def)

                    if tool:
                        # Register without emitting event (already persisted)
                        self.tools[tool.name] = tool
                        loaded += 1
                except Exception as e:
                    # Disable broken tool rather than crash
                    failed += 1
                    try:
                        self.memory.store.disable_tool(
                            tool_def.name,
                            reason=f"Failed to load on startup: {str(e)[:200]}"
                        )
                    except Exception as e:
                        logger.warning("Failed to disable broken tool '%s' in database: %s", tool_def.name, e)

            if loaded > 0 or failed > 0:
                details = []
                if skills_loaded > 0:
                    details.append(f"{skills_loaded} skills")
                if composio_loaded > 0:
                    details.append(f"{composio_loaded} composio")
                exec_count = loaded - skills_loaded - composio_loaded
                if exec_count > 0:
                    details.append(f"{exec_count} executable")

                detail_str = f" ({', '.join(details)})" if details else ""

                if failed == 0:
                    console.success(f"Tools: loaded {loaded} persisted tool(s){detail_str}")
                else:
                    console.warning(
                        f"Tools: loaded {loaded}{detail_str}, disabled {failed} broken tool(s)"
                    )

        except Exception as e:
            # Don't crash startup if tool loading fails
            console.warning(f"Could not load persisted tools: {e}")

    def _reconstruct_tool(self, tool_def) -> Tool | None:
        """
        Reconstruct a Tool from its persisted ToolDefinition.

        For tools with external packages, creates a sandboxed executor.
        For local tools, executes source code to get the function.

        Returns:
            Tool instance if successful, None if reconstruction fails.
        """
        # No source code - can't reconstruct
        if not tool_def.source_code:
            return None

        # Has external packages - needs sandbox execution
        if tool_def.packages:
            return self._create_sandboxed_tool_from_definition(tool_def)

        # Local tool - execute source to get function
        try:
            # Set up namespace with required imports
            namespace = {
                "Tool": Tool,
                "datetime": datetime,
                "json": json,
            }

            # Execute the source code
            exec(tool_def.source_code, namespace)

            # Get the tool variable
            if tool_def.tool_var_name and tool_def.tool_var_name in namespace:
                return namespace[tool_def.tool_var_name]

            # Try to find a Tool instance in namespace
            for name, value in namespace.items():
                if isinstance(value, Tool):
                    return value

            return None

        except Exception as e:
            raise RuntimeError(f"Failed to reconstruct tool '{tool_def.name}': {e}")

    def _create_sandboxed_tool_from_definition(self, tool_def) -> Tool:
        """
        Create a sandboxed tool from a ToolDefinition.

        Uses e2b sandbox for tools with external package dependencies.
        """
        def sandboxed_fn(params: dict, agent) -> dict:
            try:
                from tools.sandbox import run_in_sandbox

                # Build the execution code
                exec_code = f'''
{tool_def.source_code}

# Execute the tool function
import json
params = json.loads("""{json.dumps(params)}""")
if "{tool_def.tool_var_name}" in dir():
    tool = {tool_def.tool_var_name}
    result = tool.fn(params, None)
else:
    result = {{"error": "Tool variable not found"}}
print("RESULT:", json.dumps(result))
'''
                result = run_in_sandbox(
                    exec_code,
                    packages=tool_def.packages,
                    timeout=120
                )
                return result
            except Exception as e:
                return {"error": f"Sandbox execution failed: {str(e)}"}

        return Tool(
            name=tool_def.name,
            description=tool_def.description,
            parameters=tool_def.parameters,
            fn=sandboxed_fn,
            packages=tool_def.packages,
            env=tool_def.env,
        )

    async def _execute_scheduled_task(self, task: ScheduledTask) -> str:
        """Execute a scheduled task - runs as the agent with its own thread."""
        import logging
        import traceback
        logger = logging.getLogger(__name__)

        logger.info(f"[Scheduler] Starting task execution: {task.name} (id={task.id})")

        # Emit task_start event
        self.emit("task_start", {
            "id": task.id,
            "name": task.name,
            "goal": task.goal
        })

        start_time = time.time()

        prompt = f"""Execute this scheduled task: {task.goal}

Task: {task.name}
Schedule: {task.schedule.human_readable()}

Work autonomously. Use tools as needed. When done, provide a brief summary of what you accomplished."""

        try:
            logger.info(f"[Scheduler] Calling run_async for task: {task.name}")
            result = await self.run_async(prompt, task.thread_id)
            logger.info(f"[Scheduler] Task completed: {task.name}, result length: {len(result)}")
            duration_ms = int((time.time() - start_time) * 1000)

            # Emit task_end event (success)
            self.emit("task_end", {
                "id": task.id,
                "name": task.name,
                "status": "completed",
                "duration_ms": duration_ms
            })

            return result
        except Exception as e:
            logger.exception(f"[Scheduler] Task failed: {task.name}, error: {e}")
            duration_ms = int((time.time() - start_time) * 1000)

            # Emit task_end event (failure)
            self.emit("task_end", {
                "id": task.id,
                "name": task.name,
                "status": "failed",
                "error": str(e),
                "duration_ms": duration_ms
            })

            # Don't re-raise - let other tasks continue
            return f"Task failed: {e}"

    def _load_external_tools(self):
        """Load tools from the tools/ folder."""
        try:
            from tools import get_all_tools
            for tool in get_all_tools(Tool):
                self.register(tool)
        except ImportError:
            pass

    def _initialize_tool_context(self):
        """Initialize the tool context builder for intelligent tool selection.

        The builder selects a subset of relevant tools for each API call instead
        of sending all tools. This manages context window usage as the agent
        creates more tools over time.

        Returns:
            ToolContextBuilder if memory is available, None otherwise.
        """
        if self.memory is None:
            return None

        try:
            from memory.tool_context import create_tool_context_builder
            return create_tool_context_builder(self.memory.store)
        except ImportError:
            return None
        except Exception as e:
            logger.debug("Could not create tool context builder, continuing without smart tool selection: %s", e)
            return None

    def _refresh_tool_selection(self, current_query: str = None, context: dict = None):
        """Refresh the tool selection for the current turn.

        Called before making API calls to select which tools to include.
        Caches the selection in _current_tool_selection for use by
        _tool_schemas() and _system_prompt().

        Args:
            current_query: The current user query for relevance matching.
            context: Channel context with metadata.
        """
        if self._tool_context_builder is None:
            self._current_tool_selection = None
            return

        # Extract topics from agent state if available
        current_topics = None
        if self.memory and hasattr(self.memory, 'store'):
            try:
                agent_state = self.memory.store.get_agent_state()
                if agent_state:
                    current_topics = agent_state.current_topics
            except Exception as e:
                logger.debug("Could not retrieve agent state for tool selection: %s", e)

        current_channel = context.get("channel") if context else None

        self._current_tool_selection = self._tool_context_builder.select_tools(
            all_tools=self.tools,
            current_query=current_query,
            current_topics=current_topics,
            current_channel=current_channel,
        )

    # -------------------------------------------------------------------------
    # Message Processing
    # -------------------------------------------------------------------------

    def run(self, user_input: str, thread_id: str = "main") -> str:
        """
        Process user input synchronously.
        For async with background objectives, use run_async().
        """
        return asyncio.get_event_loop().run_until_complete(
            self.run_async(user_input, thread_id)
        )

    def _get_thread_lock(self, thread_id: str) -> asyncio.Lock:
        """Get or create a lock for a thread_id.

        Ensures serialized access to each conversation thread,
        preventing race conditions when multiple sources (scheduler,
        email, CLI) access the same thread concurrently.
        """
        if thread_id not in self._thread_locks:
            self._thread_locks[thread_id] = asyncio.Lock()
        return self._thread_locks[thread_id]

    def _spawn_background_task(self, coro) -> bool:
        """Schedule an async coroutine as a background task from any thread.

        This method safely schedules a coroutine on the main event loop,
        even when called from a thread pool worker (e.g., during tool execution).

        Args:
            coro: The coroutine to run as a background task

        Returns:
            True if the task was scheduled, False if no event loop is available.
            The task runs asynchronously; errors should be handled by the coroutine.
        """
        if self._main_loop is None:
            return False

        asyncio.run_coroutine_threadsafe(coro, self._main_loop)
        return True

    async def run_async(self, user_input: str, thread_id: str = "main", context: dict = None) -> str:
        """Process user input and return response. Objectives run in background.

        Args:
            user_input: The message to process
            thread_id: Conversation thread ID (e.g., "main", "email:123", "voice:session")
            context: Channel context dict with:
                - channel: Source channel ("cli", "email", "voice", etc.)
                - is_owner: Whether message is from the agent's owner
                - sender: Sender identifier (email address, phone, etc.)
                - Additional channel-specific metadata

        Thread Safety:
            Operations on each thread_id are serialized via per-thread locks.
            This prevents message interleaving when concurrent requests
            (e.g., scheduled task + email) target the same thread.
        """
        # Capture main event loop for tools that need it
        if self._main_loop is None:
            self._main_loop = asyncio.get_running_loop()

        # Acquire per-thread lock to serialize operations on this thread
        async with self._get_thread_lock(thread_id):
            context = context or {"channel": "cli", "is_owner": True}
            self._current_context = context

            # Track interactive activity so background tasks can yield
            if not thread_id.startswith("objective_"):
                self._last_interactive_activity = time.time()

            thread = self.threads.setdefault(thread_id, [])

            # Auto-repair corrupted threads (orphaned tool_use without tool_result)
            # This prevents "tool_use ids were found without tool_result blocks" errors
            if thread:
                repair_result = self.repair_thread(thread_id)
                if repair_result.get("repaired", 0) > 0:
                    self.emit("thread_repaired", {
                        "thread_id": thread_id,
                        "repaired": repair_result["repaired"]
                    })

            thread.append({"role": "user", "content": user_input})

            # Log inbound message to memory for context assembly
            self._current_memory_event = None
            if self.memory:
                try:
                    from memory.integration import log_message
                    self._current_memory_event = log_message(
                        self.memory, user_input, context, "inbound"
                    )
                except Exception as e:
                    logger.warning("Failed to log inbound message to memory: %s", e)

            # Refresh tool selection for this turn
            # This selects relevant tools based on query, context, and usage patterns
            self._refresh_tool_selection(user_input, context)

            # Generate context-aware system prompt
            is_owner = context.get("is_owner", True)
            system_prompt = self._system_prompt(thread_id, is_owner, context)

            while True:
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=8096,
                    system=system_prompt,
                    tools=self._tool_schemas(),
                    messages=thread
                )

                # Track budget and tokens for objectives
                if thread_id.startswith("objective_"):
                    obj_id = thread_id.replace("objective_", "")
                    await self._track_objective_usage(obj_id, response)

                # Check cancellation for objectives
                if thread_id.startswith("objective_"):
                    obj_id = thread_id.replace("objective_", "")
                    if obj_id in self._cancellation_tokens and self._cancellation_tokens[obj_id].is_set():
                        raise ObjectiveCancelledException(f"Objective {obj_id} was cancelled")

                thread.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    return self._extract_text(response)

                # Execute tools (in thread pool to avoid blocking event loop)
                # CRITICAL: We must ensure a tool_result is appended for every tool_use,
                # even if execution fails. Otherwise the thread becomes corrupted and
                # the API will reject subsequent messages.
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        # Emit tool_start event
                        self.emit("tool_start", {
                            "name": block.name,
                            "input": block.input
                        })

                        start_time = time.time()
                        result = None
                        error_msg = None

                        try:
                            # Check if tool exists
                            if block.name not in self.tools:
                                error_msg = f"Tool '{block.name}' not found"
                                result = {"error": error_msg}
                            else:
                                # Check if tool function is async
                                tool_fn = self.tools[block.name].fn
                                if inspect.iscoroutinefunction(tool_fn):
                                    # Async tools: await directly (they're I/O bound, not CPU bound)
                                    result = await tool_fn(block.input, self)
                                else:
                                    # Sync tools: run in thread pool to prevent blocking the event loop
                                    result = await asyncio.to_thread(
                                        self.tools[block.name].execute, block.input, self
                                    )
                        except Exception as e:
                            # Tool execution failed - capture the error
                            error_msg = f"Tool execution failed: {str(e)}"
                            result = {"error": error_msg}

                        duration_ms = int((time.time() - start_time) * 1000)

                        # Emit tool_end event (even on error, for logging)
                        self.emit("tool_end", {
                            "name": block.name,
                            "result": result,
                            "duration_ms": duration_ms
                        })

                        # Serialize result safely
                        try:
                            result_json = json.dumps(result, default=json_serialize)
                        except Exception as e:
                            # JSON serialization failed - use error message instead
                            result_json = json.dumps({
                                "error": f"Result serialization failed: {str(e)}",
                                "original_type": type(result).__name__
                            })

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_json
                        })

                thread.append({"role": "user", "content": tool_results})

    def _build_objective_prompt(self, obj: "Objective") -> str:
        """Build a structured objective prompt with planning, verification, and error awareness.

        Includes:
        - Planning step (break goal into steps before executing)
        - Verification step (confirm work achieves the goal before finishing)
        - Re-plan instruction (stop and re-plan if approach isn't working)
        - Elegance check (consider simpler approaches for non-trivial work)
        - Error history (on retries, include previous errors so approach can adapt)
        """
        # Base objective with planning structure
        parts = [f"Complete this objective: {obj.goal}"]

        # If retrying, include error history so the agent can adapt
        if obj.error_history:
            error_lines = "\n".join(
                f"- Attempt {i + 1}: {e}" for i, e in enumerate(obj.error_history)
            )
            parts.append(f"""
PREVIOUS ATTEMPTS FAILED:
{error_lines}

Adjust your approach to avoid repeating these errors. Consider a different strategy.""")

        # Planning and verification instructions
        parts.append("""
APPROACH:
1. PLAN: Break this into concrete steps. Identify what tools and information you need.
2. EXECUTE: Work through your plan step by step. Use the notes tool to track your plan and mark steps complete.
3. VERIFY: Before finishing, confirm your work actually achieves the goal. Check outputs, results, and correctness.
4. If something isn't working after 2-3 attempts at a step, stop and re-plan with a different approach.
5. For non-trivial solutions, briefly consider if there's a simpler or more elegant approach before finalizing.

Work autonomously. When done, provide a summary of what was accomplished and how it was verified.""")

        return "\n".join(parts)

    async def run_objective(self, objective_id: str):
        """Execute an objective in the background with concurrency control and retry logic."""
        obj = self.objectives.get(objective_id)
        if not obj or obj.status not in ("pending", "running"):
            return

        # Check if cancelled before starting
        if obj.status == "cancelled":
            return

        # Create cancellation token for this objective
        cancel_event = self._cancellation_tokens.setdefault(objective_id, asyncio.Event())

        # Acquire concurrency semaphore (limits concurrent objectives)
        async with self._concurrency_semaphore:
            # Check cancellation after acquiring semaphore
            if cancel_event.is_set() or obj.status == "cancelled":
                self._cleanup_objective(objective_id)
                return

            async with self._lock:
                if objective_id in self._running_objectives:
                    return
                self._running_objectives.add(objective_id)
                obj.status = "running"

            # Emit objective_start event
            self.emit("objective_start", {
                "id": objective_id,
                "goal": obj.goal,
                "priority": obj.priority,
                "attempt": obj.retry_count + 1
            })

            try:
                # Run the objective with its own thread
                prompt = self._build_objective_prompt(obj)

                result = await self.run_async(prompt, obj.thread_id)

                # Check if cancelled during execution
                if cancel_event.is_set() or obj.status == "cancelled":
                    self._cleanup_objective(objective_id)
                    return

                obj.result = result
                obj.status = "completed"
                obj.completed = datetime.now().isoformat()

                # Emit objective_end event (success)
                self.emit("objective_end", {
                    "id": objective_id,
                    "status": "completed",
                    "result": result,
                    "spent_usd": obj.spent_usd,
                    "tokens_used": obj.tokens_used
                })

            except BudgetExceededException as e:
                # Budget exceeded - do not retry
                obj.status = "failed"
                obj.error = str(e)
                self.emit("objective_end", {
                    "id": objective_id,
                    "status": "failed",
                    "error": str(e),
                    "spent_usd": obj.spent_usd,
                    "tokens_used": obj.tokens_used
                })

            except ObjectiveCancelledException:
                # Cancelled - already handled
                self._cleanup_objective(objective_id)

            except Exception as e:
                obj.last_error = str(e)
                obj.error_history.append(str(e))
                obj.retry_count += 1

                if obj.retry_count < obj.max_retries:
                    # Exponential backoff: 2^retry_count seconds (2s, 4s, 8s)
                    delay = 2 ** obj.retry_count
                    obj.status = "pending"  # Reset for retry

                    self.emit("objective_retry", {
                        "id": objective_id,
                        "attempt": obj.retry_count,
                        "max_retries": obj.max_retries,
                        "delay_seconds": delay,
                        "error": str(e)
                    })

                    # Schedule retry after delay
                    await asyncio.sleep(delay)
                    self._running_objectives.discard(objective_id)
                    # Re-queue with same priority
                    self._spawn_background_task(self.run_objective(objective_id))
                    return
                else:
                    # Max retries exceeded
                    obj.status = "failed"
                    obj.error = f"Failed after {obj.max_retries} attempts. Last error: {e}"

                    self.emit("objective_end", {
                        "id": objective_id,
                        "status": "failed",
                        "error": obj.error,
                        "attempts": obj.retry_count
                    })

            finally:
                self._running_objectives.discard(objective_id)

    def _cleanup_objective(self, objective_id: str):
        """Clean up resources for a cancelled objective."""
        obj = self.objectives.get(objective_id)
        if obj:
            obj.status = "cancelled"
            obj.completed = datetime.now().isoformat()
        self._running_objectives.discard(objective_id)
        self._cancellation_tokens.pop(objective_id, None)
        self.emit("objective_end", {
            "id": objective_id,
            "status": "cancelled"
        })

    async def _track_objective_usage(self, objective_id: str, response):
        """Track token usage and cost for an objective, raising if budget exceeded."""
        obj = self.objectives.get(objective_id)
        if not obj:
            return

        # Extract usage from response
        usage = getattr(response, 'usage', None)
        if not usage:
            return

        input_tokens = getattr(usage, 'input_tokens', 0)
        output_tokens = getattr(usage, 'output_tokens', 0)
        total_tokens = input_tokens + output_tokens

        # Update token count
        obj.tokens_used += total_tokens

        # Calculate cost (using Claude Sonnet pricing as default)
        # Input: $3/1M tokens, Output: $15/1M tokens
        cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
        obj.spent_usd += cost

        # Emit usage event
        self.emit("objective_usage", {
            "id": objective_id,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "total_spent_usd": obj.spent_usd,
            "total_tokens_used": obj.tokens_used
        })

        # Check token limit
        if obj.token_limit and obj.tokens_used >= obj.token_limit:
            raise BudgetExceededException(
                f"Objective {objective_id} exceeded token limit: "
                f"{obj.tokens_used} >= {obj.token_limit} tokens"
            )

        # Check budget limit
        if obj.budget_usd and obj.spent_usd >= obj.budget_usd:
            raise BudgetExceededException(
                f"Objective {objective_id} exceeded budget: "
                f"${obj.spent_usd:.4f} >= ${obj.budget_usd:.4f}"
            )

    # -------------------------------------------------------------------------
    # Scheduling
    # -------------------------------------------------------------------------

    async def run_scheduler(self):
        """Run the unified scheduler loop."""
        await self.scheduler.start()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _system_prompt(self, thread_id: str = "main", is_owner: bool = True, context: dict = None) -> str:
        """Generate context-aware system prompt.

        Assembles three layers:
        1. Base prompt (static): agent identity, owner info, capabilities, rules
        2. Memory context (dynamic): knowledge, preferences, learnings, state
        3. Situation context (per-message): channel, thread, is_owner, formatting
        """
        context = context or {}
        base = self._build_base_prompt()
        memory_ctx = self._build_memory_context()
        situation = self._build_context_section(thread_id, is_owner, context)
        return base + memory_ctx + situation

    def _build_memory_context(self) -> str:
        """Build dynamic context from the memory system.

        Includes: knowledge summary, agent state, user preferences,
        recent activity, counterparty info, and context-specific learnings.
        Returns empty string if memory is unavailable.
        """
        if not self.memory:
            return ""
        try:
            from memory.integration import get_memory_context_prompt
            event = getattr(self, "_current_memory_event", None)
            return get_memory_context_prompt(self.memory, event)
        except Exception as e:
            logger.warning("Failed to build memory context prompt: %s", e)
            return ""

    def _build_owner_section(self) -> str:
        """Build owner info for the system prompt from config.

        Only includes fields that are actually set. This gives the agent
        basic knowledge about its owner (contact info, timezone) that
        comes from the owner's explicit configuration.
        """
        owner_config = self.config.get("owner", {})
        lines = []
        if owner_config.get("name"):
            lines.append(f"- Name: {owner_config['name']}")
        if owner_config.get("email"):
            lines.append(f"- Email: {owner_config['email']}")
        if owner_config.get("phone"):
            lines.append(f"- Phone: {owner_config['phone']}")
        if owner_config.get("timezone"):
            lines.append(f"- Timezone: {owner_config['timezone']}")
        if not lines:
            return ""
        return "\nYOUR OWNER:\n" + "\n".join(lines) + "\n"

    def _build_status_summaries(self) -> str:
        """Build status summaries for objectives, tasks, and channels."""
        parts = []

        # Active objectives
        active = [o for o in self.objectives.values() if o.status in ("pending", "running")]
        if active:
            parts.append("\n\nActive objectives:\n" + "\n".join(
                f"- [{o.id[:8]}] {o.goal} ({o.status})" for o in active
            ))

        # Scheduled tasks
        tasks = self.scheduler.list()
        if tasks:
            parts.append("\n\nScheduled tasks:\n" + "\n".join(
                f"- [{t.id}] {t.name}: {t.schedule.human_readable()} (next: {t.next_run_at[:16] if t.next_run_at else 'none'})"
                for t in tasks[:5]
            ))

        # Available channels
        if self.senders:
            parts.append(f"\n\nAvailable output channels: {', '.join(self.senders.keys())}")

        return "".join(parts)

    def _build_base_prompt(self) -> str:
        """Build the base capabilities prompt."""
        status = self._build_status_summaries()

        # Add tool inventory summary if available (for intelligent tool selection)
        tool_inventory = ""
        if self._current_tool_selection is not None:
            tool_inventory = f"\n\n{self._current_tool_selection.tool_inventory_summary}"

        # Get agent identity from config
        agent_config = self.config.get("agent", {})
        agent_name = agent_config.get("name", "Assistant")
        agent_description = agent_config.get("description", "a helpful AI assistant")
        agent_objective = agent_config.get("objective", "Help my owner with tasks and handle communications on their behalf.")

        # Build owner info section from config
        owner_section = self._build_owner_section()

        # Get behavior settings
        behavior = agent_config.get("behavior", {})
        spending_config = behavior.get("spending", {})
        require_approval = spending_config.get("require_approval", True)
        auto_approve_limit = spending_config.get("auto_approve_limit", 0.0)
        accounts_config = behavior.get("accounts", {})
        check_existing = accounts_config.get("check_existing_first", True)

        # Build spending rules
        if require_approval and auto_approve_limit > 0:
            spending_rules = f"Auto-approve purchases under ${auto_approve_limit:.2f}. For larger amounts, consult owner first."
        elif require_approval:
            spending_rules = "ALWAYS consult owner before making any purchase or financial commitment."
        else:
            spending_rules = "You may make purchases as needed for tasks."

        # Build account check rules
        account_check_rule = ""
        if check_existing:
            account_check_rule = """
BEFORE CREATING ANY ACCOUNT:
1. Use search_credentials(query="servicename") to check if you already have an account
2. Use memory(action="search", query="servicename") to check if you've used this service before
3. Only create a new account if you confirm none exists"""

        return f"""You are {agent_name}, {agent_description}.

YOUR IDENTITY:
- Your name is {agent_name}
- Your purpose: {agent_objective}
- You have your own email address via AgentMail - use get_agent_email() to retrieve it
- You are an autonomous agent acting on behalf of your owner
{owner_section}
CRITICAL IDENTITY RULES:
1. You have your OWN email address - ALWAYS use get_agent_email() when you need an email for signups
2. NEVER create new email accounts (Gmail, Yahoo, Outlook, Hotmail, etc.) - you already have your own email
3. NEVER use temporary email services (temp-mail.org, 10minutemail, guerrillamail, mailinator, etc.)
4. When signing up for ANY service, use YOUR email from get_agent_email()
5. Your email is for YOUR accounts - if the owner wants to create an account for themselves, ask for their email

ACCOUNT CREATION WORKFLOW:
When asked to sign up for a service or create an account:
1. Call get_agent_email() to get YOUR email address
2. Use browse() to navigate to the signup page
3. Fill in the signup form using YOUR email from step 1
4. Use wait_for_email(from_contains="servicename") to receive verification
5. Use read_email() to get the verification link/code
6. Complete verification with browse() or browse_with_credentials()
7. Store credentials with store_credential(service="servicename", username="your-email", password="...")
{account_check_rule}

DISTINGUISHING ACCOUNT OWNERSHIP:
- "Create an account" / "Sign up for X" → Use YOUR agent email (get_agent_email())
- "Help me create an account" / "Sign me up" / "Use my email" → Ask owner for their email
- "Create an account for me" from owner → Clarify: "Should I use my agent email, or would you like to provide your personal email?"

WORKING PRINCIPLES:
- **Simplicity first**: Make every change as simple as possible. Impact minimal code and resources.
- **Find root causes**: Investigate and fix underlying issues. No temporary workarounds or band-aids.
- **Minimal impact**: Only touch what's necessary to accomplish the goal. Avoid introducing side effects.
- **Verify before done**: Never consider a task complete without confirming it actually works.

CAPABILITIES:

1. **Direct Response**: Answer questions and have conversations normally.

2. **Background Objectives**: For immediate complex tasks, use the 'objective' tool.
   Chat continues while objectives work in the background.
   - **Priority**: Set priority 1-10 (lower = higher priority, default 5)
   - **Budget Control**: Set budget_usd or token_limit to cap costs
   - **Retry**: Failed objectives retry automatically (default 3 attempts) with exponential backoff
   - **Concurrency**: Max 5 objectives run simultaneously; others queue by priority
   - **Cancellation**: Cancel running objectives with immediate effect

3. **Scheduling**: Use the 'schedule' tool for time-based automation:
   - **One-time**: "in 5m", "in 2h", "at 2024-01-15T09:00"
   - **Recurring**: "every 5m", "every 2h", "daily at 9:00", "weekdays at 9am"
   - **Cron**: Full cron expressions like "0 9 * * 1-5" (9am weekdays)

4. **Notes**: Simple reminders and todos (passive tracking).

5. **Memory**: Store and recall facts across all conversations.

6. **Multi-Channel Communication**: Send messages via email, SMS, etc.

7. **Register New Tools**: Extend capabilities at runtime with Python code.

8. **Secure Credentials**: Store and manage sensitive data securely.
{status}{tool_inventory}

TOOL REFERENCE:

Email Operations:
- get_agent_email() - Get YOUR email address for signups and receiving emails
- send_email(to, subject, body) - Send email from your address
- check_inbox(limit, unread_only) - Check for new emails
- read_email(message_id) - Read full email content
- wait_for_email(from_contains, subject_contains, timeout_seconds) - Wait for specific email (e.g., verification)

Web Browsing:
- browse(task, url) - General web browsing and automation
- browse_with_credentials(task, credential_service, url) - Login using stored credentials (credentials injected securely)
- browse_checkout(checkout_url, card_service) - Complete payment forms (card data never visible to you)

Credential Management:
- store_credential(service, username, password) - Store account credentials
- get_credential(service, include_secrets) - Retrieve credentials (use include_secrets=False for cards)
- list_credentials() - List all stored credentials
- search_credentials(query) - Search credentials by service/username
- store_secret(key, value) - Store API keys and tokens
- get_secret(key) - Retrieve stored secrets

Memory (use the memory tool):
- memory(action="store", content="fact") - Store a fact for later recall
- memory(action="search", query="...") - Search your memory for relevant facts
- memory(action="list") - List recent memories
- memory(action="find_entity", query="...", type="person/org/concept") - Find entities in knowledge graph

CREDENTIAL SECURITY (CRITICAL):
When you create accounts, obtain API keys, or handle any sensitive credentials:
1. ALWAYS use store_credential() for usernames/passwords/account details
2. ALWAYS use store_secret() for API keys and tokens
3. NEVER leave credentials only in memory or conversation - they WILL be lost
4. For credit cards, use store_credential() with type="credit_card"

Example: After creating an account on example.com:
- store_credential(service="example.com", username="your-agent-email", password="pass123")

This ensures credentials persist across sessions and can be retrieved later.

PAYMENT SECURITY (CRITICAL):
When handling credit card payments:
1. NEVER ask users to share full card numbers in chat - use store_credential() with credential_type="credit_card"
2. NEVER use get_credential(include_secrets=True) for credit cards - you don't need to see the number
3. For checkouts, use browse_checkout(checkout_url, card_service="service_name") - it securely injects card data
4. Use list_payment_methods() to see stored cards (shows only ****1234 format)
5. Card numbers flow directly from keyring to browser - they are NEVER in your context

SECURE PAYMENT WORKFLOW:
1. Store card once: store_credential(service="payment", credential_type="credit_card", card_number="...", card_expiry="MM/YY", card_cvv="...", billing_name="...")
2. For payment: browse_checkout(checkout_url="https://...", card_service="payment")
3. The card data goes directly to the browser - you only see ****1234

SPENDING AND FINANCIAL RULES:
{spending_rules}
- Never commit to subscriptions, paid plans, or recurring charges without explicit owner approval
- If a "free" signup requires payment info, warn the owner before proceeding

PRIVACY AND SECURITY:
- NEVER share owner's personal information (email, phone, address, real name) with external services
- Use YOUR agent email (get_agent_email()) for all service signups
- If a service requires a phone number, ask owner how to proceed
- When external parties ask about your owner, be professional but protective of their privacy

WHEN TO USE SCHEDULING:

Use the 'schedule' tool when the user wants something to happen:
- At a specific future time: "remind me at 3pm" → schedule(at="15:00")
- Repeatedly: "check my email every hour" → schedule(every="1h")
- On a pattern: "send me a summary every weekday at 9am" → schedule(cron)

Use 'objective' for immediate background work without a time component.

OBJECTIVE BEST PRACTICES:

- **High priority** (1-3): Urgent tasks, time-sensitive work
- **Normal priority** (4-6): Standard background research, reports
- **Low priority** (7-10): Nice-to-have, exploratory tasks
- **Budget limits**: Set budget_usd for expensive operations to prevent runaway costs
- **Token limits**: Set token_limit for tasks that might generate excessive output
- **Decompose complex work**: If an objective involves multiple independent phases (e.g., research + analysis + report), consider spawning sub-objectives for parts that can run in parallel. Keep each sub-objective focused on one deliverable.

Example objective with controls:
- objective(action="spawn", goal="Research competitors", priority=2, budget_usd=0.50, max_retries=5)

SCHEDULE EXAMPLES:
- "Remind me in 30 minutes" → schedule add, spec="in 30m"
- "Check server status every 5 minutes" → schedule add, spec="every 5m"
- "Send daily standup at 9am" → schedule add, spec="daily at 9:00"
- "Run report on weekdays at 6pm EST" → schedule add, spec={{"kind":"cron","cron":"0 18 * * 1-5","tz":"America/New_York"}}"""

    def _build_context_section(self, thread_id: str, is_owner: bool, context: dict) -> str:
        """Build the context-specific section of the prompt."""
        channel = context.get("channel", "cli" if is_owner else "unknown")
        verbose_info = f"- Verbose: {console.get_verbose().name.lower()} (use set_verbose tool to change)"

        # Get external policy settings from config
        agent_config = self.config.get("agent", {})
        behavior = agent_config.get("behavior", {})
        external_policy = behavior.get("external_policy", {})
        consult_threshold = external_policy.get("consult_owner_threshold", "medium")

        # Build channel-specific formatting rules
        formatting_rules = ""
        if channel == "sendblue":
            formatting_rules = """

iMESSAGE FORMATTING (CRITICAL):
- Be EXTREMELY CONCISE - this is a text message, not an email
- Keep responses to 1-3 short sentences maximum
- NO markdown formatting whatsoever (no **, *, #, `, ```, -, [], () link syntax)
- Write in plain, casual text like a normal text message
- If you need to explain something complex, offer to send details via email instead
- Never send walls of text - break into multiple messages if truly necessary
- Think "text message brevity" not "assistant explanation"
"""
        elif channel == "email":
            formatting_rules = """

EMAIL FORMATTING:
- Do NOT use markdown formatting (no **, *, #, `, ```, -, [], () link syntax, etc.)
- Write in plain text only - email clients may not render markdown
- Use natural language emphasis instead of formatting (e.g., say "important" rather than **important**)
- For lists, use simple numbering (1. 2. 3.) or write items in prose"""

        if is_owner:
            return f"""

CURRENT CONTEXT:
- Channel: {channel}
- Thread: {thread_id}
{verbose_info}
- Speaking with: Owner (full access){formatting_rules}

You are speaking with your owner. You have full access to:
- Shared memory across all conversations
- Ability to spawn background objectives
- All configured communication channels
- Full context about their preferences and history

OWNER COMMUNICATION RULES:
- ALWAYS respond to your owner's messages - never ignore them
- Be helpful, proactive, and casual - you know them well
- You can be more informal and direct with your owner
- Proactively share relevant information they might find useful
- If you notice issues or have concerns, bring them up"""

        sender = context.get("sender", "unknown")

        # Build consultation rules based on threshold
        if consult_threshold == "low":
            consult_rules = """- Consult owner for any non-trivial request before acting
- Only answer simple questions independently"""
        elif consult_threshold == "high":
            consult_rules = """- You may help with most requests independently
- Consult owner only for major decisions (large purchases, data sharing, commitments)"""
        else:  # medium (default)
            consult_rules = """- You may answer questions and provide information independently
- Consult owner before: making purchases, sharing data, scheduling meetings, making commitments
- When in doubt, check with owner first"""

        return f"""

CURRENT CONTEXT:
- Channel: {channel}
- Thread: {thread_id}
{verbose_info}
- Speaking with: {sender} (external - NOT your owner){formatting_rules}

You are responding to a message from {sender}, who is NOT your owner.

EXTERNAL COMMUNICATION RULES:
- Be helpful and professional, but prioritize your owner's interests
- NEVER reveal owner's private information (personal email, phone, address, schedule details)
- NEVER share owner's credentials, API keys, or sensitive business information
- You CAN use your memory and knowledge to be helpful with general questions
- You CAN help with information requests that don't compromise owner privacy

WHEN TO CONSULT OWNER:
{consult_rules}

HOW TO CONSULT:
- Use send_message(channel="email", to="owner", ...) to reach your owner
- Say naturally: "Let me check with my owner and get back to you"
- For urgent matters, indicate the urgency in your message to owner

RESPONDING TO EXTERNAL REQUESTS:
- Spam or clearly automated messages: You may ignore these
- Legitimate inquiries: Respond professionally, consult owner if needed
- Requests for owner info: Politely decline, offer to relay a message instead
- Meeting/call requests: "I'll check my owner's availability and get back to you\""""

    def _tool_schemas(self) -> list:
        """Get tool schemas for API calls.

        Uses intelligent selection when ToolContextBuilder is available,
        otherwise returns all tools. The selection prioritizes:
        1. Core tools (memory, objective, notes, schedule, etc.)
        2. Most frequently used tools
        3. Recently used tools
        4. Tools relevant to current context

        Returns:
            List of tool schema dicts for the API call.
        """
        # Use smart selection if available
        if self._current_tool_selection is not None:
            return [
                t.schema
                for name, t in self.tools.items()
                if name in self._current_tool_selection.selected_tools
            ]
        # Fallback: all tools
        return [t.schema for t in self.tools.values()]

    def _extract_text(self, response) -> str:
        return "".join(b.text for b in response.content if hasattr(b, "text"))

    def get_thread(self, thread_id: str = "main") -> list:
        return self.threads.get(thread_id, [])

    def clear_thread(self, thread_id: str = "main"):
        self.threads[thread_id] = []

    # -------------------------------------------------------------------------
    # Memory Convenience Methods
    # -------------------------------------------------------------------------

    def memory_recall(self, query: str) -> dict:
        """
        Search memory for information matching the query.

        This is a convenience method for dynamic tools and external code.
        It works whether enhanced memory is enabled or not.

        Args:
            query: Search query string

        Returns:
            dict with "memories" list of matching results, or "error" on failure
        """
        if self.memory is not None:
            try:
                events = self.memory.search_events(query, limit=10)
                return {
                    "memories": [
                        {
                            "content": e.content[:500] if e.content else "",
                            "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                            "channel": getattr(e, "channel", None),
                        }
                        for e in events
                    ]
                }
            except Exception as e:
                return {"error": f"Memory search failed: {e}"}
        else:
            # Fallback to simple in-memory search
            query_lower = query.lower()
            matches = [m for m in MEMORIES if query_lower in m.get("content", "").lower()]
            return {"memories": matches[-10:]}

    def memory_store(self, content: str) -> dict:
        """
        Store a fact or observation in memory.

        This is a convenience method for dynamic tools and external code.
        It works whether enhanced memory is enabled or not.

        Args:
            content: The fact or observation to store

        Returns:
            dict with "stored": True on success, or "error" on failure
        """
        if self.memory is not None:
            try:
                event = self.memory.log_event(
                    content=content,
                    event_type="observation",
                    channel=None,
                    direction="internal",
                    is_owner=True,
                )
                return {"stored": True, "event_id": event.id}
            except Exception as e:
                return {"error": f"Memory store failed: {e}"}
        else:
            # Fallback to simple in-memory storage
            memory = {"content": content, "ts": datetime.now().isoformat()}
            MEMORIES.append(memory)
            return {"stored": True}

    # -------------------------------------------------------------------------
    # Thread Repair
    # -------------------------------------------------------------------------

    def repair_thread(self, thread_id: str = "main") -> dict:
        """
        Repair a corrupted thread by fixing orphaned tool_use blocks.

        The Anthropic API requires every tool_use block to have a matching
        tool_result. If tool execution crashes, the thread can be left in
        a corrupted state with tool_use but no tool_result.

        This method scans the thread and adds error tool_results for any
        orphaned tool_use blocks.

        Args:
            thread_id: The thread to repair

        Returns:
            dict with repair statistics
        """
        thread = self.threads.get(thread_id)
        if not thread:
            return {"repaired": 0, "message": "Thread not found or empty"}

        # Find all tool_use IDs that need results
        pending_tool_use_ids = set()
        repaired_count = 0

        for msg in thread:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            pending_tool_use_ids.add(block.get("id"))
                        elif hasattr(block, "type") and block.type == "tool_use":
                            pending_tool_use_ids.add(block.id)

            elif msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            pending_tool_use_ids.discard(block.get("tool_use_id"))

        # If there are orphaned tool_use blocks, add error results
        if pending_tool_use_ids:
            error_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": json.dumps({
                        "error": "Tool execution failed - thread repaired",
                        "repaired": True
                    })
                }
                for tool_id in pending_tool_use_ids
            ]
            thread.append({"role": "user", "content": error_results})
            repaired_count = len(pending_tool_use_ids)

        return {
            "repaired": repaired_count,
            "message": f"Repaired {repaired_count} orphaned tool_use block(s)" if repaired_count else "Thread is healthy"
        }


# =============================================================================
# Core Tools
# =============================================================================

# Shared storage (thread-safe for concurrent access)
MEMORIES: ThreadSafeList = ThreadSafeList()
NOTES: ThreadSafeList = ThreadSafeList()


def _memory_tool(agent: Agent) -> Tool:
    """Memory: store and recall facts."""

    def fn(params: dict, _agent: Agent) -> dict:
        action = params["action"]

        if action == "store":
            memory = {"content": params["content"], "ts": datetime.now().isoformat()}
            MEMORIES.append(memory)
            return {"stored": True}

        elif action == "search":
            query = params.get("query", "").lower()
            matches = [m for m in MEMORIES if query in m["content"].lower()] if query else MEMORIES
            return {"memories": matches[-10:]}

        elif action == "list":
            return {"memories": MEMORIES[-20:]}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="memory",
        description="Store or search memories. Actions: store (content), search (query), list.",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["store", "search", "list"]},
                "content": {"type": "string", "description": "Content to store"},
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["action"]
        },
        fn=fn
    )


def _objective_tool(agent: Agent) -> Tool:
    """Objective management: spawn, list, check background work."""

    def fn(params: dict, ag: Agent) -> dict:
        action = params["action"]

        if action == "spawn":
            obj_id = str(uuid.uuid4())[:8]
            priority = params.get("priority", 5)
            budget_usd = params.get("budget_usd")
            token_limit = params.get("token_limit")
            max_retries = params.get("max_retries", 3)

            obj = Objective(
                id=obj_id,
                goal=params["goal"],
                schedule=params.get("schedule"),
                priority=priority,
                budget_usd=budget_usd,
                token_limit=token_limit,
                max_retries=max_retries
            )
            ag.objectives[obj_id] = obj

            # Add to priority queue
            heapq.heappush(ag._objective_queue, (priority, time.time(), obj_id))

            # Start in background on main event loop (safe from thread pool)
            if not ag._spawn_background_task(ag.run_objective(obj_id)):
                del ag.objectives[obj_id]
                return {"error": "Agent not initialized (no event loop)"}

            return {
                "spawned": obj_id,
                "goal": obj.goal,
                "schedule": obj.schedule,
                "priority": obj.priority,
                "budget_usd": obj.budget_usd,
                "token_limit": obj.token_limit,
                "max_retries": obj.max_retries,
                "message": "Objective started in background. Chat can continue."
            }

        elif action == "list":
            return {"objectives": [
                {
                    "id": o.id,
                    "goal": o.goal,
                    "status": o.status,
                    "priority": o.priority,
                    "schedule": o.schedule,
                    "retry_count": o.retry_count,
                    "spent_usd": round(o.spent_usd, 6),
                    "tokens_used": o.tokens_used,
                    "result": o.result[:200] if o.result else None
                }
                for o in sorted(ag.objectives.values(), key=lambda x: (x.priority, x.created))
            ]}

        elif action == "check":
            obj_id = params["id"]
            obj = ag.objectives.get(obj_id)
            if not obj:
                return {"error": f"Objective {obj_id} not found"}
            return {
                "id": obj.id,
                "goal": obj.goal,
                "status": obj.status,
                "priority": obj.priority,
                "result": obj.result,
                "error": obj.error,
                "retry_count": obj.retry_count,
                "max_retries": obj.max_retries,
                "last_error": obj.last_error,
                "budget_usd": obj.budget_usd,
                "spent_usd": round(obj.spent_usd, 6),
                "token_limit": obj.token_limit,
                "tokens_used": obj.tokens_used,
                "created": obj.created,
                "completed": obj.completed
            }

        elif action == "cancel":
            obj_id = params["id"]
            obj = ag.objectives.get(obj_id)
            if not obj:
                return {"error": f"Objective {obj_id} not found"}

            # Signal cancellation token
            if obj_id in ag._cancellation_tokens:
                ag._cancellation_tokens[obj_id].set()

            obj.status = "cancelled"
            return {
                "cancelled": obj_id,
                "message": "Cancellation signal sent. Running objective will stop at next checkpoint."
            }

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="objective",
        description="""Manage background objectives (async work) with priority, retry, and budget controls.

Actions:
- spawn: Create new objective with options:
  - goal: What to accomplish (required)
  - priority: 1-10, lower = higher priority (default: 5)
  - schedule: Recurring schedule like 'hourly', 'daily', 'every 5 minutes'
  - budget_usd: Maximum cost allowed (stops if exceeded)
  - token_limit: Maximum tokens allowed (stops if exceeded)
  - max_retries: Number of retry attempts on failure (default: 3)
- list: See all objectives sorted by priority, with status and cost tracking
- check: Get full details of specific objective (id)
- cancel: Stop an objective (id) - sends cancellation signal

Max 5 concurrent objectives. Higher priority objectives run first. Failed objectives retry with exponential backoff.""",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["spawn", "list", "check", "cancel"]},
                "goal": {"type": "string", "description": "What to accomplish (for spawn)"},
                "schedule": {"type": "string", "description": "Recurring schedule (for spawn)"},
                "priority": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Priority 1-10, lower = higher (for spawn)"},
                "budget_usd": {"type": "number", "description": "Maximum cost in USD (for spawn)"},
                "token_limit": {"type": "integer", "description": "Maximum tokens allowed (for spawn)"},
                "max_retries": {"type": "integer", "minimum": 0, "maximum": 10, "description": "Retry attempts on failure (for spawn)"},
                "id": {"type": "string", "description": "Objective ID (for check/cancel)"}
            },
            "required": ["action"]
        },
        fn=fn
    )


def _schedule_tool(agent: Agent) -> Tool:
    """Schedule: time-based task automation with full cron support."""

    def fn(params: dict, ag: Agent) -> dict:
        action = params["action"]

        if action == "add":
            name = params.get("name", "Scheduled task")
            goal = params["goal"]
            spec = params["spec"]

            try:
                # Parse schedule specification
                if isinstance(spec, str):
                    schedule = parse_schedule(spec)
                else:
                    schedule = Schedule(**spec)

                task = ScheduledTask(
                    id=str(uuid.uuid4())[:8],
                    name=name,
                    goal=goal,
                    schedule=schedule
                )
                ag.scheduler.add(task)

                return {
                    "scheduled": task.id,
                    "name": task.name,
                    "goal": task.goal,
                    "schedule": task.schedule.human_readable(),
                    "next_run": task.next_run_at,
                    "message": f"Task scheduled: {task.schedule.human_readable()}"
                }
            except Exception as e:
                return {"error": f"Invalid schedule: {e}"}

        elif action == "list":
            include_disabled = params.get("include_disabled", False)
            tasks = ag.scheduler.list(include_disabled=include_disabled)
            return {"tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "goal": t.goal[:100],
                    "schedule": t.schedule.human_readable(),
                    "next_run": t.next_run_at,
                    "last_status": t.last_status,
                    "run_count": t.run_count,
                    "enabled": t.enabled
                }
                for t in tasks
            ]}

        elif action == "get":
            task_id = params["id"]
            task = ag.scheduler.get(task_id)
            if not task:
                return {"error": f"Task {task_id} not found"}
            return {
                "id": task.id,
                "name": task.name,
                "goal": task.goal,
                "schedule": task.schedule.human_readable(),
                "next_run": task.next_run_at,
                "last_run": task.last_run_at,
                "last_status": task.last_status,
                "last_error": task.last_error,
                "run_count": task.run_count,
                "enabled": task.enabled
            }

        elif action == "update":
            task_id = params["id"]
            updates = {}
            if "name" in params:
                updates["name"] = params["name"]
            if "goal" in params:
                updates["goal"] = params["goal"]
            if "enabled" in params:
                updates["enabled"] = params["enabled"]
            if "spec" in params:
                spec = params["spec"]
                if isinstance(spec, str):
                    updates["schedule"] = parse_schedule(spec)
                else:
                    updates["schedule"] = Schedule(**spec)

            task = ag.scheduler.update(task_id, **updates)
            if not task:
                return {"error": f"Task {task_id} not found"}
            return {
                "updated": task.id,
                "schedule": task.schedule.human_readable(),
                "next_run": task.next_run_at
            }

        elif action == "remove":
            task_id = params["id"]
            if ag.scheduler.remove(task_id):
                return {"removed": task_id}
            return {"error": f"Task {task_id} not found"}

        elif action == "run":
            task_id = params["id"]
            task = ag.scheduler.get(task_id)
            if not task:
                return {"error": f"Task {task_id} not found"}

            # Spawn task execution in background on main event loop (safe from thread pool)
            if not ag._spawn_background_task(ag.scheduler.run_now(task_id, force=True)):
                return {"error": "Agent not initialized (no event loop)"}
            return {
                "status": "triggered",
                "task_id": task_id,
                "name": task.name,
                "message": "Task execution started in background"
            }

        elif action == "history":
            task_id = params["id"]
            limit = params.get("limit", 10)
            runs = ag.scheduler.get_runs(task_id, limit=limit)
            return {"runs": [
                {
                    "started_at": r.started_at,
                    "status": r.status,
                    "duration_ms": r.duration_ms,
                    "result": r.result[:200] if r.result else None,
                    "error": r.error
                }
                for r in runs
            ]}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="schedule",
        description="""Schedule tasks to run at specific times or intervals.

ACTIONS:
- add: Create scheduled task (name, goal, spec)
- list: View all scheduled tasks
- get: Get task details (id)
- update: Modify task (id, name?, goal?, spec?, enabled?)
- remove: Delete task (id)
- run: Execute task now (id, force=True)
- history: View execution history (id, limit=10)

SCHEDULE SPEC (string shortcuts):
- "in 5m" or "in 2h" → one-time, relative
- "every 5m" or "every 1h" → recurring interval
- "daily at 9:00" → daily at specific time
- "weekdays at 9am" → weekdays only
- "hourly" or "daily" → simple patterns
- "0 9 * * 1-5" → raw cron expression

SCHEDULE SPEC (full control):
{"kind": "at", "at": "2024-01-15T09:00:00", "tz": "America/New_York"}
{"kind": "every", "every": "5m"}
{"kind": "cron", "cron": "0 9 * * 1-5", "tz": "America/New_York"}

WHEN TO USE:
- User wants reminder: "in 30m", "at 3pm"
- User wants recurring check: "every 5m", "hourly"
- User wants daily/weekly automation: "daily at 9:00", "weekdays at 9am"

Tasks persist across restarts and run autonomously.""",
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "get", "update", "remove", "run", "history"]
                },
                "name": {
                    "type": "string",
                    "description": "Human-readable task name"
                },
                "goal": {
                    "type": "string",
                    "description": "What to accomplish when task runs"
                },
                "spec": {
                    "description": "Schedule: string shortcut or {kind, at/every/cron, tz}"
                },
                "id": {
                    "type": "string",
                    "description": "Task ID for get/update/remove/run/history"
                },
                "enabled": {
                    "type": "boolean",
                    "description": "Enable/disable task (for update)"
                },
                "force": {
                    "type": "boolean",
                    "description": "Force run even if not due (for run)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max history entries to return"
                },
                "include_disabled": {
                    "type": "boolean",
                    "description": "Include disabled tasks in list"
                }
            },
            "required": ["action"]
        },
        fn=fn
    )


def _notes_tool(agent: Agent) -> Tool:
    """Notes: simple todo list (passive, unlike objectives)."""

    def fn(params: dict, _agent: Agent) -> dict:
        action = params["action"]

        if action == "add":
            note = {
                "id": len(NOTES),
                "text": params["text"],
                "done": False,
                "created": datetime.now().isoformat()
            }
            NOTES.append(note)
            return {"added": note}

        elif action == "list":
            return {"notes": NOTES.copy()}

        elif action == "complete":
            note_id = params["id"]
            if 0 <= note_id < len(NOTES):
                NOTES[note_id]["done"] = True
                return {"completed": NOTES[note_id]}
            return {"error": f"Note {note_id} not found"}

        elif action == "remove":
            note_id = params["id"]
            if 0 <= note_id < len(NOTES):
                removed = NOTES.pop(note_id)
                for i, note in enumerate(NOTES):
                    note["id"] = i
                return {"removed": removed}
            return {"error": f"Note {note_id} not found"}

        return {"error": f"Unknown action: {action}"}

    return Tool(
        name="notes",
        description="Simple notes/todos. Actions: add (text), list, complete (id), remove (id).",
        parameters={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "list", "complete", "remove"]},
                "text": {"type": "string", "description": "Note text (for add)"},
                "id": {"type": "integer", "description": "Note ID (for complete/remove)"}
            },
            "required": ["action"]
        },
        fn=fn
    )


def _register_tool_tool(agent: Agent) -> Tool:
    """Register new tools at runtime with e2b sandbox for external packages."""

    def fn(params: dict, ag: Agent) -> dict:
        code = params["code"]
        tool_var_name = params["tool_var_name"]

        # Detect imports to determine if we need sandboxing
        try:
            from tools.sandbox import detect_imports, run_in_sandbox
            packages = detect_imports(code)
        except ImportError:
            packages = []

        if packages:
            # External dependencies detected - use e2b sandbox
            sandbox_code = f"""
{code}

# Export tool definition as dict for serialization
_tool = {tool_var_name}
result = {{
    "name": _tool.name,
    "description": _tool.schema["description"],
    "parameters": _tool.schema["input_schema"],
}}
result
"""
            result = run_in_sandbox(sandbox_code, packages)

            if "error" in result:
                return {"error": result["error"]}

            # Create a wrapper that executes in sandbox
            tool_def = eval(result["result"])  # Safe: we control sandbox output

            def sandboxed_executor(exec_code, pkgs):
                def executor(params: dict, _ag: Agent) -> dict:
                    full_code = f"""
{exec_code}
import json
_tool = {tool_var_name}
_result = _tool.fn({json.dumps(params)}, None)
json.dumps(_result)
"""
                    res = run_in_sandbox(full_code, pkgs)
                    if "error" in res:
                        return {"error": res["error"]}
                    return json.loads(res["result"])
                return executor

            new_tool = Tool(
                name=tool_def["name"],
                description=tool_def["description"],
                parameters=tool_def["parameters"],
                fn=sandboxed_executor(code, packages),
                packages=packages,
            )
            # Register with persistence enabled
            ag.register(
                new_tool,
                emit_event=True,
                source_code=code,
                tool_var_name=tool_var_name,
                category="custom",
                is_dynamic=True,
            )
            return {
                "registered": new_tool.name,
                "description": new_tool.schema["description"],
                "sandboxed": True,
                "packages": packages,
                "persisted": True
            }

        # No external dependencies - run locally (fast path)
        namespace = {
            "Tool": Tool,
            "datetime": datetime,
            "json": json,
        }

        try:
            exec(code, namespace)
            new_tool = namespace[tool_var_name]
            # Register with persistence enabled
            ag.register(
                new_tool,
                emit_event=True,
                source_code=code,
                tool_var_name=tool_var_name,
                category="custom",
                is_dynamic=True,
            )
            return {
                "registered": new_tool.name,
                "description": new_tool.schema["description"],
                "sandboxed": False,
                "persisted": True
            }
        except ToolValidationError as e:
            # Provide clear guidance on how to fix
            return {
                "error": str(e),
                "hint": "Use the Tool class: Tool(name='...', description='...', parameters={...}, fn=your_function)"
            }
        except KeyError:
            return {
                "error": f"Variable '{tool_var_name}' not found in code",
                "hint": f"Ensure your code defines a variable named '{tool_var_name}'"
            }
        except Exception as e:
            return {"error": str(e)}

    return Tool(
        name="register_tool",
        description="""Register a new tool by providing Python code.

Tools you create are PERSISTED and will be available after restart. This enables
self-improvement: build tools for yourself that persist across sessions.

IMPORTANT: You MUST use the provided Tool class to define your tool:

    def my_fn(params: dict, agent) -> dict:
        return {"result": params["input"]}

    my_tool = Tool(
        name="my_tool",
        description="Does something useful",
        parameters={
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"]
        },
        fn=my_fn
    )

The Tool class is pre-loaded in the namespace. External packages (requests, pandas, etc.)
are auto-detected and installed in a sandbox.

Tool execution is automatically tracked: success/error counts, duration, and usage statistics
are recorded for monitoring and debugging.""",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code using the Tool class to define a tool instance"
                },
                "tool_var_name": {
                    "type": "string",
                    "description": "Variable name of the Tool instance in the code"
                }
            },
            "required": ["code", "tool_var_name"]
        },
        fn=fn
    )


def _send_message_tool(agent: Agent) -> Tool:
    """Send messages across any configured channel."""

    def fn(params: dict, ag: Agent) -> dict:
        import asyncio

        channel = params["channel"]
        to = params["to"]
        content = params["content"]

        # Check if channel is configured
        if channel not in ag.senders:
            available = list(ag.senders.keys()) if ag.senders else ["none configured"]
            return {
                "error": f"Channel '{channel}' not configured",
                "available_channels": available
            }

        # Resolve "owner" to actual contact
        if to == "owner":
            owner_contacts = ag.config.get("owner", {}).get("contacts", {})
            to = owner_contacts.get(channel)
            if not to:
                return {
                    "error": f"No owner contact configured for {channel}",
                    "hint": "Set owner.contacts in config"
                }

        # Get sender and validate capabilities
        sender = ag.senders[channel]

        # Check capability requirements
        if params.get("attachments") and "attachments" not in getattr(sender, 'capabilities', []):
            return {"error": f"{channel} doesn't support attachments"}
        if params.get("image") and "images" not in getattr(sender, 'capabilities', []):
            return {"error": f"{channel} doesn't support images"}

        # Build kwargs from optional params
        kwargs = {}
        for key in ["subject", "attachments", "image", "reply_to"]:
            if params.get(key):
                kwargs[key] = params[key]

        # Send via the main event loop (tools run in thread pool)
        async def do_send():
            try:
                return await sender.send(to, content, **kwargs)
            except Exception as e:
                return {"error": str(e)}

        # Use the agent's main loop to schedule the async send
        if ag._main_loop is None:
            return {"error": "Agent not initialized (no event loop)"}

        future = asyncio.run_coroutine_threadsafe(do_send(), ag._main_loop)
        try:
            return future.result(timeout=30)
        except Exception as e:
            return {"error": f"Send failed: {e}"}

    return Tool(
        name="send_message",
        description="""Send a message via any configured channel.

Use this to communicate across channels - email, SMS, WhatsApp, etc.
Use to="owner" to message your owner on that channel.

Examples:
- send_message(channel="email", to="alice@example.com", subject="Hi", content="Hello!")
- send_message(channel="email", to="owner", content="Task complete!")
- send_message(channel="sms", to="owner", content="Urgent: need your input")

Note: Channels must be configured in the agent. Check available_channels in error responses.""",
        parameters={
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Channel to send via (email, sms, whatsapp, etc.)"
                },
                "to": {
                    "type": "string",
                    "description": "Recipient identifier or 'owner'"
                },
                "content": {
                    "type": "string",
                    "description": "Message content"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject (email only)"
                },
                "attachments": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Attachment URLs (if supported)"
                },
                "image": {
                    "type": "string",
                    "description": "Image URL (WhatsApp/Telegram)"
                },
                "reply_to": {
                    "type": "string",
                    "description": "Message ID to reply to"
                }
            },
            "required": ["channel", "to", "content"]
        },
        fn=fn
    )


# =============================================================================
# CLI
# =============================================================================

async def main_async():
    """Async REPL with background objective support."""
    print("BabyAGI v0.2.2")
    print("=" * 40)

    agent = Agent()

    # Start scheduler for recurring objectives
    scheduler_task = asyncio.create_task(agent.run_scheduler())

    # Get tool health and generate AI greeting
    try:
        from tools import get_health_summary
        health_summary = get_health_summary()
    except Exception as e:
        logger.debug("Could not get tool health summary: %s", e)
        health_summary = "Core tools ready."

    # Generate personalized greeting from the AI
    greeting_prompt = f"""Generate a brief, friendly greeting (2-3 sentences max).

Tool Status:
{health_summary}

Be concise. Mention what you can help with based on available tools. If any tools need setup, briefly note it. End with an invitation to chat."""

    try:
        greeting = await agent.client.messages.create(
            model=agent.model,
            max_tokens=200,
            messages=[{"role": "user", "content": greeting_prompt}]
        )
        greeting_text = "".join(b.text for b in greeting.content if hasattr(b, "text"))
        print(f"\n{greeting_text}\n")
    except Exception as e:
        logger.debug("Could not generate AI greeting: %s", e)
        print("\nReady to assist. Type 'quit' to exit.\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "You: ")
            user_input = user_input.strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not user_input:
                continue

            response = await agent.run_async(user_input)
            print(f"Assistant: {response}\n")

    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        scheduler_task.cancel()


# ═══════════════════════════════════════════════════════════
# GLOBAL METRICS ACCESSOR
# ═══════════════════════════════════════════════════════════

_metrics_collector_instance: MetricsCollector | None = None


def _set_metrics_collector(collector: MetricsCollector):
    """Set the global metrics collector instance."""
    global _metrics_collector_instance
    _metrics_collector_instance = collector


def _get_metrics_collector() -> MetricsCollector | None:
    """Get the global metrics collector instance for tool access."""
    return _metrics_collector_instance


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
