# BabyAGI v0.3.0 Architecture

This document provides a comprehensive visual overview of the BabyAGI codebase architecture using Mermaid diagrams.

## High-Level System Architecture

```mermaid
flowchart TB
    subgraph Entry["Entry Points (main.py)"]
        CLI_MODE["CLI Mode<br/>python main.py"]
        CHANNELS_MODE["Channels Mode<br/>python main.py channels"]
        SERVER_MODE["Server Mode<br/>python main.py serve"]
    end

    subgraph Config["Configuration"]
        CONFIG_YAML["config.yaml"]
        ENV_VARS["Environment Variables"]
        CONFIG_PY["config.py<br/>ConfigLoader"]
    end

    subgraph Core["Agent Core (agent.py)"]
        AGENT["Agent Class"]
        THREADS["threads: Dict[str, List[Message]]"]
        OBJECTIVES["objectives: Dict[str, Objective]"]
        SCHEDULER["Scheduler<br/>(scheduler.py)"]
        EVENT_EMITTER["EventEmitter<br/>(utils/events.py)"]
    end

    subgraph Listeners["Input Channels (listeners/)"]
        CLI_LISTENER["cli.py<br/>CLI REPL"]
        EMAIL_LISTENER["email.py<br/>Email Poller"]
        VOICE_LISTENER["voice.py<br/>Voice Input"]
    end

    subgraph Senders["Output Channels (senders/)"]
        CLI_SENDER["cli.py<br/>Terminal Output"]
        EMAIL_SENDER["email.py<br/>Email Output"]
    end

    subgraph Tools["Tools System (tools/)"]
        TOOL_DECORATOR["@tool Decorator"]
        WEB_TOOLS["web.py<br/>search, browse, fetch"]
        EMAIL_TOOLS["email.py<br/>send, check inbox"]
        SANDBOX["sandbox.py<br/>Code Execution (E2B)"]
        SECRETS["secrets.py<br/>Credential Storage"]
        VERBOSE["verbose.py<br/>Logging Control"]
    end

    subgraph Memory["Memory System (memory/)"]
        MEMORY_FACADE["Memory Facade<br/>(__init__.py)"]
        STORE["MemoryStore<br/>(store.py)"]
        EXTRACTION["ExtractionPipeline<br/>(extraction.py)"]
        SUMMARIES["SummaryManager<br/>(summaries.py)"]
        CONTEXT["ContextAssembler<br/>(context.py)"]
        RETRIEVAL["QuickRetrieval<br/>(retrieval.py)"]
        EMBEDDINGS["Embeddings<br/>(embeddings.py)"]
    end

    subgraph External["External Services"]
        CLAUDE["Anthropic Claude API"]
        E2B["E2B Sandbox"]
        AGENTMAIL["AgentMail API"]
        DUCKDUCKGO["DuckDuckGo Search"]
        BROWSER_USE["Browser Use Cloud"]
    end

    subgraph Storage["Data Storage"]
        SQLITE["SQLite Database"]
        KEYRING["Encrypted Keyring"]
    end

    %% Entry flow
    CLI_MODE --> AGENT
    CHANNELS_MODE --> AGENT
    SERVER_MODE --> AGENT

    %% Config flow
    CONFIG_YAML --> CONFIG_PY
    ENV_VARS --> CONFIG_PY
    CONFIG_PY --> AGENT

    %% Agent connections
    AGENT --> THREADS
    AGENT --> OBJECTIVES
    AGENT --> SCHEDULER
    AGENT -.-> EVENT_EMITTER
    AGENT --> CLAUDE

    %% Listeners to Agent
    CLI_LISTENER --> AGENT
    EMAIL_LISTENER --> AGENT
    VOICE_LISTENER --> AGENT

    %% Agent to Senders
    AGENT --> CLI_SENDER
    AGENT --> EMAIL_SENDER

    %% Tools registration
    TOOL_DECORATOR --> AGENT
    WEB_TOOLS --> TOOL_DECORATOR
    EMAIL_TOOLS --> TOOL_DECORATOR
    SANDBOX --> TOOL_DECORATOR
    SECRETS --> TOOL_DECORATOR
    VERBOSE --> TOOL_DECORATOR

    %% Memory system
    AGENT --> MEMORY_FACADE
    MEMORY_FACADE --> STORE
    MEMORY_FACADE --> EXTRACTION
    MEMORY_FACADE --> SUMMARIES
    MEMORY_FACADE --> CONTEXT
    MEMORY_FACADE --> RETRIEVAL
    MEMORY_FACADE --> EMBEDDINGS
    STORE --> SQLITE

    %% External services
    WEB_TOOLS --> DUCKDUCKGO
    WEB_TOOLS --> BROWSER_USE
    EMAIL_TOOLS --> AGENTMAIL
    EMAIL_LISTENER --> AGENTMAIL
    EMAIL_SENDER --> AGENTMAIL
    SANDBOX --> E2B
    SECRETS --> KEYRING

    %% Event subscriptions
    EVENT_EMITTER -.-> CLI_LISTENER
    EVENT_EMITTER -.-> MEMORY_FACADE
```

## Message Processing Flow

```mermaid
sequenceDiagram
    participant User
    participant Listener as Listener<br/>(CLI/Email/Voice)
    participant Agent
    participant Claude as Claude API
    participant Tools
    participant Memory
    participant Sender as Sender<br/>(CLI/Email)

    User->>Listener: Input message
    Listener->>Agent: run_async(content, thread_id, context)

    Note over Agent: Acquire thread lock
    Agent->>Agent: Add message to thread
    Agent->>Memory: Get context (if enabled)
    Memory-->>Agent: Relevant memories
    Agent->>Agent: Build system prompt

    loop Until stop_reason == "end_turn"
        Agent->>Claude: API call with messages + tools
        Claude-->>Agent: Response (text + tool_use)

        alt Tool Use Block
            Agent->>Agent: emit("tool_start")
            Agent->>Tools: Execute tool
            Tools-->>Agent: Tool result
            Agent->>Agent: emit("tool_end")
            Agent->>Agent: Add tool result to messages
        else Text Block
            Agent->>Agent: Prepare response text
        end
    end

    Agent->>Memory: Log conversation event
    Agent->>Sender: Send response
    Sender->>User: Display/Deliver response

    Note over Agent: Release thread lock
```

## Memory System Architecture

```mermaid
flowchart TB
    subgraph Events["Event Logging"]
        LOG_EVENT["log_event()"]
        EVENT_MODEL["Event Model<br/>- id, timestamp<br/>- channel, direction<br/>- event_type<br/>- content<br/>- embedding"]
    end

    subgraph Storage["SQLite Storage (store.py)"]
        EVENTS_TABLE["events table"]
        ENTITIES_TABLE["entities table<br/>(People, Orgs, Concepts)"]
        EDGES_TABLE["edges table<br/>(Relationships)"]
        TOPICS_TABLE["topics table"]
        TASKS_TABLE["tasks table"]
        SUMMARIES_TABLE["summaries table<br/>(Hierarchical)"]
    end

    subgraph Extraction["Extraction Pipeline (extraction.py)"]
        EXTRACT_ENTITIES["Extract Entities"]
        EXTRACT_EDGES["Extract Relationships"]
        EXTRACT_TOPICS["Extract Topics"]
    end

    subgraph Summaries["Summary System (summaries.py)"]
        SUMMARY_TREE["Hierarchical Summary Tree"]
        ROLLUP["Periodic Rollup"]
        REFRESH["Stale Summary Refresh"]
    end

    subgraph Retrieval["Retrieval (retrieval.py, context.py)"]
        QUICK_RETRIEVAL["Quick Retrieval"]
        SEMANTIC_SEARCH["Semantic Search"]
        CONTEXT_ASSEMBLY["Context Assembly"]
    end

    subgraph Embeddings["Embeddings (embeddings.py)"]
        EMBED_CACHE["Embedding Cache"]
        OPENAI_EMBED["OpenAI Embeddings"]
        SQLITE_VEC["sqlite-vec Extension"]
    end

    %% Event logging flow
    LOG_EVENT --> EVENT_MODEL
    EVENT_MODEL --> EVENTS_TABLE

    %% Extraction flow
    EVENTS_TABLE --> EXTRACT_ENTITIES
    EXTRACT_ENTITIES --> ENTITIES_TABLE
    EXTRACT_ENTITIES --> EXTRACT_EDGES
    EXTRACT_EDGES --> EDGES_TABLE
    EXTRACT_ENTITIES --> EXTRACT_TOPICS
    EXTRACT_TOPICS --> TOPICS_TABLE

    %% Summary flow
    EVENTS_TABLE --> SUMMARY_TREE
    ENTITIES_TABLE --> SUMMARY_TREE
    SUMMARY_TREE --> ROLLUP
    ROLLUP --> SUMMARIES_TABLE
    ROLLUP --> REFRESH

    %% Retrieval flow
    ENTITIES_TABLE --> QUICK_RETRIEVAL
    EDGES_TABLE --> QUICK_RETRIEVAL
    SUMMARIES_TABLE --> CONTEXT_ASSEMBLY
    EVENTS_TABLE --> SEMANTIC_SEARCH
    SEMANTIC_SEARCH --> CONTEXT_ASSEMBLY

    %% Embeddings
    EVENT_MODEL --> EMBED_CACHE
    EMBED_CACHE --> OPENAI_EMBED
    EMBED_CACHE --> SQLITE_VEC
    SQLITE_VEC --> SEMANTIC_SEARCH
```

## Tool System Architecture

```mermaid
flowchart LR
    subgraph Registration["Tool Registration"]
        DECORATOR["@tool decorator<br/>- packages: list<br/>- env: list"]
        SCHEMA_GEN["JSON Schema<br/>Generator"]
        DOCSTRING["Docstring Parser"]
    end

    subgraph BuiltIn["Built-in Tools (agent.py)"]
        MEMORY_TOOL["memory<br/>Query memories"]
        OBJECTIVE_TOOL["objective<br/>Background tasks"]
        NOTES_TOOL["notes<br/>Save/retrieve notes"]
        SCHEDULE_TOOL["schedule<br/>Create schedules"]
        REGISTER_TOOL["register_tool<br/>Dynamic registration"]
        SEND_MSG_TOOL["send_message<br/>Multi-channel output"]
    end

    subgraph External["External Tools (tools/)"]
        WEB_SEARCH["web_search<br/>DuckDuckGo"]
        BROWSE["browse<br/>Browser automation"]
        FETCH_URL["fetch_url<br/>HTTP requests"]
        SEND_EMAIL["send_email<br/>Email sending"]
        GET_INBOX["get_email_inbox<br/>Inbox polling"]
        GET_SECRET["get_secret<br/>Credential retrieval"]
        SET_SECRET["set_secret<br/>Credential storage"]
        EXEC_CODE["execute_code<br/>E2B sandbox"]
    end

    subgraph Execution["Tool Execution"]
        TOOL_EXECUTOR["Tool Executor"]
        THREAD_POOL["Thread Pool"]
        RESULT_HANDLER["Result Handler"]
    end

    DECORATOR --> SCHEMA_GEN
    DECORATOR --> DOCSTRING

    MEMORY_TOOL --> TOOL_EXECUTOR
    OBJECTIVE_TOOL --> TOOL_EXECUTOR
    NOTES_TOOL --> TOOL_EXECUTOR
    SCHEDULE_TOOL --> TOOL_EXECUTOR
    REGISTER_TOOL --> TOOL_EXECUTOR
    SEND_MSG_TOOL --> TOOL_EXECUTOR

    WEB_SEARCH --> DECORATOR
    BROWSE --> DECORATOR
    FETCH_URL --> DECORATOR
    SEND_EMAIL --> DECORATOR
    GET_INBOX --> DECORATOR
    GET_SECRET --> DECORATOR
    SET_SECRET --> DECORATOR
    EXEC_CODE --> DECORATOR

    TOOL_EXECUTOR --> THREAD_POOL
    THREAD_POOL --> RESULT_HANDLER
```

## Scheduler and Background Objectives

```mermaid
flowchart TB
    subgraph Scheduler["Scheduler (scheduler.py)"]
        SCHEDULER_LOOP["Scheduler Loop<br/>run_scheduler()"]
        TASK_QUEUE["Scheduled Tasks Queue"]
        SCHEDULE_TYPES["Schedule Types<br/>- at: ISO timestamp<br/>- every: interval<br/>- cron: expression"]
    end

    subgraph Objectives["Background Objectives"]
        CREATE_OBJ["Create Objective<br/>(objective tool)"]
        OBJ_THREAD["Dedicated Thread"]
        OBJ_STATUS["Status Tracking<br/>pending → running → completed/failed"]
    end

    subgraph Execution["Execution"]
        AGENT_RUN["agent.run_async()"]
        EMIT_START["emit('task_start')"]
        EMIT_END["emit('task_end')"]
    end

    subgraph Persistence["Task State"]
        TASK_MODEL["ScheduledTask Model<br/>- id, name, goal<br/>- schedule<br/>- thread_id<br/>- next_run"]
    end

    %% Scheduler flow
    SCHEDULER_LOOP --> TASK_QUEUE
    TASK_QUEUE --> SCHEDULE_TYPES

    %% Check and execute
    SCHEDULER_LOOP -->|"next_run <= now"| EMIT_START
    EMIT_START --> AGENT_RUN
    AGENT_RUN --> EMIT_END

    %% Objectives
    CREATE_OBJ --> OBJ_THREAD
    OBJ_THREAD --> OBJ_STATUS
    OBJ_THREAD --> AGENT_RUN

    %% Persistence
    TASK_MODEL --> TASK_QUEUE
    SCHEDULE_TYPES --> TASK_MODEL
```

## Event System

```mermaid
flowchart TB
    subgraph EventEmitter["EventEmitter (utils/events.py)"]
        ON["on(event, handler)"]
        OFF["off(event, handler)"]
        EMIT["emit(event, data)"]
        ONCE["once(event, handler)"]
    end

    subgraph AgentEvents["Agent Events"]
        TOOL_START["tool_start<br/>{name, input}"]
        TOOL_END["tool_end<br/>{name, result, duration_ms}"]
        OBJ_START["objective_start<br/>{id, goal}"]
        OBJ_END["objective_end<br/>{id, status, result}"]
        TASK_START["task_start<br/>{id, name, goal}"]
        TASK_END["task_end<br/>{id, status, duration_ms}"]
        RESPONSE["agent_response<br/>{content, thread_id}"]
    end

    subgraph Subscribers["Event Subscribers"]
        CLI_HANDLER["CLI Handler<br/>Display in terminal"]
        MEMORY_HOOKS["Memory Integration<br/>Log to SQLite"]
        CUSTOM["Custom Handlers"]
    end

    %% EventEmitter methods
    ON --> EMIT
    ONCE --> EMIT
    OFF -.-> EMIT

    %% Events flow to subscribers
    TOOL_START --> EMIT
    TOOL_END --> EMIT
    OBJ_START --> EMIT
    OBJ_END --> EMIT
    TASK_START --> EMIT
    TASK_END --> EMIT
    RESPONSE --> EMIT

    EMIT --> CLI_HANDLER
    EMIT --> MEMORY_HOOKS
    EMIT --> CUSTOM
```

## Multi-Channel Architecture

```mermaid
flowchart TB
    subgraph Channels["Communication Channels"]
        subgraph Input["Input Listeners"]
            CLI_IN["CLI Listener<br/>(stdin REPL)"]
            EMAIL_IN["Email Listener<br/>(AgentMail polling)"]
            VOICE_IN["Voice Listener<br/>(Whisper STT)"]
            API_IN["API Server<br/>(FastAPI endpoints)"]
        end

        subgraph Output["Output Senders"]
            CLI_OUT["CLI Sender<br/>(styled terminal)"]
            EMAIL_OUT["Email Sender<br/>(AgentMail API)"]
            VOICE_OUT["Voice Output<br/>(TTS)"]
            API_OUT["API Response<br/>(JSON)"]
        end
    end

    subgraph Agent["Agent Core"]
        AGENT_CORE["Agent"]
        SENDERS_REG["Senders Registry<br/>Dict[channel, Sender]"]
        CONTEXT["Current Context<br/>{channel, is_owner, sender}"]
    end

    subgraph Concurrency["Concurrency Control"]
        THREAD_LOCKS["Per-Thread Locks<br/>Dict[thread_id, Lock]"]
        ASYNCIO["asyncio.gather()"]
        THREAD_SAFE["ThreadSafeList<br/>(MEMORIES, NOTES)"]
    end

    %% Input to Agent
    CLI_IN --> AGENT_CORE
    EMAIL_IN --> AGENT_CORE
    VOICE_IN --> AGENT_CORE
    API_IN --> AGENT_CORE

    %% Agent to Output
    AGENT_CORE --> SENDERS_REG
    SENDERS_REG --> CLI_OUT
    SENDERS_REG --> EMAIL_OUT
    SENDERS_REG --> VOICE_OUT
    SENDERS_REG --> API_OUT

    %% Context tracking
    CLI_IN --> CONTEXT
    EMAIL_IN --> CONTEXT
    VOICE_IN --> CONTEXT
    API_IN --> CONTEXT
    CONTEXT --> AGENT_CORE

    %% Concurrency
    AGENT_CORE --> THREAD_LOCKS
    CLI_IN --> ASYNCIO
    EMAIL_IN --> ASYNCIO
    VOICE_IN --> ASYNCIO
    AGENT_CORE --> THREAD_SAFE
```

## Data Models

```mermaid
classDiagram
    class Agent {
        +Dict threads
        +Dict objectives
        +Dict tools
        +Dict senders
        +Scheduler scheduler
        +Memory memory
        +run_async(content, thread_id, context)
        +register(tool)
        +register_sender(channel, sender)
        +run_scheduler()
    }

    class Objective {
        +str id
        +str goal
        +str status
        +str thread_id
        +str schedule
        +str result
        +str error
    }

    class Tool {
        +str name
        +Callable fn
        +dict schema
        +list packages
        +list env
        +execute(params, agent)
    }

    class ScheduledTask {
        +str id
        +str name
        +str goal
        +Schedule schedule
        +str thread_id
        +datetime next_run
    }

    class Schedule {
        +str kind
        +str at
        +str every
        +str cron
        +str tz
    }

    class Event {
        +str id
        +datetime timestamp
        +str channel
        +str direction
        +str event_type
        +str task_id
        +str tool_id
        +str person_id
        +bool is_owner
        +str content
        +list content_embedding
    }

    class Entity {
        +str id
        +str name
        +str type
        +str summary
        +datetime last_updated
    }

    class Edge {
        +str id
        +str source_id
        +str target_id
        +str relation
        +float weight
    }

    class SummaryNode {
        +str id
        +str parent_id
        +int level
        +str content
        +datetime start_time
        +datetime end_time
    }

    Agent "1" --> "*" Objective
    Agent "1" --> "*" Tool
    Agent "1" --> "1" Scheduler
    Scheduler "1" --> "*" ScheduledTask
    ScheduledTask "1" --> "1" Schedule
    Agent "1" --> "1" Memory
    Memory "1" --> "*" Event
    Memory "1" --> "*" Entity
    Memory "1" --> "*" Edge
    Memory "1" --> "*" SummaryNode
```

## File Structure Overview

```mermaid
flowchart TB
    subgraph Root["babyagi/"]
        MAIN["main.py<br/>Entry point"]
        AGENT_PY["agent.py<br/>Core Agent class"]
        SCHEDULER_PY["scheduler.py<br/>Task scheduling"]
        CONFIG_PY["config.py<br/>Config loader"]
        CONFIG_YAML["config.yaml<br/>Settings"]
        SERVER_PY["server.py<br/>FastAPI server"]
    end

    subgraph MemoryDir["memory/"]
        MEM_INIT["__init__.py<br/>Memory facade"]
        STORE_PY["store.py<br/>SQLite backend"]
        MODELS_PY["models.py<br/>Data models"]
        CONTEXT_PY["context.py<br/>Context assembly"]
        EXTRACT_PY["extraction.py<br/>NLP pipeline"]
        RETRIEVAL_PY["retrieval.py<br/>Search"]
        SUMMARIES_PY["summaries.py<br/>Summary tree"]
        EMBED_PY["embeddings.py<br/>Vector embeddings"]
        INTEG_PY["integration.py<br/>Event hooks"]
    end

    subgraph ListenersDir["listeners/"]
        L_CLI["cli.py<br/>CLI REPL"]
        L_EMAIL["email.py<br/>Email poller"]
        L_VOICE["voice.py<br/>Voice input"]
    end

    subgraph SendersDir["senders/"]
        S_INIT["__init__.py<br/>Sender protocol"]
        S_CLI["cli.py<br/>Terminal output"]
        S_EMAIL["email.py<br/>Email output"]
    end

    subgraph ToolsDir["tools/"]
        T_INIT["__init__.py<br/>Tool framework"]
        T_WEB["web.py<br/>Web tools"]
        T_EMAIL["email.py<br/>Email tools"]
        T_SANDBOX["sandbox.py<br/>Code execution"]
        T_SECRETS["secrets.py<br/>Credentials"]
        T_VERBOSE["verbose.py<br/>Logging"]
    end

    subgraph UtilsDir["utils/"]
        U_EVENTS["events.py<br/>EventEmitter"]
        U_CONSOLE["console.py<br/>Terminal styling"]
    end

    MAIN --> AGENT_PY
    MAIN --> SERVER_PY
    AGENT_PY --> SCHEDULER_PY
    AGENT_PY --> CONFIG_PY
    CONFIG_PY --> CONFIG_YAML
    AGENT_PY --> MEM_INIT
    AGENT_PY --> L_CLI
    AGENT_PY --> T_INIT
    AGENT_PY --> U_EVENTS
```

## Complete System Flow

```mermaid
flowchart TB
    START((Start)) --> MODE{Mode?}

    MODE -->|"cli"| CLI_START["Initialize Agent"]
    MODE -->|"channels"| CHAN_START["Initialize Agent"]
    MODE -->|"serve"| SERVER_START["Initialize FastAPI"]

    CLI_START --> LOAD_CONFIG["Load config.yaml"]
    CHAN_START --> LOAD_CONFIG
    SERVER_START --> LOAD_CONFIG

    LOAD_CONFIG --> INIT_AGENT["Create Agent instance"]
    INIT_AGENT --> LOAD_TOOLS["Load tools from tools/"]
    LOAD_TOOLS --> INIT_MEMORY{"Memory enabled?"}

    INIT_MEMORY -->|Yes| SETUP_MEMORY["Initialize SQLite Memory"]
    INIT_MEMORY -->|No| SKIP_MEMORY["Use in-memory fallback"]

    SETUP_MEMORY --> START_BG["Start background extraction"]
    SKIP_MEMORY --> START_LISTENERS
    START_BG --> START_LISTENERS

    START_LISTENERS --> GATHER["asyncio.gather()"]

    GATHER --> CLI_LOOP["CLI Listener Loop"]
    GATHER --> EMAIL_LOOP["Email Listener Loop"]
    GATHER --> VOICE_LOOP["Voice Listener Loop"]
    GATHER --> SCHED_LOOP["Scheduler Loop"]

    CLI_LOOP --> RECV_INPUT["Receive user input"]
    EMAIL_LOOP --> POLL_EMAIL["Poll email inbox"]
    VOICE_LOOP --> LISTEN_VOICE["Listen for voice"]
    SCHED_LOOP --> CHECK_TASKS["Check scheduled tasks"]

    RECV_INPUT --> PROCESS["agent.run_async()"]
    POLL_EMAIL --> PROCESS
    LISTEN_VOICE --> PROCESS
    CHECK_TASKS --> PROCESS

    PROCESS --> LOCK["Acquire thread lock"]
    LOCK --> BUILD_PROMPT["Build system prompt"]
    BUILD_PROMPT --> CALL_CLAUDE["Call Claude API"]

    CALL_CLAUDE --> PARSE["Parse response"]
    PARSE --> HAS_TOOLS{Tool calls?}

    HAS_TOOLS -->|Yes| EXEC_TOOLS["Execute tools"]
    EXEC_TOOLS --> LOG_TOOLS["Log to memory"]
    LOG_TOOLS --> CALL_CLAUDE

    HAS_TOOLS -->|No| SEND_RESPONSE["Send response"]
    SEND_RESPONSE --> LOG_RESP["Log to memory"]
    LOG_RESP --> UNLOCK["Release lock"]
    UNLOCK --> CLI_LOOP
    UNLOCK --> EMAIL_LOOP
    UNLOCK --> VOICE_LOOP

    SERVER_START --> API_LOOP["FastAPI event loop"]
    API_LOOP --> RECV_API["Receive HTTP request"]
    RECV_API --> PROCESS
```

---

## Key Architectural Principles

1. **Everything is a Message**: User input, agent responses, tool execution, and background work all flow through the same message processing pipeline.

2. **Event-Driven Architecture**: The `EventEmitter` mixin enables loose coupling between components. Listeners subscribe to events without tight dependencies.

3. **Thread-Safe Concurrency**: Per-thread locks ensure serialized access to conversation threads, while `ThreadSafeList` protects shared data structures.

4. **Pluggable Channels**: Input listeners and output senders implement simple protocols, making it easy to add new communication channels.

5. **Graceful Degradation**: The memory system falls back to in-memory storage if SQLite is unavailable, ensuring the agent remains functional.

6. **Background Processing**: Objectives and scheduled tasks run independently in their own threads/tasks, not blocking the main conversation flow.

7. **Decorator-Based Tools**: The `@tool` decorator provides automatic JSON schema generation, dependency tracking, and registration.
