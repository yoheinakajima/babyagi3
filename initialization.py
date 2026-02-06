"""
LLM-Driven Interactive Initialization for BabyAGI.

Self-contained onboarding module where an LLM conversationally guides the user
through first-time setup. The user can ask questions, provide info in any order,
skip things, and get explanations about how BabyAGI works.

The LLM has a rich system prompt with full context about BabyAGI's architecture,
channels, memory, and tools. It collects setup info through natural conversation
and calls a `complete_initialization` tool when it has enough to proceed.

Design principles:
    - All init context (system prompt, tool schema, conversation) is discarded
      after setup. None of this persists into the main agent's context window.
    - Uses the same LLM client infrastructure but runs a standalone conversation
      loop separate from the Agent class (which doesn't exist yet at init time).
    - The `complete_initialization` tool captures structured data from the
      conversation and applies it to config/env in one shot.

Flow:
    1. Detect first run (no ~/.babyagi/initialized marker)
    2. Create lightweight LLM client (no Agent, no memory, no tools)
    3. Run conversation loop with rich system prompt + one tool
    4. LLM guides user, answers questions, collects info naturally
    5. LLM calls complete_initialization tool with structured data
    6. Apply config, set env vars, validate services, write marker
    7. Schedule daily tasks after agent starts (separate step)
    8. Discard all conversation context
"""

import os
import re
import sys
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# Marker file location
INIT_MARKER = Path(os.path.expanduser("~/.babyagi/initialized"))
INIT_STATE_FILE = Path(os.path.expanduser("~/.babyagi/init_state.json"))

# ANSI colors for terminal output
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_RESET = "\033[0m"


# =============================================================================
# Public API
# =============================================================================

def needs_initialization() -> bool:
    """Check if BabyAGI needs first-time initialization."""
    return not INIT_MARKER.exists()


def run_initialization(config: dict) -> dict:
    """Run the LLM-driven interactive initialization.

    Creates a lightweight LLM client and runs a conversation loop where
    the LLM guides the user through setup. The LLM has full context about
    BabyAGI and can answer questions. When ready, it calls the
    complete_initialization tool with structured data.

    All conversation context is discarded after this returns.

    Args:
        config: The loaded config dict (will be modified in-place).

    Returns:
        Updated config dict with user-provided values applied.
    """
    _print_welcome_banner()

    # Detect what's already configured
    existing = _detect_existing_config(config)

    # Build the system prompt with full BabyAGI context
    system_prompt = _build_system_prompt(existing)

    # Create a lightweight LLM client (synchronous, no Agent needed)
    try:
        client, model = _create_init_client(config)
    except Exception as e:
        logger.warning("Could not create LLM client for interactive init: %s", e)
        print(f"\n  {_YELLOW}Could not start interactive setup: {e}{_RESET}")
        print(f"  {_DIM}Please configure manually via config.yaml and environment variables.{_RESET}\n")
        _write_marker({"name": "", "email": ""})
        return config

    # Run the conversation loop
    result = _run_init_conversation(client, model, system_prompt, config)

    if result:
        _apply_init_result(config, result)
        _save_init_state(result)
        _write_marker(result)
        _print_completion()
    else:
        # User quit or conversation failed - still mark as initialized
        # so we don't block startup repeatedly
        print(f"\n  {_DIM}Setup incomplete. You can reconfigure by deleting ~/.babyagi/initialized{_RESET}\n")
        _write_marker({"name": "", "email": ""})

    return config


def schedule_post_init_tasks(agent):
    """Schedule the two daily tasks after agent is fully initialized.

    Called from main.py after the agent is created. Reads saved init state
    and sets up:
      1. Daily stats analysis email (5 min after init, then every 24h)
      2. Daily self-improvement task (1h after init, then every 24h)

    After scheduling, deletes the init state file so this is truly one-time.
    """
    if not INIT_STATE_FILE.exists():
        return

    try:
        with open(INIT_STATE_FILE) as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return

    owner_email = state.get("owner_email", "")
    if not owner_email:
        logger.info("No owner email found, skipping daily stats task")

    from scheduler import Schedule, ScheduledTask

    now = datetime.now(timezone.utc)

    # --- Task 1: Daily Stats Analysis ---
    if owner_email:
        stats_anchor = (now + timedelta(minutes=5)).isoformat()
        stats_task = ScheduledTask(
            id=uuid.uuid4().hex[:8],
            name="Daily Stats Report",
            goal=_DAILY_STATS_GOAL.format(owner_email=owner_email),
            schedule=Schedule(
                kind="every",
                every="24h",
                anchor=stats_anchor,
            ),
        )
        agent.scheduler.add(stats_task)
        logger.info("Scheduled daily stats report (first run in ~5 minutes)")

    # --- Task 2: Daily Self-Improvement ---
    improvement_anchor = (now + timedelta(hours=1)).isoformat()
    improvement_task = ScheduledTask(
        id=uuid.uuid4().hex[:8],
        name="Daily Self-Improvement",
        goal=_DAILY_IMPROVEMENT_GOAL,
        schedule=Schedule(
            kind="every",
            every="24h",
            anchor=improvement_anchor,
        ),
    )
    agent.scheduler.add(improvement_task)
    logger.info("Scheduled daily self-improvement (first run in ~1 hour)")

    # Clean up init state file - no longer needed
    try:
        INIT_STATE_FILE.unlink()
    except OSError:
        pass


# =============================================================================
# Goal Templates for Scheduled Tasks
# =============================================================================

_DAILY_STATS_GOAL = """Compile and email a detailed daily statistics report to {owner_email}.

Gather the following data:
1. TOOL USAGE: Use get_cost_summary to get overall metrics. List every tool used in the last 24 hours with call counts, success rates, and average latency.
2. MEMORY & EXTRACTION: Check memory system stats - how many entities extracted, relationships found, summaries generated, and events recorded in the last 24 hours.
3. LLM COSTS: Break down costs by model and by source (agent, extraction, summary, retrieval). Show total tokens and total spend.
4. SCHEDULED TASKS: List all scheduled tasks with their last run status and next run time.
5. ERRORS: List any errors from the last 24 hours with tool name and error message.

Format this as a clean, readable email with sections and send it using send_email to {owner_email} with subject "BabyAGI Daily Stats Report - [today's date]".

Keep the email concise but thorough. Use plain text formatting with clear section headers."""

_DAILY_IMPROVEMENT_GOAL = """Think of ONE concrete thing to do right now to be more helpful to your owner.

IMPORTANT: First, check your memory for notes tagged "self_improvement_log" to see what you've done in previous runs. Do NOT repeat recent actions - vary your approach.

Choose ONE of these action types (rotate between them):

A) CREATE A USEFUL SKILL: Think about what your owner frequently asks for or might need. Create and save a new tool/skill that would help them. Test it to make sure it works.

B) SET UP A HELPFUL SCHEDULED TASK: Think of something useful to monitor or do regularly that isn't already scheduled. Examples: check for interesting news in owner's field, review and organize recent memories, check if any API keys are expiring, test that all channels are working.

C) ASK YOUR OWNER A QUESTION: Come up with one thoughtful question to better understand your owner's needs, preferences, or workflows. Send it via the most appropriate channel (email or SMS). Make it specific and actionable, not generic.

After completing your chosen action, store a note in memory with tag "self_improvement_log" recording:
- Date
- Action type (A/B/C)
- What you did
- Brief rationale

This helps you track what you've done and vary your approach. Keep the log entry under 100 words to prevent context bloat."""


# =============================================================================
# Tool Schema for LLM-driven initialization
# =============================================================================

_COMPLETE_INIT_TOOL = {
    "name": "complete_initialization",
    "description": (
        "Call this when you have gathered enough information from the user to "
        "complete BabyAGI setup. Pass all collected values. Fields left empty "
        "or omitted will use defaults or be skipped. You MUST collect at least "
        "the owner's name and email before calling this."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "owner_name": {
                "type": "string",
                "description": "The owner's name",
            },
            "owner_email": {
                "type": "string",
                "description": "The owner's email address (required for daily stats reports)",
            },
            "owner_bio": {
                "type": "string",
                "description": (
                    "A brief description of who the owner is - their role, profession, "
                    "interests, or background. E.g. 'VC focused on AI startups' or "
                    "'software engineer working on developer tools'. 1-3 sentences."
                ),
            },
            "owner_goal": {
                "type": "string",
                "description": (
                    "How the owner hopes this AI assistant will help them - what kinds "
                    "of tasks, what areas of their life/work, what they'd find most valuable. "
                    "E.g. 'Help me stay on top of AI news, manage deal flow, and draft investor updates'. "
                    "1-3 sentences."
                ),
            },
            "owner_phone": {
                "type": "string",
                "description": "Owner's phone number in E.164 format (e.g. +15551234567). Needed for SMS channel.",
            },
            "owner_timezone": {
                "type": "string",
                "description": "IANA timezone (e.g. America/New_York, US/Pacific, Europe/London, UTC)",
            },
            "agent_name": {
                "type": "string",
                "description": "What the agent should be called (default: Assistant)",
            },
            "agentmail_api_key": {
                "type": "string",
                "description": "AgentMail API key for email channel. Get one at https://agentmail.to",
            },
            "sendblue_api_key": {
                "type": "string",
                "description": "SendBlue API key for SMS/iMessage channel. Get one at https://sendblue.co",
            },
            "sendblue_api_secret": {
                "type": "string",
                "description": "SendBlue API secret (required along with API key for SMS)",
            },
        },
        "required": ["owner_name", "owner_email"],
    },
}


# =============================================================================
# System Prompt for Initialization LLM
# =============================================================================

def _build_system_prompt(existing: dict) -> str:
    """Build the system prompt with full BabyAGI context.

    This prompt gives the LLM everything it needs to guide setup and
    answer questions. It's only used during init and then discarded.
    """

    # Build a section about what's already configured
    existing_lines = []
    if existing.get("owner_name"):
        existing_lines.append(f"- Owner name: {existing['owner_name']}")
    if existing.get("owner_email"):
        existing_lines.append(f"- Owner email: {existing['owner_email']}")
    if existing.get("owner_bio"):
        existing_lines.append(f"- Owner bio: {existing['owner_bio']}")
    if existing.get("owner_goal"):
        existing_lines.append(f"- Owner goal for agent: {existing['owner_goal']}")
    if existing.get("owner_phone"):
        existing_lines.append(f"- Owner phone: {existing['owner_phone']}")
    if existing.get("owner_timezone"):
        existing_lines.append(f"- Timezone: {existing['owner_timezone']}")
    if existing.get("agent_name") and existing["agent_name"] != "Assistant":
        existing_lines.append(f"- Agent name: {existing['agent_name']}")
    if existing.get("agentmail_configured"):
        existing_lines.append("- AgentMail: already configured (API key detected)")
    if existing.get("sendblue_configured"):
        existing_lines.append("- SendBlue: already configured (API key + secret detected)")

    existing_section = ""
    if existing_lines:
        existing_section = f"""
ALREADY CONFIGURED (detected from environment):
{chr(10).join(existing_lines)}

Confirm these with the user or let them change them. Don't re-ask for things already set unless the user wants to change them.
"""

    return f"""You are the BabyAGI setup assistant. You're helping a new user configure their personal AI agent for the first time. Be friendly, concise, and helpful.

ABOUT BABYAGI:
BabyAGI is a persistent AI agent that runs in the background and communicates through multiple channels. Here's what it does:

1. MEMORY SYSTEM: Three-layer persistent memory:
   - Event Log: Every interaction stored with timestamps (ground truth)
   - Knowledge Graph: Background LLM extracts entities (people, companies, concepts) and relationships between them
   - Hierarchical Summaries: Pre-computed summaries organized in a tree, refreshed as new events accumulate
   - Self-Improvement: Learns from owner feedback, builds user_preferences over time

2. COMMUNICATION CHANNELS:
   - CLI: Terminal chat (always available)
   - Email (AgentMail): Agent gets its own email address (@agentmail.to). Can send/receive emails, handle verifications, manage subscriptions. Requires AGENTMAIL_API_KEY from https://agentmail.to (free tier available).
   - SMS/iMessage (SendBlue): Text the agent from your phone. Requires SENDBLUE_API_KEY and SENDBLUE_API_SECRET from https://sendblue.co. Also needs the owner's phone number to know which texts are from the owner.
   - Voice: Speech input/output (optional, requires extra packages)

3. CAPABILITIES:
   - Background research with priority queuing and budget caps
   - Scheduled tasks (one-time, recurring intervals, cron expressions)
   - Web search and browsing
   - Code execution in sandboxed environments
   - Dynamic tool creation (create new tools through conversation)
   - Skill learning from SKILL.md files
   - 250+ app integrations via Composio

4. CHANNELS WE RECOMMEND SETTING UP:
   - AgentMail (email) - strongly recommended. Enables daily stats reports, email communication, and account management.
   - SendBlue (SMS/iMessage) - recommended if user has an iPhone or wants SMS access. Makes it easy to reach the agent from anywhere.

WHAT YOU NEED TO COLLECT:
Required:
  - Owner's name (for personalization)
  - Owner's email (for daily stats reports and as owner identifier)

Strongly Recommended (these help the agent be genuinely useful from day one):
  - A brief bio of the owner: who they are, what they do, their role/profession, interests.
    This helps the agent understand context and tailor its behavior. Even one sentence helps.
    Examples: "I'm a VC focused on AI startups", "I'm a freelance designer juggling multiple clients"
  - How the owner wants the agent to help: what tasks, what areas of life/work, what would be most valuable.
    This directly shapes the agent's daily self-improvement - it will proactively build skills and
    set up tasks aligned with these goals.
    Examples: "Help me track AI news and manage deal flow", "Help me stay organized and handle client emails"

Recommended:
  - AgentMail API key (for email channel - get at https://agentmail.to)
  - SendBlue API key + secret (for SMS - get at https://sendblue.co)
  - Owner's phone number (needed for SMS to identify owner messages)

Optional:
  - Owner's timezone (for scheduling)
  - Agent name (defaults to "Assistant")
{existing_section}
YOUR BEHAVIOR:
- Start by briefly introducing yourself and explaining you'll help set up BabyAGI
- Be conversational. Let the user provide info naturally. Don't interrogate.
- If the user gives multiple pieces of info at once, acknowledge all of them
- Early in the conversation, ask the user to tell you a bit about themselves and how they hope this agent will help.
  Frame it naturally: "Tell me a bit about yourself and what you'd like me to help you with."
  This is important - it shapes how the agent improves itself over time. But don't force it if they want to skip.
- Answer questions about BabyAGI, channels, memory, or anything else knowledgeably
- Proactively explain what AgentMail and SendBlue are when asking for their keys
- If the user doesn't have an API key, explain where to get one and that they can skip it for now
- When you have at least the owner's name and email, you can call complete_initialization
- Before calling the tool, briefly confirm what you have and ask if they want to add anything else
- Be efficient - don't over-explain. A few sentences per response is ideal.
- If the user says "skip" or wants to finish quickly, respect that and call the tool with what you have
- NEVER make up or guess API keys, emails, or phone numbers

AFTER INITIALIZATION:
Two daily tasks will be automatically scheduled:
1. Daily Stats Report - detailed usage analytics emailed to the owner
2. Daily Self-Improvement - the agent independently finds ways to be more helpful

The user can always reconfigure later by editing config.yaml or deleting ~/.babyagi/initialized."""


# =============================================================================
# LLM Client Creation
# =============================================================================

def _create_init_client(config: dict):
    """Create a lightweight synchronous LLM client for initialization.

    Returns (client, model_id) tuple. Uses the same LLM infrastructure
    as the main agent but synchronous (no async needed during init).
    """
    from llm_config import get_available_provider, get_default_models_for_provider, init_llm_config

    # Initialize LLM config from the loaded config
    init_llm_config(config)

    provider = get_available_provider()
    if provider == "none":
        raise RuntimeError("No LLM provider configured")

    defaults = get_default_models_for_provider(provider)
    model = defaults["agent"]

    # Use the LiteLLM adapter (synchronous version) - supports both providers
    from metrics.clients import LiteLLMAnthropicAdapter
    client = LiteLLMAnthropicAdapter()

    return client, model


# =============================================================================
# Conversation Loop
# =============================================================================

def _run_init_conversation(client, model: str, system_prompt: str, config: dict) -> dict | None:
    """Run the initialization conversation loop.

    The LLM converses with the user and eventually calls
    complete_initialization with structured data.

    Returns:
        Dict with initialization data, or None if user quit.
    """
    messages = []
    tool = _COMPLETE_INIT_TOOL
    max_turns = 30  # Safety limit

    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
                tools=[tool],
            )
        except Exception as e:
            logger.error("LLM call failed during init: %s", e)
            print(f"\n  {_YELLOW}Error communicating with LLM: {e}{_RESET}")
            return None

        # Process response content
        text_parts = []
        tool_call = None

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_call = block

        # Display any text from the LLM
        if text_parts:
            text = "\n".join(text_parts)
            print(f"\n{_CYAN}{text}{_RESET}")

        # If the LLM called the tool, we're done collecting
        if tool_call and tool_call.name == "complete_initialization":
            result = tool_call.input
            # Validate minimum requirements
            if not result.get("owner_name") or not result.get("owner_email"):
                # Tell the LLM it needs more info
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": json.dumps({
                            "error": "owner_name and owner_email are required. Please collect these from the user first."
                        }),
                    }],
                })
                continue

            # Validate and attempt AgentMail connection
            agentmail_result = _try_agentmail_connection(result.get("agentmail_api_key"))

            # Build tool result to show the LLM what happened
            tool_result_data = {
                "status": "success",
                "message": "Initialization complete!",
                "channels_configured": ["cli"],
            }
            if agentmail_result.get("inbox_id"):
                tool_result_data["channels_configured"].append(f"email ({agentmail_result['inbox_id']})")
                result["agentmail_inbox_id"] = agentmail_result["inbox_id"]
            elif agentmail_result.get("error"):
                tool_result_data["agentmail_note"] = agentmail_result["error"]

            if result.get("sendblue_api_key") and result.get("sendblue_api_secret"):
                tool_result_data["channels_configured"].append("sendblue (SMS/iMessage)")

            # Append tool use + result to conversation so LLM can give final message
            messages.append({
                "role": "assistant",
                "content": response.content,
            })
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": json.dumps(tool_result_data),
                }],
            })

            # One more LLM turn for a farewell message
            try:
                farewell_response = client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=system_prompt,
                    messages=messages,
                )
                for block in farewell_response.content:
                    if block.type == "text":
                        print(f"\n{_CYAN}{block.text}{_RESET}")
            except Exception:
                pass  # Farewell is non-critical

            return result

        # Not a tool call - get user input for next turn
        # Append assistant message
        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        # Get user input
        try:
            print()
            user_input = input(f"  {_BLUE}You: {_RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if user_input.lower() in ("quit", "exit", "q"):
            return None

        if not user_input:
            user_input = "continue"

        messages.append({
            "role": "user",
            "content": user_input,
        })

    # Exceeded max turns
    print(f"\n  {_YELLOW}Setup conversation exceeded maximum turns.{_RESET}")
    return None


# =============================================================================
# Service Validation
# =============================================================================

def _try_agentmail_connection(api_key: str | None) -> dict:
    """Try to connect to AgentMail and get/create an inbox.

    Returns dict with inbox_id on success or error message on failure.
    """
    if not api_key:
        return {}

    try:
        from agentmail import AgentMail
        client = AgentMail(api_key=api_key)

        inboxes_response = client.inboxes.list()
        inboxes = getattr(inboxes_response, 'inboxes', None) or inboxes_response or []

        if inboxes:
            inbox = inboxes[0]
            inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
            return {"inbox_id": inbox_id, "status": "existing"}

        inbox = client.inboxes.create()
        inbox_id = getattr(inbox, 'inbox_id', None) or getattr(inbox, 'id', None)
        return {"inbox_id": inbox_id, "status": "created"}

    except ImportError:
        return {"error": "agentmail package not installed (pip install agentmail). Key saved for later."}
    except Exception as e:
        return {"error": f"Could not connect to AgentMail: {e}. Key saved for later."}


# =============================================================================
# Apply Results
# =============================================================================

def _detect_existing_config(config: dict) -> dict:
    """Detect what's already configured from env vars and config."""
    owner = config.get("owner", {})
    return {
        "owner_name": owner.get("name") or os.environ.get("OWNER_NAME", ""),
        "owner_email": owner.get("email") or os.environ.get("OWNER_EMAIL", ""),
        "owner_bio": owner.get("bio") or os.environ.get("OWNER_BIO", ""),
        "owner_goal": owner.get("goal") or os.environ.get("OWNER_GOAL", ""),
        "owner_phone": owner.get("phone") or os.environ.get("OWNER_PHONE", ""),
        "owner_timezone": owner.get("timezone") or os.environ.get("OWNER_TIMEZONE", ""),
        "agent_name": config.get("agent", {}).get("name") or os.environ.get("AGENT_NAME", ""),
        "agentmail_configured": bool(os.environ.get("AGENTMAIL_API_KEY")),
        "sendblue_configured": bool(
            os.environ.get("SENDBLUE_API_KEY") and os.environ.get("SENDBLUE_API_SECRET")
        ),
    }


def _apply_init_result(config: dict, result: dict):
    """Apply the initialization result to config and environment."""
    # Owner config
    if "owner" not in config:
        config["owner"] = {}
    config["owner"]["name"] = result.get("owner_name", "")
    config["owner"]["email"] = result.get("owner_email", "")
    config["owner"]["bio"] = result.get("owner_bio", "")
    config["owner"]["goal"] = result.get("owner_goal", "")
    config["owner"]["phone"] = result.get("owner_phone", "")
    config["owner"]["timezone"] = result.get("owner_timezone", "")
    config["owner"]["contacts"] = {
        "email": result.get("owner_email", ""),
        "phone": result.get("owner_phone", ""),
    }

    # Agent config
    if "agent" not in config:
        config["agent"] = {}
    config["agent"]["name"] = result.get("agent_name", "Assistant")

    # Set environment variables for immediate use by the agent
    if result.get("owner_name"):
        os.environ["OWNER_NAME"] = result["owner_name"]
    if result.get("owner_email"):
        os.environ["OWNER_EMAIL"] = result["owner_email"]
    if result.get("owner_bio"):
        os.environ["OWNER_BIO"] = result["owner_bio"]
    if result.get("owner_goal"):
        os.environ["OWNER_GOAL"] = result["owner_goal"]
    if result.get("owner_phone"):
        os.environ["OWNER_PHONE"] = result["owner_phone"]
    if result.get("owner_timezone"):
        os.environ["OWNER_TIMEZONE"] = result["owner_timezone"]
    if result.get("agent_name"):
        os.environ["AGENT_NAME"] = result["agent_name"]

    # AgentMail
    if result.get("agentmail_api_key"):
        os.environ["AGENTMAIL_API_KEY"] = result["agentmail_api_key"]
        if result.get("agentmail_inbox_id"):
            os.environ["AGENTMAIL_INBOX_ID"] = result["agentmail_inbox_id"]

        if "channels" not in config:
            config["channels"] = {}
        if "email" not in config["channels"]:
            config["channels"]["email"] = {}
        config["channels"]["email"]["enabled"] = True

    # SendBlue
    if result.get("sendblue_api_key") and result.get("sendblue_api_secret"):
        os.environ["SENDBLUE_API_KEY"] = result["sendblue_api_key"]
        os.environ["SENDBLUE_API_SECRET"] = result["sendblue_api_secret"]

        if "channels" not in config:
            config["channels"] = {}
        if "sendblue" not in config["channels"]:
            config["channels"]["sendblue"] = {}
        config["channels"]["sendblue"]["enabled"] = True


# =============================================================================
# Persistence
# =============================================================================

def _save_init_state(result: dict):
    """Save minimal state for post-agent task scheduling.

    Read once by schedule_post_init_tasks() then deleted.
    """
    state = {
        "owner_email": result.get("owner_email", ""),
        "owner_name": result.get("owner_name", ""),
        "owner_bio": result.get("owner_bio", ""),
        "owner_goal": result.get("owner_goal", ""),
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "agentmail_configured": bool(result.get("agentmail_api_key")),
        "sendblue_configured": bool(
            result.get("sendblue_api_key") and result.get("sendblue_api_secret")
        ),
    }

    INIT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(INIT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def _write_marker(result: dict):
    """Write the initialization marker file."""
    INIT_MARKER.parent.mkdir(parents=True, exist_ok=True)
    marker_data = {
        "initialized_at": datetime.now(timezone.utc).isoformat(),
        "owner": result.get("owner_name", result.get("name", "")),
        "version": "0.3.0",
    }
    with open(INIT_MARKER, "w") as f:
        json.dump(marker_data, f, indent=2)


# =============================================================================
# UI Helpers
# =============================================================================

def _print_welcome_banner():
    """Print a brief welcome banner. The LLM handles the rest."""
    print(f"""
{_BOLD}{_CYAN}{'=' * 60}
   BabyAGI - First-Time Setup
{'=' * 60}{_RESET}
""")


def _print_completion():
    """Print post-initialization info."""
    print(f"""
{_BOLD}{_GREEN}{'=' * 60}
   Setup Complete!
{'=' * 60}{_RESET}

{_DIM}Two daily tasks have been scheduled:
  1. Daily Stats Report (first run in ~5 min, then every 24h)
  2. Daily Self-Improvement (first run in ~1h, then every 24h)

Reconfigure anytime: delete ~/.babyagi/initialized and restart.{_RESET}

{_BOLD}Starting your agent...{_RESET}
""")
