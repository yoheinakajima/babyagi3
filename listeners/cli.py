"""
CLI Listener - Command-line interface input with styled output.

Simple REPL that reads from stdin and sends to the agent.
Subscribes to agent events for verbose output display.
"""

import asyncio
import logging
import sys

from utils.console import console

logger = logging.getLogger(__name__)


def setup_event_handlers(agent):
    """Subscribe to agent events for CLI display."""

    @agent.on("tool_start")
    def on_tool_start(event):
        console.tool_start(event["name"], event.get("input"))

    @agent.on("tool_end")
    def on_tool_end(event):
        console.tool_end(
            event["name"],
            event.get("result"),
            event.get("duration_ms")
        )

    @agent.on("objective_start")
    def on_objective_start(event):
        console.objective_start(event["id"], event["goal"])

    @agent.on("objective_end")
    def on_objective_end(event):
        console.objective_end(event["id"], event["status"])

    @agent.on("task_start")
    def on_task_start(event):
        console.task_start(event["id"], event["name"])

    @agent.on("task_end")
    def on_task_end(event):
        console.task_end(
            event["id"],
            event["status"],
            event.get("duration_ms")
        )


async def run_cli_listener(agent, config: dict = None, start_scheduler: bool = False):
    """Run the CLI listener.

    Args:
        agent: The Agent instance
        config: Optional configuration dict
        start_scheduler: Whether to start the scheduler after greeting
    """
    config = config or {}
    scheduler_task = None

    # Set up event handlers for verbose output
    setup_event_handlers(agent)

    # Show verbose mode hint
    from utils.console import VerboseLevel
    level = console.get_verbose()
    if level >= VerboseLevel.LIGHT:
        level_name = "light" if level == VerboseLevel.LIGHT else "deep"
        console.system(f'Verbose mode: {level_name} (say "turn off verbose" to hide logs)')

    # Get tool health and generate AI greeting
    try:
        from tools import get_health_summary
        health_summary = get_health_summary()
    except Exception as e:
        logger.debug("Could not get tool health summary: %s", e)
        health_summary = "Core tools ready."

    # Generate personalized greeting
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
        console.system(f"\n{greeting_text}\n")
    except Exception as e:
        logger.exception(f"Greeting generation failed: {e}")
        console.system("\nReady to assist. Type 'quit' to exit.\n")
    
    # Start scheduler AFTER greeting is displayed (so it doesn't block startup)
    if start_scheduler:
        scheduler_task = asyncio.create_task(agent.run_scheduler())

    # Main REPL loop
    while True:
        try:
            # Buffer verbose logs while waiting for input so they don't
            # push the prompt away from where the user types.
            console.begin_input()
            print(console.user_prompt(), end="", file=sys.stderr, flush=True)
            try:
                user_input = await asyncio.to_thread(input)
            except KeyboardInterrupt:
                console.end_input()
                print("", file=sys.stderr)
                continue
            console.end_input()
            console.flush_pending_logs()
            user_input = user_input.strip()

            # Check for verbose toggle commands
            if user_input.lower().startswith("/verbose"):
                _handle_verbose_command(user_input)
                continue

            if user_input.lower() in ("quit", "exit", "q"):
                console.system("Goodbye!")
                break

            if not user_input:
                continue

            # Process through agent with CLI context
            try:
                response = await agent.run_async(
                    user_input,
                    thread_id="main",
                    context={
                        "channel": "cli",
                        "is_owner": True,
                    }
                )
                console.agent(response)
            except Exception as e:
                logger.exception(f"Error processing message: {e}")
                console.system(f"Error: {e}")

        except EOFError:
            # Handle Ctrl+D
            console.end_input()
            console.system("\nGoodbye!")
            break
        except Exception as e:
            # Catch any unexpected errors in the REPL loop
            console.end_input()
            logger.exception(f"Unexpected error in REPL loop: {e}")
            console.system(f"Unexpected error: {e}")
    
    # Cleanup scheduler task if running
    if scheduler_task:
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass


def _handle_verbose_command(command: str):
    """Handle /verbose commands for runtime toggle."""
    parts = command.lower().split()

    if len(parts) == 1:
        # Just "/verbose" - show current level
        level = console.get_verbose()
        console.system(f"Verbose level: {level.name.lower()} ({level.value})")
        console.system("Usage: /verbose [off|light|deep]")
        return

    level_str = parts[1]
    if level_str in ("off", "0"):
        console.set_verbose("off")
        console.system("Verbose output: off")
    elif level_str in ("light", "1", "on"):
        console.set_verbose("light")
        console.success("Verbose output: light (key operations)")
    elif level_str in ("deep", "2", "full", "all"):
        console.set_verbose("deep")
        console.success("Verbose output: deep (everything)")
    else:
        console.warning(f"Unknown verbose level: {level_str}")
        console.system("Usage: /verbose [off|light|deep]")
