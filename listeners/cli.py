"""
CLI Listener - Command-line interface input with styled output.

Simple REPL that reads from stdin and sends to the agent.
Subscribes to agent events for verbose output display.
"""

import asyncio

from utils.console import console


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


async def run_cli_listener(agent, config: dict = None):
    """Run the CLI listener.

    Args:
        agent: The Agent instance
        config: Optional configuration dict
    """
    config = config or {}

    # Set up event handlers for verbose output
    setup_event_handlers(agent)

    # Get tool health and generate AI greeting
    try:
        from tools import get_health_summary
        health_summary = get_health_summary()
    except Exception:
        health_summary = "Core tools ready."

    # Generate personalized greeting
    greeting_prompt = f"""Generate a brief, friendly greeting (2-3 sentences max).

Tool Status:
{health_summary}

Be concise. Mention what you can help with based on available tools. If any tools need setup, briefly note it. End with an invitation to chat."""

    try:
        greeting = await asyncio.to_thread(
            agent.client.messages.create,
            model=agent.model,
            max_tokens=200,
            messages=[{"role": "user", "content": greeting_prompt}]
        )
        greeting_text = "".join(b.text for b in greeting.content if hasattr(b, "text"))
        console.system(f"\n{greeting_text}\n")
    except Exception:
        console.system("\nReady to assist. Type 'quit' to exit.\n")

    # Main REPL loop
    while True:
        try:
            # Print prompt separately with flush, then read input
            # This ensures the prompt appears even in buffered environments
            print(console.user_prompt(), end="", flush=True)
            user_input = await asyncio.to_thread(input)
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
            response = await agent.run_async(
                user_input,
                thread_id="main",
                context={
                    "channel": "cli",
                    "is_owner": True,
                }
            )

            console.agent(response)

        except EOFError:
            # Handle Ctrl+D
            console.system("\nGoodbye!")
            break


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
