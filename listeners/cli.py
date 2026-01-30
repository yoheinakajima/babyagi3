"""
CLI Listener - Command-line interface input.

Simple REPL that reads from stdin and sends to the agent.
"""

import asyncio


async def run_cli_listener(agent, config: dict = None):
    """Run the CLI listener.

    Args:
        agent: The Agent instance
        config: Optional configuration dict
    """
    config = config or {}

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
        print(f"\n{greeting_text}\n")
    except Exception:
        print("\nReady to assist. Type 'quit' to exit.\n")

    # Main REPL loop
    while True:
        try:
            user_input = await asyncio.to_thread(input, "You: ")
            user_input = user_input.strip()

            if user_input.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
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

            print(f"Assistant: {response}\n")

        except EOFError:
            # Handle Ctrl+D
            print("\nGoodbye!")
            break
