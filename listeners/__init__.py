"""
Listeners - Input channels for the agent.

Listeners receive messages from various sources and route them to the agent.
Each listener is a simple async function that:
1. Waits for input from its channel
2. Calls agent.run_async() with content and context
3. Handles the response appropriately

No base class needed - just follow the pattern.

Usage:
    from listeners.cli import run_cli_listener
    from listeners.email import run_email_listener

    await asyncio.gather(
        run_cli_listener(agent),
        run_email_listener(agent),
    )
"""

from listeners.cli import run_cli_listener
from listeners.email import run_email_listener
from listeners.voice import run_voice_listener

__all__ = ["run_cli_listener", "run_email_listener", "run_voice_listener"]
