"""
BabyAGI - Multi-Channel AI Assistant

Entry point for the agent with support for multiple input/output channels.

Usage:
    python main.py              # Run with CLI (default)
    python main.py serve        # Run API server
    python main.py serve 8080   # Run on custom port
    python main.py channels     # Run all enabled channels

Channels:
    - CLI: Command-line interface (always enabled)
    - Email: Receive and respond to emails via AgentMail
    - Voice: Speech input/output (requires additional packages)

Configuration:
    Set options in config.yaml or via environment variables.
    See config.yaml for all available options.
"""

import asyncio
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "serve":
            # API server mode
            import uvicorn
            from server import app
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            uvicorn.run(app, host="0.0.0.0", port=port)

        elif command == "channels":
            # Multi-channel mode
            asyncio.run(run_all_channels())

        elif command == "cli":
            # Explicit CLI mode
            asyncio.run(run_cli_only())

        else:
            print(f"Unknown command: {command}")
            print("Usage: python main.py [serve|channels|cli]")
            sys.exit(1)
    else:
        # Default: CLI only (original behavior)
        asyncio.run(run_cli_only())


async def run_cli_only():
    """Run CLI listener only (original behavior)."""
    from agent import Agent
    from listeners.cli import run_cli_listener

    print("BabyAGI v0.3.0")
    print("=" * 40)

    # Load config
    from config import load_config
    config = load_config()

    # Initialize agent
    agent = Agent(config=config)

    # Start scheduler
    scheduler_task = asyncio.create_task(agent.run_scheduler())

    try:
        await run_cli_listener(agent, config.get("channels", {}).get("cli", {}))
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        scheduler_task.cancel()


async def run_all_channels():
    """Run all enabled channels concurrently."""
    from agent import Agent
    from config import load_config, is_channel_enabled, get_channel_config

    print("BabyAGI v0.3.0 - Multi-Channel Mode")
    print("=" * 40)

    # Load config
    config = load_config()

    # Initialize agent with config
    agent = Agent(config=config)

    # Register senders for enabled channels
    _register_senders(agent, config)

    # Build list of tasks
    tasks = []

    # Always run scheduler
    tasks.append(agent.run_scheduler())

    # CLI (always enabled, but check config)
    if is_channel_enabled(config, "cli"):
        from listeners.cli import run_cli_listener
        cli_config = get_channel_config(config, "cli")
        tasks.append(run_cli_listener(agent, cli_config))
        logger.info("CLI listener enabled")

    # Email
    if is_channel_enabled(config, "email"):
        # Check if AgentMail is configured
        if os.environ.get("AGENTMAIL_API_KEY"):
            from listeners.email import run_email_listener
            email_config = get_channel_config(config, "email")
            email_config["owner_email"] = config.get("owner", {}).get("email")
            tasks.append(run_email_listener(agent, email_config))
            logger.info("Email listener enabled")
        else:
            logger.warning("Email enabled but AGENTMAIL_API_KEY not set")

    # Voice
    if is_channel_enabled(config, "voice"):
        from listeners.voice import run_voice_listener
        voice_config = get_channel_config(config, "voice")
        tasks.append(run_voice_listener(agent, voice_config))
        logger.info("Voice listener enabled")

    # Log active channels
    active_channels = [name for name in ["cli", "email", "voice"]
                      if is_channel_enabled(config, name)]
    print(f"\nActive channels: {', '.join(active_channels)}")
    print("Press Ctrl+C to stop\n")

    # Run all tasks
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutting down...")


def _register_senders(agent, config):
    """Register senders for enabled channels."""
    from config import is_channel_enabled, get_channel_config

    # CLI sender (always available)
    from senders.cli import CLISender
    agent.register_sender("cli", CLISender())

    # Email sender
    if is_channel_enabled(config, "email"):
        if os.environ.get("AGENTMAIL_API_KEY"):
            from senders.email import EmailSender
            email_config = get_channel_config(config, "email")
            agent.register_sender("email", EmailSender(email_config))
            logger.info("Email sender registered")

    # Add more senders here as they're implemented
    # if is_channel_enabled(config, "sms"):
    #     from senders.sms import SMSSender
    #     agent.register_sender("sms", SMSSender(get_channel_config(config, "sms")))


if __name__ == "__main__":
    main()
