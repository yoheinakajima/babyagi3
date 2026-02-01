"""
BabyAGI - Multi-Channel AI Assistant

Entry point for the agent with support for multiple input/output channels.

Usage:
    python main.py              # Run with CLI (default)
    python main.py serve        # Run API server only
    python main.py serve 8080   # Run on custom port
    python main.py channels     # Run listeners only (no webhook server)
    python main.py all          # Run EVERYTHING: CLI + Email + SMS webhooks
    python main.py all 8080     # Run all on custom port

Channels:
    - CLI: Command-line interface (always enabled)
    - Email: Receive and respond to emails via AgentMail
    - Voice: Speech input/output (requires additional packages)
    - SendBlue: SMS/iMessage via SendBlue webhooks (requires 'all' or 'serve' mode)

Configuration:
    Set options in config.yaml or via environment variables.
    See config.yaml for all available options.

Verbose Output:
    Control with BABYAGI_VERBOSE environment variable or config.yaml:
    - 0/off: No verbose output (default)
    - 1/light: Key operations (tool names, task starts)
    - 2/deep: Everything (inputs, outputs, full details)

    Runtime toggle: /verbose [off|light|deep]
"""

import asyncio
import logging
import os
import sys

from utils.console import console

# Configure logging with immediate stderr output
handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

# Suppress noisy library loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)
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
            # Multi-channel mode (listeners only, no webhook server)
            asyncio.run(run_all_channels())

        elif command == "all":
            # Combined mode: API server + all channel listeners
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
            asyncio.run(run_all_with_server(port))

        elif command == "cli":
            # Explicit CLI mode
            asyncio.run(run_cli_only())

        else:
            console.error(f"Unknown command: {command}")
            console.system("Usage: python main.py [serve|channels|cli|all]")
            sys.exit(1)
    else:
        # Default: CLI only (original behavior)
        asyncio.run(run_cli_only())


async def run_cli_only():
    """Run CLI listener only (original behavior)."""
    from agent import Agent
    from listeners.cli import run_cli_listener

    console.banner("BabyAGI v0.3.0")

    # Load config
    from config import load_config
    config = load_config()

    # Configure verbose level from config (env var takes precedence)
    if not os.environ.get("BABYAGI_VERBOSE"):
        verbose_config = config.get("verbose", "off")
        console.set_verbose(verbose_config)

    # Initialize agent (memory status is printed during initialization)
    agent = Agent(config=config)

    # Start optional memory background tasks
    background_tasks = []
    if agent.memory is not None:
        memory_config = config.get("memory", {})
        if memory_config.get("background_extraction", True):
            from memory.integration import create_extraction_background_task
            extraction_loop = create_extraction_background_task(
                agent.memory,
                interval_seconds=memory_config.get("extraction_interval", 60)
            )
            background_tasks.append(asyncio.create_task(extraction_loop()))

    # Start CLI listener first (includes greeting), then start scheduler
    # Scheduler is started inside the listener after greeting displays
    try:
        await run_cli_listener(agent, config.get("channels", {}).get("cli", {}), start_scheduler=True)
    except KeyboardInterrupt:
        console.system("\nGoodbye!")
    finally:
        # Clean up background tasks
        for task in background_tasks:
            task.cancel()
        if agent.memory is not None:
            agent.memory.store.close()


async def run_all_channels():
    """Run all enabled channels concurrently."""
    from agent import Agent
    from config import load_config, is_channel_enabled, get_channel_config

    console.banner("BabyAGI v0.3.0 - Multi-Channel Mode")

    # Load config
    config = load_config()

    # Configure verbose level from config (env var takes precedence)
    if not os.environ.get("BABYAGI_VERBOSE"):
        verbose_config = config.get("verbose", "off")
        console.set_verbose(verbose_config)

    # Initialize agent with config (memory status is printed during initialization)
    agent = Agent(config=config)

    # Register senders for enabled channels
    _register_senders(agent, config)

    # Build list of tasks
    tasks = []

    # Always run scheduler
    tasks.append(agent.run_scheduler())

    # Memory background extraction task (if SQLite memory is active)
    if agent.memory is not None:
        memory_config = config.get("memory", {})
        if memory_config.get("background_extraction", True):
            from memory.integration import create_extraction_background_task
            extraction_loop = create_extraction_background_task(
                agent.memory,
                interval_seconds=memory_config.get("extraction_interval", 60)
            )
            tasks.append(extraction_loop())

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

    # SendBlue (SMS/iMessage)
    if is_channel_enabled(config, "sendblue"):
        # Check if SendBlue is configured
        if os.environ.get("SENDBLUE_API_KEY") and os.environ.get("SENDBLUE_API_SECRET"):
            from listeners.sendblue import run_sendblue_listener
            sendblue_config = get_channel_config(config, "sendblue")
            sendblue_config["owner_phone"] = config.get("owner", {}).get("phone")
            tasks.append(run_sendblue_listener(agent, sendblue_config))
            logger.info("SendBlue listener enabled")
        else:
            logger.warning("SendBlue enabled but SENDBLUE_API_KEY or SENDBLUE_API_SECRET not set")

    # Log active channels
    active_channels = [name for name in ["cli", "email", "voice", "sendblue"]
                      if is_channel_enabled(config, name)]
    console.system(f"\nActive channels: {', '.join(active_channels)}")
    console.system("Press Ctrl+C to stop\n")

    # Run all tasks
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Clean up memory
        if agent.memory is not None:
            agent.memory.store.close()


async def run_all_with_server(port: int = 8000):
    """Run API server + all channel listeners together.

    This mode enables:
    - CLI input
    - Email listener
    - SendBlue SMS webhooks (via /webhooks/sendblue)
    - All other configured channels

    The server and listeners share the same agent instance.
    """
    import uvicorn
    from config import load_config, is_channel_enabled, get_channel_config

    # Import server module - this creates the shared agent
    from server import app, agent, _register_server_senders

    console.banner("BabyAGI v0.3.0 - Full Mode (Server + Channels)")

    # Load config
    config = load_config()

    # Configure verbose level from config (env var takes precedence)
    if not os.environ.get("BABYAGI_VERBOSE"):
        verbose_config = config.get("verbose", "off")
        console.set_verbose(verbose_config)

    # Register all senders on the server's agent
    _register_senders(agent, config)

    # Build list of listener tasks
    tasks = []

    # Scheduler
    tasks.append(agent.run_scheduler())

    # Memory background extraction (if SQLite memory is active)
    if agent.memory is not None:
        memory_config = config.get("memory", {})
        if memory_config.get("background_extraction", True):
            from memory.integration import create_extraction_background_task
            extraction_loop = create_extraction_background_task(
                agent.memory,
                interval_seconds=memory_config.get("extraction_interval", 60)
            )
            tasks.append(extraction_loop())

    # CLI listener
    if is_channel_enabled(config, "cli"):
        from listeners.cli import run_cli_listener
        cli_config = get_channel_config(config, "cli")
        tasks.append(run_cli_listener(agent, cli_config))
        logger.info("CLI listener enabled")

    # Email listener
    if is_channel_enabled(config, "email"):
        if os.environ.get("AGENTMAIL_API_KEY"):
            from listeners.email import run_email_listener
            email_config = get_channel_config(config, "email")
            email_config["owner_email"] = config.get("owner", {}).get("email")
            tasks.append(run_email_listener(agent, email_config))
            logger.info("Email listener enabled")
        else:
            logger.warning("Email enabled but AGENTMAIL_API_KEY not set")

    # Voice listener
    if is_channel_enabled(config, "voice"):
        from listeners.voice import run_voice_listener
        voice_config = get_channel_config(config, "voice")
        tasks.append(run_voice_listener(agent, voice_config))
        logger.info("Voice listener enabled")

    # Note: SendBlue inbound is handled via webhook at /webhooks/sendblue
    # The polling listener is NOT needed in this mode
    if is_channel_enabled(config, "sendblue"):
        if os.environ.get("SENDBLUE_API_KEY") and os.environ.get("SENDBLUE_API_SECRET"):
            logger.info("SendBlue SMS enabled via webhook at /webhooks/sendblue")
        else:
            logger.warning("SendBlue enabled but API keys not set")

    # Create uvicorn server config
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)

    # Add server to tasks
    tasks.append(server.serve())

    # Log active channels
    active_channels = []
    if is_channel_enabled(config, "cli"):
        active_channels.append("cli")
    if is_channel_enabled(config, "email") and os.environ.get("AGENTMAIL_API_KEY"):
        active_channels.append("email")
    if is_channel_enabled(config, "voice"):
        active_channels.append("voice")
    if is_channel_enabled(config, "sendblue") and os.environ.get("SENDBLUE_API_KEY"):
        active_channels.append("sendblue (webhook)")

    console.system(f"\nActive channels: {', '.join(active_channels)}")
    console.system(f"API server: http://0.0.0.0:{port}")
    console.system(f"SendBlue webhook: http://0.0.0.0:{port}/webhooks/sendblue")
    console.system("Press Ctrl+C to stop\n")

    # Run everything
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if agent.memory is not None:
            agent.memory.store.close()


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

    # SendBlue sender (SMS/iMessage)
    if is_channel_enabled(config, "sendblue"):
        if os.environ.get("SENDBLUE_API_KEY") and os.environ.get("SENDBLUE_API_SECRET"):
            from senders.sendblue import SendBlueSender
            sendblue_config = get_channel_config(config, "sendblue")
            agent.register_sender("sendblue", SendBlueSender(sendblue_config))
            logger.info("SendBlue sender registered")


if __name__ == "__main__":
    main()
