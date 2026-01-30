"""
CLI Sender - Output to terminal with styled formatting.

Sender that prints messages to stdout using console styling.
Used for CLI channel responses and agent-initiated messages.
"""

from utils.console import console


class CLISender:
    """Sender that outputs to terminal with styling."""

    name = "cli"
    capabilities = ["text_only"]

    def __init__(self, prefix: str = "Assistant"):
        self.prefix = prefix

    async def send(self, to: str, content: str, **kwargs) -> dict:
        """Print styled message to terminal.

        Args:
            to: Ignored for CLI (always prints to stdout)
            content: Message to print
        """
        console.agent(content, prefix=self.prefix)
        return {"sent": True, "channel": "cli"}
