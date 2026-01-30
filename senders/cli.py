"""
CLI Sender - Output to terminal.

Simple sender that prints messages to stdout.
Used for CLI channel responses and debugging.
"""


class CLISender:
    """Sender that outputs to terminal."""

    name = "cli"
    capabilities = ["text_only"]

    def __init__(self, prefix: str = "Assistant"):
        self.prefix = prefix

    async def send(self, to: str, content: str, **kwargs) -> dict:
        """Print message to terminal.

        Args:
            to: Ignored for CLI (always prints to stdout)
            content: Message to print
        """
        print(f"\n{self.prefix}: {content}\n")
        return {"sent": True, "channel": "cli"}
