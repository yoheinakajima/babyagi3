"""
Console Output Styling for BabyAGI CLI

Provides colored, formatted output for the terminal interface.
Supports verbose levels for showing internal operations.

Verbose Levels:
    - OFF (0): No verbose output, just user/agent messages
    - LIGHT (1): Key operations only (tool names, task starts) [default]
    - DEEP (2): Everything (tool inputs/outputs, full details)

Configuration:
    - Environment: BABYAGI_VERBOSE=0|1|2 or off|light|deep
    - Runtime: console.set_verbose(level) or tell the agent "turn off verbose"

Colors:
    - User messages: Green
    - Agent messages: Cyan
    - Verbose/logs: Yellow (light) / Dim gray (deep details)
    - Errors: Red
    - System/banner: Blue
"""

import os
import sys
from enum import IntEnum
from typing import Any


class VerboseLevel(IntEnum):
    """Verbose output levels."""
    OFF = 0
    LIGHT = 1
    DEEP = 2


def parse_verbose_level(value: str | int | VerboseLevel) -> VerboseLevel:
    """Parse a verbose level from string, int, or VerboseLevel.

    This is the single source of truth for verbose level parsing.
    Used by Console and tools/verbose.py.

    Args:
        value: String ("off", "light", "deep", "0", "1", "2"),
               int (0-2), or VerboseLevel enum.

    Returns:
        VerboseLevel enum value.

    Examples:
        parse_verbose_level("off") -> VerboseLevel.OFF
        parse_verbose_level("1") -> VerboseLevel.LIGHT
        parse_verbose_level(2) -> VerboseLevel.DEEP
    """
    if isinstance(value, VerboseLevel):
        return value

    if isinstance(value, int):
        return VerboseLevel(min(max(value, 0), 2))

    # String parsing
    level_str = str(value).lower().strip()

    # Map of string values to levels
    level_map = {
        # OFF
        "0": VerboseLevel.OFF,
        "off": VerboseLevel.OFF,
        "none": VerboseLevel.OFF,
        "false": VerboseLevel.OFF,
        # LIGHT
        "1": VerboseLevel.LIGHT,
        "light": VerboseLevel.LIGHT,
        "on": VerboseLevel.LIGHT,
        "true": VerboseLevel.LIGHT,
        # DEEP
        "2": VerboseLevel.DEEP,
        "deep": VerboseLevel.DEEP,
        "full": VerboseLevel.DEEP,
        "all": VerboseLevel.DEEP,
    }

    if level_str in level_map:
        return level_map[level_str]

    # Try numeric conversion
    try:
        return VerboseLevel(int(level_str))
    except (ValueError, TypeError):
        return VerboseLevel.OFF


# ANSI color codes
class Colors:
    """ANSI escape codes for terminal colors."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def _supports_color() -> bool:
    """Check if the terminal supports color output."""
    # Check for explicit disable
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("BABYAGI_NO_COLOR"):
        return False

    # Check for explicit enable
    if os.environ.get("FORCE_COLOR"):
        return True

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False

    # Check for dumb terminal
    if os.environ.get("TERM") == "dumb":
        return False

    return True


class Console:
    """
    Styled console output for CLI interface.

    Singleton-ish: import and use the `console` instance.

    Example:
        from utils.console import console

        console.user("Hello!")
        console.agent("Hi there!")
        console.verbose("Tool executed", level=VerboseLevel.LIGHT)
    """

    def __init__(self):
        self._verbose_level = self._load_verbose_level()
        self._use_color = _supports_color()
        self._filter_categories: set[str] | None = None  # None = all categories
        self._awaiting_input = False
        self._pending_logs: list[str] = []

    def _load_verbose_level(self) -> VerboseLevel:
        """Load verbose level from environment. Defaults to LIGHT."""
        env_val = os.environ.get("BABYAGI_VERBOSE", "1")
        return parse_verbose_level(env_val)

    def set_verbose(self, level: VerboseLevel | int | str):
        """
        Set verbose level at runtime.

        Args:
            level: VerboseLevel enum, int (0-2), or string ("off", "light", "deep")
        """
        self._verbose_level = parse_verbose_level(level)

    def get_verbose(self) -> VerboseLevel:
        """Get current verbose level."""
        return self._verbose_level

    def set_filter(self, categories: list[str] | None):
        """
        Filter verbose output to specific categories.

        Args:
            categories: List of categories to show, or None for all.
                       Categories: "tool", "objective", "scheduler", "memory"
        """
        self._filter_categories = set(categories) if categories else None

    def _colorize(self, text: str, *codes: str) -> str:
        """Apply color codes to text if color is supported."""
        if not self._use_color:
            return text
        return f"{''.join(codes)}{text}{Colors.RESET}"

    # -------------------------------------------------------------------------
    # Primary output methods
    # -------------------------------------------------------------------------

    def banner(self, text: str, width: int = 40):
        """Print a banner/header."""
        print(self._colorize(text, Colors.BOLD, Colors.BLUE), file=sys.stderr, flush=True)
        print(self._colorize("=" * width, Colors.DIM, Colors.BLUE), file=sys.stderr, flush=True)

    def user(self, text: str, prompt: str = "You"):
        """Print user input."""
        prefix = self._colorize(f"{prompt}: ", Colors.BOLD, Colors.GREEN)
        print(f"{prefix}{text}", file=sys.stderr, flush=True)

    def user_prompt(self) -> str:
        """Get styled user input prompt string."""
        return self._colorize("> ", Colors.BOLD, Colors.GREEN)

    def begin_input(self):
        """Signal that we're waiting for user input. Verbose logs will be buffered."""
        self._awaiting_input = True
        self._pending_logs = []

    def end_input(self):
        """Signal that user input has been received. Stop buffering logs."""
        self._awaiting_input = False

    def flush_pending_logs(self):
        """Print any verbose logs that were buffered during input waiting."""
        if self._pending_logs:
            for log_line in self._pending_logs:
                print(log_line, file=sys.stderr, flush=True)
            self._pending_logs = []

    def agent(self, text: str, prefix: str = "Assistant"):
        """Print agent response."""
        styled_prefix = self._colorize(f"{prefix}: ", Colors.BOLD, Colors.CYAN)
        print(f"{styled_prefix}{text}\n", file=sys.stderr, flush=True)

    def system(self, text: str):
        """Print system message (info, status)."""
        print(self._colorize(text, Colors.BLUE), file=sys.stderr, flush=True)

    def error(self, text: str):
        """Print error message."""
        print(self._colorize(f"Error: {text}", Colors.BOLD, Colors.RED), file=sys.stderr, flush=True)

    def success(self, text: str):
        """Print success message."""
        print(self._colorize(text, Colors.GREEN), file=sys.stderr, flush=True)

    def warning(self, text: str):
        """Print warning message."""
        print(self._colorize(f"Warning: {text}", Colors.YELLOW), file=sys.stderr, flush=True)

    # -------------------------------------------------------------------------
    # Verbose output methods
    # -------------------------------------------------------------------------

    def verbose(
        self,
        text: str,
        level: VerboseLevel = VerboseLevel.LIGHT,
        category: str = None
    ):
        """
        Print verbose/debug output if level is enabled.

        Args:
            text: Message to print
            level: Minimum verbose level required to show this
            category: Category for filtering ("tool", "objective", "scheduler")
        """
        if self._verbose_level < level:
            return

        if self._filter_categories and category and category not in self._filter_categories:
            return

        # Light level: yellow, more prominent
        # Deep level: dim gray, less prominent
        if level == VerboseLevel.LIGHT:
            styled = self._colorize(f"  {text}", Colors.YELLOW)
        else:
            styled = self._colorize(f"    {text}", Colors.DIM, Colors.BRIGHT_BLACK)

        if self._awaiting_input:
            self._pending_logs.append(styled)
        else:
            print(styled, file=sys.stderr, flush=True)

    def tool_start(self, name: str, inputs: dict[str, Any] = None):
        """Log tool execution start."""
        self.verbose(f"[tool] {name}", VerboseLevel.LIGHT, "tool")
        if inputs and self._verbose_level >= VerboseLevel.DEEP:
            safe_inputs = self._redact_dict(inputs, name)
            summary = self._summarize_dict(safe_inputs)
            self.verbose(f"  input: {summary}", VerboseLevel.DEEP, "tool")

    def tool_end(self, name: str, result: Any = None, duration_ms: int = None):
        """Log tool execution end."""
        timing = f" ({duration_ms}ms)" if duration_ms else ""
        self.verbose(f"[tool] {name} done{timing}", VerboseLevel.LIGHT, "tool")
        if result and self._verbose_level >= VerboseLevel.DEEP:
            safe_result = self._redact_dict(result, name) if isinstance(result, dict) else result
            summary = self._summarize_value(safe_result)
            self.verbose(f"  result: {summary}", VerboseLevel.DEEP, "tool")

    def objective_start(self, obj_id: str, goal: str):
        """Log objective start."""
        short_goal = goal[:50] + "..." if len(goal) > 50 else goal
        self.verbose(f"[objective] Starting {obj_id}: {short_goal}", VerboseLevel.LIGHT, "objective")

    def objective_end(self, obj_id: str, status: str):
        """Log objective end."""
        self.verbose(f"[objective] {obj_id} {status}", VerboseLevel.LIGHT, "objective")

    def task_start(self, task_id: str, name: str):
        """Log scheduled task start."""
        self.verbose(f"[scheduler] Running: {name} ({task_id})", VerboseLevel.LIGHT, "scheduler")

    def task_end(self, task_id: str, status: str, duration_ms: int = None):
        """Log scheduled task end."""
        timing = f" ({duration_ms}ms)" if duration_ms else ""
        self.verbose(f"[scheduler] {task_id} {status}{timing}", VerboseLevel.LIGHT, "scheduler")

    def memory_op(self, action: str, detail: str = None):
        """Log memory operation."""
        msg = f"[memory] {action}"
        if detail:
            msg += f": {detail[:50]}"
        self.verbose(msg, VerboseLevel.DEEP, "memory")

    def activity(self, channel: str, detail: str):
        """Log cross-channel or webhook activity.

        Replaces raw uvicorn/HTTP access logs with styled, concise output.
        Only shown at LIGHT verbose level and above.

        Examples:
            console.activity("sendblue", "inbound message from +1234567890")
            console.activity("recall", "bot status → in_call_recording")
            console.activity("email", "new message from user@example.com")
        """
        self.verbose(f"[{channel}] {detail}", VerboseLevel.LIGHT, "activity")

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    # Parameter names whose values are ALWAYS redacted in verbose output
    _SENSITIVE_KEYS = frozenset({
        "password", "secret", "api_key", "api_secret", "auth_token",
        "token", "card_number", "card_cvv", "oauth_client_secret",
        "access_token", "refresh_token", "private_key",
    })

    # Tools where the generic "value" parameter holds a secret
    _SENSITIVE_VALUE_TOOLS = frozenset({
        "store_secret",
    })

    def _redact_dict(self, d: dict, tool_name: str = None) -> dict:
        """Return a shallow copy of d with sensitive values replaced by '***'.

        Redaction rules:
        - Keys in _SENSITIVE_KEYS are always redacted
        - The 'value' key is redacted for tools in _SENSITIVE_VALUE_TOOLS
        - Nested dicts are recursively redacted
        """
        if not isinstance(d, dict):
            return d
        redacted = {}
        for k, v in d.items():
            k_lower = k.lower()
            if k_lower in self._SENSITIVE_KEYS:
                redacted[k] = "***"
            elif k_lower == "value" and tool_name in self._SENSITIVE_VALUE_TOOLS:
                redacted[k] = "***"
            elif isinstance(v, dict):
                redacted[k] = self._redact_dict(v, tool_name)
            elif isinstance(v, list):
                redacted[k] = [
                    self._redact_dict(item, tool_name) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                redacted[k] = v
        return redacted

    def _summarize_dict(self, d: dict, max_len: int = 80) -> str:
        """Summarize a dict for display."""
        if not d:
            return "{}"
        parts = []
        for k, v in d.items():
            v_str = self._summarize_value(v, max_len=30)
            parts.append(f"{k}={v_str}")
        result = ", ".join(parts)
        if len(result) > max_len:
            result = result[:max_len-3] + "..."
        return "{" + result + "}"

    def _summarize_value(self, v: Any, max_len: int = 60) -> str:
        """Summarize a value for display."""
        if v is None:
            return "null"
        if isinstance(v, bool):
            return str(v).lower()
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            if len(v) > max_len:
                return f'"{v[:max_len-3]}..."'
            return f'"{v}"'
        if isinstance(v, dict):
            return self._summarize_dict(v, max_len)
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return "[]"
            if len(v) <= 3:
                items = ", ".join(self._summarize_value(x, max_len=20) for x in v)
                return f"[{items}]"
            return f"[...{len(v)} items]"
        return str(v)[:max_len]


# Global console instance
console = Console()
