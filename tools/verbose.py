"""
Verbose Mode Control Tool

Allows the agent to control verbose output level at runtime.

Verbose Levels:
    - OFF (0): No verbose output, just user/agent messages
    - LIGHT (1): Key operations only (tool names, task starts)
    - DEEP (2): Everything (tool inputs/outputs, full details)

Note: Current verbose level is shown in the system prompt context.
"""

from tools import tool
from utils.console import console, VerboseLevel


@tool
def set_verbose(level: str) -> dict:
    """Set the verbose output level.

    Controls how much detail is shown in the console output.
    Changes take effect immediately.

    Args:
        level: The verbose level to set. Options:
               - "off" or "0": No verbose output
               - "light" or "1": Key operations only
               - "deep" or "2": Full details
    """
    level_lower = level.lower()

    valid_levels = {
        "off": VerboseLevel.OFF,
        "0": VerboseLevel.OFF,
        "light": VerboseLevel.LIGHT,
        "1": VerboseLevel.LIGHT,
        "on": VerboseLevel.LIGHT,
        "deep": VerboseLevel.DEEP,
        "2": VerboseLevel.DEEP,
        "full": VerboseLevel.DEEP,
    }

    if level_lower not in valid_levels:
        return {
            "success": False,
            "error": f"Invalid level '{level}'. Valid options: off, light, deep (or 0, 1, 2)"
        }

    console.set_verbose(valid_levels[level_lower])
    new_level = console.get_verbose()

    return {
        "success": True,
        "level": new_level.name.lower()
    }
