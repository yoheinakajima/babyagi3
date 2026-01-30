"""
Verbose Mode Control Tool

Allows the agent to control and query the verbose output level at runtime.

Verbose Levels:
    - OFF (0): No verbose output, just user/agent messages
    - LIGHT (1): Key operations only (tool names, task starts)
    - DEEP (2): Everything (tool inputs/outputs, full details)
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

    # Validate input
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
            "error": f"Invalid level '{level}'. Valid options: off, light, deep (or 0, 1, 2)",
            "current_level": console.get_verbose().name.lower()
        }

    console.set_verbose(valid_levels[level_lower])
    new_level = console.get_verbose()

    return {
        "success": True,
        "level": new_level.name.lower(),
        "level_value": int(new_level),
        "description": _level_description(new_level)
    }


@tool
def get_verbose() -> dict:
    """Get the current verbose output level.

    Returns the current verbose setting and what it means.
    """
    current = console.get_verbose()

    return {
        "level": current.name.lower(),
        "level_value": int(current),
        "description": _level_description(current),
        "available_levels": [
            {"name": "off", "value": 0, "description": "No verbose output"},
            {"name": "light", "value": 1, "description": "Key operations only"},
            {"name": "deep", "value": 2, "description": "Full details"},
        ]
    }


def _level_description(level: VerboseLevel) -> str:
    """Get a human-readable description of a verbose level."""
    descriptions = {
        VerboseLevel.OFF: "No verbose output - only user and agent messages shown",
        VerboseLevel.LIGHT: "Light verbose - shows tool names and task starts",
        VerboseLevel.DEEP: "Deep verbose - shows all details including inputs/outputs",
    }
    return descriptions.get(level, "Unknown level")
