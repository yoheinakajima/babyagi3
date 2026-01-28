"""
Tools Framework

Elegant auto-registration of tools using decorators and type hints.

Usage:
    from tools import tool

    @tool
    def web_search(query: str, max_results: int = 5) -> dict:
        '''Search the web.

        Args:
            query: What to search for
            max_results: Number of results
        '''
        return {"results": [...]}

The @tool decorator:
- Generates JSON schema from type hints
- Extracts descriptions from docstrings
- Auto-registers when the module loads
"""

import inspect
import re
from typing import Callable, get_type_hints

# Import Tool from agent.py (will be set during init)
_Tool = None
_registered_tools = []


def _python_type_to_json(py_type) -> dict:
    """Convert Python type hints to JSON schema types."""
    type_map = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        list: {"type": "array"},
        dict: {"type": "object"},
    }

    # Handle Optional types
    origin = getattr(py_type, "__origin__", None)
    if origin is type(None):
        return {"type": "null"}

    # Handle basic types
    if py_type in type_map:
        return type_map[py_type]

    # Default to string
    return {"type": "string"}


def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Parse a docstring to extract description and argument descriptions.

    Returns:
        (main_description, {arg_name: arg_description})
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    description_lines = []
    arg_descriptions = {}

    in_args = False
    current_arg = None

    for line in lines:
        stripped = line.strip()

        # Check for Args: section
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            continue

        # Check for other sections that end Args
        if stripped.lower() in ("returns:", "raises:", "examples:", "example:"):
            in_args = False
            continue

        if in_args:
            # Match "arg_name: description" or "arg_name (type): description"
            match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", stripped)
            if match:
                current_arg = match.group(1)
                arg_descriptions[current_arg] = match.group(2).strip()
            elif current_arg and stripped:
                # Continuation of previous arg description
                arg_descriptions[current_arg] += " " + stripped
        else:
            if stripped:
                description_lines.append(stripped)

    return " ".join(description_lines), arg_descriptions


def tool(fn: Callable = None, *, name: str = None, description: str = None):
    """
    Decorator to convert a function into a Tool.

    Can be used as:
        @tool
        def my_func(...): ...

    Or with options:
        @tool(name="custom_name", description="Custom description")
        def my_func(...): ...
    """
    def decorator(func: Callable):
        tool_name = name or func.__name__

        # Parse docstring for descriptions
        doc_desc, arg_descs = _parse_docstring(func.__doc__ or "")
        tool_description = description or doc_desc or f"Tool: {tool_name}"

        # Get type hints
        hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
        hints.pop("return", None)  # Remove return type

        # Get function signature for defaults
        sig = inspect.signature(func)

        # Build JSON schema properties
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            if param_name in ("agent", "self"):
                continue

            # Get type from hints or default to string
            param_type = hints.get(param_name, str)
            prop = _python_type_to_json(param_type)

            # Add description from docstring
            if param_name in arg_descs:
                prop["description"] = arg_descs[param_name]

            properties[param_name] = prop

            # Check if required (no default value)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required

        # Create wrapper that handles the agent parameter
        def wrapper(params: dict, agent):
            # Build kwargs, filtering to only params the function accepts
            sig_params = set(sig.parameters.keys()) - {"agent"}
            kwargs = {k: v for k, v in params.items() if k in sig_params}

            # Check if function expects agent parameter
            if "agent" in sig.parameters:
                return func(agent=agent, **kwargs)
            return func(**kwargs)

        # Store tool info for later registration
        tool_info = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
            "fn": wrapper,
            "original_fn": func,
        }
        _registered_tools.append(tool_info)

        # Return the original function so it can still be called directly
        func._tool_info = tool_info
        return func

    # Handle both @tool and @tool(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator


def get_all_tools(tool_class):
    """
    Get all registered tools as Tool instances.

    Args:
        tool_class: The Tool class from agent.py
    """
    # Import all tool modules to trigger registration
    from tools import web, email, secrets

    tools = []
    for info in _registered_tools:
        t = tool_class(
            name=info["name"],
            description=info["description"],
            parameters=info["parameters"],
            fn=info["fn"]
        )
        tools.append(t)

    return tools


def init_tools(tool_class):
    """Initialize the tools framework with the Tool class."""
    global _Tool
    _Tool = tool_class
