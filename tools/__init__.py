"""
Tools Framework

Elegant auto-registration of tools using decorators and type hints.

Usage:
    from tools import tool

    @tool
    def web_search(query: str, max_results: int = 5) -> dict:
        '''Search the web.'''
        return {"results": [...]}

    # With requirements metadata:
    @tool(packages=["httpx"], env=["BROWSER_USE_API_KEY"])
    def browse(task: str) -> dict:
        '''Browse the web.'''
        ...

The @tool decorator:
- Generates JSON schema from type hints
- Extracts descriptions from docstrings
- Tracks package and environment variable requirements
- Auto-registers when the module loads
"""

import inspect
import os
import re
from typing import Callable, get_type_hints

# Import Tool from agent.py (will be set during init)
_Tool = None
_registered_tools = []


# =============================================================================
# Tool Response Helpers
# =============================================================================

def tool_error(error: str, fix: str = None, **extras) -> dict:
    """Create a standardized tool error response.

    Use this helper to ensure consistent error format across all tools.

    Args:
        error: The error message
        fix: Optional hint on how to fix the error
        **extras: Additional fields to include in the response

    Returns:
        Dict with error key and optional fix/extra fields.

    Example:
        return tool_error("API key not set", fix="Set MYAPI_KEY env var")
        return tool_error("Not found", status_code=404)
    """
    result = {"error": error}
    if fix:
        result["fix"] = fix
    result.update(extras)
    return result


# =============================================================================
# Health Check Utilities (Reusable)
# =============================================================================

def check_requirements(packages: list[str] = None, env: list[str] = None) -> dict:
    """
    Check if packages are importable and environment variables are set.

    This is the core reusable health check function. Can be used by:
    - The @tool decorator's generated health checks
    - Dynamic tools created via register_tool
    - Any code that needs to verify requirements

    Args:
        packages: List of package names to check (e.g., ["httpx", "bs4"])
        env: List of environment variable names (e.g., ["API_KEY"])

    Returns:
        {
            "ready": True/False,
            "missing_packages": [...],
            "missing_env": [...],
        }
    """
    packages = packages or []
    env = env or []

    missing_packages = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)

    missing_env = [var for var in env if not os.environ.get(var)]

    return {
        "ready": not missing_packages and not missing_env,
        "missing_packages": missing_packages,
        "missing_env": missing_env,
    }


# =============================================================================
# Type Conversion and Parsing
# =============================================================================

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


# =============================================================================
# Tool Decorator
# =============================================================================

def tool(
    fn: Callable = None,
    *,
    name: str = None,
    description: str = None,
    packages: list[str] = None,
    env: list[str] = None
):
    """
    Decorator to convert a function into a Tool.

    Can be used as:
        @tool
        def my_func(...): ...

    Or with options:
        @tool(name="custom_name", packages=["requests"], env=["API_KEY"])
        def my_func(...): ...

    Args:
        name: Override the tool name (defaults to function name)
        description: Override description (defaults to docstring)
        packages: List of required package imports (e.g., ["httpx", "bs4"])
        env: List of required environment variables (e.g., ["API_KEY"])
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

        # Store tool info for later registration (including requirements)
        tool_info = {
            "name": tool_name,
            "description": tool_description,
            "parameters": schema,
            "fn": wrapper,
            "original_fn": func,
            # Requirements metadata
            "packages": packages or [],
            "env": env or [],
        }
        _registered_tools.append(tool_info)

        # Return the original function so it can still be called directly
        func._tool_info = tool_info
        return func

    # Handle both @tool and @tool(...) syntax
    if fn is not None:
        return decorator(fn)
    return decorator


# =============================================================================
# Tool Registration and Discovery
# =============================================================================

def get_all_tools(tool_class):
    """
    Get all registered tools as Tool instances.

    Args:
        tool_class: The Tool class from agent.py
    """
    # Import all tool modules to trigger registration
    from tools import web, email, secrets, verbose, credentials, metrics

    tools = []
    for info in _registered_tools:
        t = tool_class(
            name=info["name"],
            description=info["description"],
            parameters=info["parameters"],
            fn=info["fn"],
            packages=info.get("packages", []),
            env=info.get("env", []),
        )
        tools.append(t)

    return tools


def get_registered_tool_info() -> list[dict]:
    """
    Get metadata for all registered tools (without instantiating).

    Useful for health checks before full initialization.
    """
    # Ensure tools are imported
    try:
        from tools import web, email, secrets, verbose, credentials, metrics
    except ImportError:
        pass

    return _registered_tools.copy()


def init_tools(tool_class):
    """Initialize the tools framework with the Tool class."""
    global _Tool
    _Tool = tool_class


# =============================================================================
# Health Check System
# =============================================================================

def check_tool_health(include_core: bool = True) -> dict:
    """
    Check the health/availability of all registered tools.

    Reads requirements metadata from each tool's decorator.
    Works with both static tools and dynamically registered ones.

    Args:
        include_core: Whether to include core tools (memory, objective, etc.)

    Returns:
        {
            "ready": ["tool1", "tool2", ...],
            "needs_setup": [{"name": "tool", "missing": ["VAR"], "reason": "api_keys"}],
            "unavailable": [{"name": "tool", "missing": ["pkg"], "reason": "packages"}],
            "summary": {"total_ready": N, "needs_api_keys": N, "needs_packages": N}
        }
    """
    health = {
        "ready": [],
        "needs_setup": [],
        "unavailable": [],
    }

    # Get all registered tools with their requirements
    tool_infos = get_registered_tool_info()

    for info in tool_infos:
        tool_name = info["name"]
        packages = info.get("packages", [])
        env_vars = info.get("env", [])

        # Use the reusable check_requirements function
        status = check_requirements(packages, env_vars)

        if status["missing_packages"]:
            health["unavailable"].append({
                "name": tool_name,
                "missing": status["missing_packages"],
                "reason": "packages"
            })
        elif status["missing_env"]:
            health["needs_setup"].append({
                "name": tool_name,
                "missing": status["missing_env"],
                "reason": "api_keys"
            })
        else:
            health["ready"].append(tool_name)

    # Add core tools (always available, no external deps)
    if include_core:
        core_tools = ["memory", "objective", "notes", "register_tool", "store_credential", "get_credential", "list_credentials"]
        health["ready"] = core_tools + health["ready"]

        # Check E2B for sandbox capability
        if os.environ.get("E2B_API_KEY"):
            health["ready"].append("sandbox")
        else:
            health["needs_setup"].append({
                "name": "sandbox",
                "missing": ["E2B_API_KEY"],
                "reason": "api_keys"
            })

    # Generate summary
    health["summary"] = {
        "total_ready": len(health["ready"]),
        "needs_api_keys": len(health["needs_setup"]),
        "needs_packages": len(health["unavailable"]),
    }

    return health


def get_health_summary() -> str:
    """
    Get a concise human-readable health summary for the AI greeting.
    """
    health = check_tool_health()

    lines = []

    # Ready tools
    if health["ready"]:
        lines.append(f"Ready: {', '.join(health['ready'])}")

    # Tools needing API keys
    if health["needs_setup"]:
        needs = [f"{t['name']}({','.join(t['missing'])})" for t in health["needs_setup"]]
        lines.append(f"Need API keys: {', '.join(needs)}")

    # Unavailable tools
    if health["unavailable"]:
        unavail = [f"{t['name']}({','.join(t['missing'])})" for t in health["unavailable"]]
        lines.append(f"Missing packages: {', '.join(unavail)}")

    return "\n".join(lines)
