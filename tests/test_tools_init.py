"""
Unit tests for tools/__init__.py

Tests cover:
- @tool decorator (both @tool and @tool(...) syntax)
- Schema generation from type hints
- Docstring parsing for descriptions and argument docs
- Parameter default handling (required vs optional)
- Health check system (check_requirements, check_tool_health)
- tool_error helper
- Type conversion (_python_type_to_json)
"""

import os
from unittest.mock import patch

import pytest

# We need to be careful with tool registration since it uses a global list.
# Import fresh for each test module.
import tools
from tools import (
    tool,
    tool_error,
    check_requirements,
    _python_type_to_json,
    _parse_docstring,
    _registered_tools,
)


# =============================================================================
# Type Conversion Tests
# =============================================================================


class TestPythonTypeToJson:
    """Test _python_type_to_json conversion."""

    def test_str(self):
        assert _python_type_to_json(str) == {"type": "string"}

    def test_int(self):
        assert _python_type_to_json(int) == {"type": "integer"}

    def test_float(self):
        assert _python_type_to_json(float) == {"type": "number"}

    def test_bool(self):
        assert _python_type_to_json(bool) == {"type": "boolean"}

    def test_list(self):
        assert _python_type_to_json(list) == {"type": "array"}

    def test_dict(self):
        assert _python_type_to_json(dict) == {"type": "object"}

    def test_unknown_defaults_to_string(self):
        assert _python_type_to_json(bytes) == {"type": "string"}


# =============================================================================
# Docstring Parsing Tests
# =============================================================================


class TestParseDocstring:
    """Test _parse_docstring extraction."""

    def test_simple_description(self):
        desc, args = _parse_docstring("Search the web for results.")
        assert desc == "Search the web for results."
        assert args == {}

    def test_with_args_section(self):
        docstring = """Search the web.

        Args:
            query: The search query
            limit: Maximum number of results
        """
        desc, args = _parse_docstring(docstring)
        assert "Search the web" in desc
        assert "query" in args
        assert "limit" in args
        assert "search query" in args["query"]

    def test_multiline_arg_description(self):
        docstring = """Tool description.

        Args:
            query: The search query that can be
                very long and span multiple lines
        """
        desc, args = _parse_docstring(docstring)
        assert "query" in args
        assert "very long" in args["query"]

    def test_empty_docstring(self):
        desc, args = _parse_docstring("")
        assert desc == ""
        assert args == {}

    def test_none_docstring(self):
        desc, args = _parse_docstring(None)
        assert desc == ""
        assert args == {}

    def test_returns_section_ends_args(self):
        docstring = """Tool.

        Args:
            query: Search query
        Returns:
            Dict with results
        """
        desc, args = _parse_docstring(docstring)
        assert "query" in args
        assert len(args) == 1

    def test_arg_with_type_annotation(self):
        docstring = """Tool.

        Args:
            query (str): The search query
        """
        desc, args = _parse_docstring(docstring)
        assert "query" in args
        assert "search query" in args["query"]


# =============================================================================
# @tool Decorator Tests
# =============================================================================


class TestToolDecorator:
    """Test the @tool decorator."""

    def setup_method(self):
        """Clear registered tools before each test."""
        _registered_tools.clear()

    def test_bare_decorator(self):
        @tool
        def my_func(query: str) -> dict:
            """Search for something."""
            return {"results": []}

        assert len(_registered_tools) == 1
        info = _registered_tools[0]
        assert info["name"] == "my_func"
        assert "Search for something" in info["description"]

    def test_decorator_with_options(self):
        @tool(name="custom_search", packages=["httpx"], env=["API_KEY"])
        def search(query: str) -> dict:
            """Custom search tool."""
            return {}

        assert len(_registered_tools) == 1
        info = _registered_tools[0]
        assert info["name"] == "custom_search"
        assert "httpx" in info["packages"]
        assert "API_KEY" in info["env"]

    def test_schema_generation(self):
        @tool
        def web_search(query: str, max_results: int = 5) -> dict:
            """Search the web.

            Args:
                query: The search term
                max_results: Maximum results to return
            """
            return {}

        info = _registered_tools[0]
        schema = info["parameters"]
        assert schema["type"] == "object"
        assert "query" in schema["properties"]
        assert "max_results" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"
        assert schema["properties"]["max_results"]["type"] == "integer"

    def test_required_vs_optional(self):
        @tool
        def my_tool(required_arg: str, optional_arg: int = 10) -> dict:
            """A tool."""
            return {}

        info = _registered_tools[0]
        schema = info["parameters"]
        assert "required_arg" in schema["required"]
        assert "optional_arg" not in schema.get("required", [])

    def test_agent_param_excluded_from_schema(self):
        @tool
        def my_tool(query: str, agent=None) -> dict:
            """A tool."""
            return {}

        info = _registered_tools[0]
        schema = info["parameters"]
        assert "agent" not in schema["properties"]

    def test_self_param_excluded_from_schema(self):
        @tool
        def my_tool(self, query: str) -> dict:
            """A tool."""
            return {}

        info = _registered_tools[0]
        schema = info["parameters"]
        assert "self" not in schema["properties"]

    def test_wrapper_sync_function(self):
        @tool
        def sync_tool(query: str) -> dict:
            """Sync tool."""
            return {"query": query}

        info = _registered_tools[0]
        result = info["fn"]({"query": "test"}, agent=None)
        assert result == {"query": "test"}

    def test_wrapper_async_function(self):
        @tool
        async def async_tool(query: str) -> dict:
            """Async tool."""
            return {"query": query}

        info = _registered_tools[0]
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            info["fn"]({"query": "test"}, agent=None)
        )
        assert result == {"query": "test"}

    def test_wrapper_filters_unknown_params(self):
        @tool
        def simple_tool(query: str) -> dict:
            """A tool."""
            return {"query": query}

        info = _registered_tools[0]
        # Pass extra params that the function doesn't accept
        result = info["fn"]({"query": "test", "extra": "ignored"}, agent=None)
        assert result == {"query": "test"}

    def test_original_function_still_callable(self):
        @tool
        def callable_tool(query: str) -> dict:
            """A tool."""
            return {"query": query}

        # The original function should still be directly callable
        result = callable_tool("direct call")
        assert result == {"query": "direct call"}

    def test_tool_info_attached(self):
        @tool
        def info_tool(query: str) -> dict:
            """A tool."""
            return {}

        assert hasattr(info_tool, "_tool_info")
        assert info_tool._tool_info["name"] == "info_tool"

    def test_no_docstring_fallback(self):
        @tool
        def no_doc(x: str) -> dict:
            return {}

        info = _registered_tools[0]
        assert "Tool: no_doc" in info["description"]

    def test_description_override(self):
        @tool(description="Custom description")
        def override_tool(x: str) -> dict:
            """Original docstring."""
            return {}

        info = _registered_tools[0]
        assert info["description"] == "Custom description"


# =============================================================================
# tool_error Helper Tests
# =============================================================================


class TestToolError:
    """Test tool_error response helper."""

    def test_basic_error(self):
        result = tool_error("Something went wrong")
        assert result == {"error": "Something went wrong"}

    def test_error_with_fix(self):
        result = tool_error("API key not set", fix="Set MY_API_KEY env var")
        assert result["error"] == "API key not set"
        assert result["fix"] == "Set MY_API_KEY env var"

    def test_error_with_extras(self):
        result = tool_error("Not found", status_code=404, url="/api/test")
        assert result["error"] == "Not found"
        assert result["status_code"] == 404
        assert result["url"] == "/api/test"


# =============================================================================
# Health Check Tests
# =============================================================================


class TestCheckRequirements:
    """Test check_requirements utility."""

    def test_all_satisfied(self):
        result = check_requirements(packages=["os", "sys"], env=[])
        assert result["ready"] is True
        assert result["missing_packages"] == []
        assert result["missing_env"] == []

    def test_missing_package(self):
        result = check_requirements(packages=["nonexistent_package_xyz"])
        assert result["ready"] is False
        assert "nonexistent_package_xyz" in result["missing_packages"]

    def test_missing_env(self):
        env = os.environ.copy()
        env.pop("TOTALLY_MISSING_VAR", None)
        with patch.dict(os.environ, env, clear=True):
            result = check_requirements(env=["TOTALLY_MISSING_VAR"])
            assert result["ready"] is False
            assert "TOTALLY_MISSING_VAR" in result["missing_env"]

    def test_env_var_set(self):
        with patch.dict(os.environ, {"MY_TEST_KEY": "abc"}):
            result = check_requirements(env=["MY_TEST_KEY"])
            assert result["ready"] is True

    def test_empty_requirements(self):
        result = check_requirements()
        assert result["ready"] is True

    def test_mixed_missing(self):
        env = os.environ.copy()
        env.pop("MISSING_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            result = check_requirements(
                packages=["nonexistent_pkg"],
                env=["MISSING_KEY"]
            )
            assert result["ready"] is False
            assert len(result["missing_packages"]) == 1
            assert len(result["missing_env"]) == 1
