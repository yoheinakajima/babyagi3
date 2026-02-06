"""
Integration tests for agent.py

Tests cover:
- Agent initialization and configuration
- Tool class creation and execution
- Tool registration and validation
- Objective dataclass lifecycle
- Thread management (get_thread, clear_thread)
- Event emission during tool execution
- Sender registration
- json_serialize helper
- ToolValidationError / BudgetExceededException / ObjectiveCancelledException
- ToolLike protocol
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# We import specific items to avoid triggering heavy side effects from the full Agent init
from agent import (
    Tool,
    Objective,
    ToolValidationError,
    BudgetExceededException,
    ObjectiveCancelledException,
    ToolLike,
    json_serialize,
)


# =============================================================================
# json_serialize Tests
# =============================================================================


class TestJsonSerialize:
    """Test the json_serialize helper."""

    def test_datetime_serialization(self):
        dt = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = json_serialize(dt)
        assert "2025-06-15" in result

    def test_non_serializable_raises(self):
        with pytest.raises(TypeError, match="not JSON serializable"):
            json_serialize(set())


# =============================================================================
# Exception Classes Tests
# =============================================================================


class TestExceptions:
    """Test custom exception classes."""

    def test_tool_validation_error(self):
        err = ToolValidationError("Invalid tool definition")
        assert str(err) == "Invalid tool definition"
        assert isinstance(err, Exception)

    def test_budget_exceeded(self):
        err = BudgetExceededException("Budget of $1.00 exceeded")
        assert "Budget" in str(err)

    def test_objective_cancelled(self):
        err = ObjectiveCancelledException("Objective was cancelled")
        assert "cancelled" in str(err)


# =============================================================================
# Objective Dataclass Tests
# =============================================================================


class TestObjective:
    """Test the Objective dataclass."""

    def test_minimal_creation(self):
        obj = Objective(id="obj1", goal="Research AI")
        assert obj.id == "obj1"
        assert obj.goal == "Research AI"
        assert obj.status == "pending"
        assert obj.thread_id == "objective_obj1"
        assert obj.created != ""
        assert obj.priority == 5

    def test_custom_thread_id(self):
        obj = Objective(id="obj2", goal="Test", thread_id="custom")
        assert obj.thread_id == "custom"

    def test_custom_priority(self):
        obj = Objective(id="obj3", goal="Urgent", priority=1)
        assert obj.priority == 1

    def test_budget_fields(self):
        obj = Objective(
            id="obj4", goal="Expensive task",
            budget_usd=5.0, token_limit=100000
        )
        assert obj.budget_usd == 5.0
        assert obj.token_limit == 100000
        assert obj.spent_usd == 0.0
        assert obj.tokens_used == 0

    def test_retry_fields(self):
        obj = Objective(id="obj5", goal="Retry me", max_retries=5)
        assert obj.retry_count == 0
        assert obj.max_retries == 5


# =============================================================================
# Tool Class Tests
# =============================================================================


class TestTool:
    """Test the Tool class."""

    def test_creation(self):
        def my_fn(params, agent):
            return {"result": "ok"}

        t = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            fn=my_fn,
        )
        assert t.name == "test_tool"
        assert t.schema["name"] == "test_tool"
        assert t.schema["description"] == "A test tool"
        assert t.packages == []
        assert t.env == []

    def test_with_requirements(self):
        def my_fn(params, agent):
            return {}

        t = Tool(
            name="web_tool",
            description="Web tool",
            parameters={"type": "object"},
            fn=my_fn,
            packages=["httpx"],
            env=["API_KEY"],
        )
        assert t.packages == ["httpx"]
        assert t.env == ["API_KEY"]

    def test_execute(self):
        def my_fn(params, agent):
            return {"sum": params["a"] + params["b"]}

        t = Tool(
            name="adder",
            description="Add numbers",
            parameters={"type": "object"},
            fn=my_fn,
        )
        result = t.execute({"a": 2, "b": 3}, agent=None)
        assert result == {"sum": 5}

    def test_check_health(self):
        def my_fn(params, agent):
            return {}

        t = Tool(
            name="healthy",
            description="test",
            parameters={},
            fn=my_fn,
            packages=["os"],
            env=[],
        )
        health = t.check_health()
        assert health["ready"] is True

    def test_schema_structure(self):
        def my_fn(params, agent):
            return {}

        params = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        t = Tool(
            name="search",
            description="Search tool",
            parameters=params,
            fn=my_fn,
        )
        assert t.schema["input_schema"] == params


# =============================================================================
# ToolLike Protocol Tests
# =============================================================================


class TestToolLikeProtocol:
    """Test ToolLike protocol duck typing."""

    def test_tool_is_toollike(self):
        def fn(params, agent):
            return {}

        t = Tool(name="t", description="d", parameters={}, fn=fn)
        # Tool should satisfy ToolLike protocol
        assert hasattr(t, "name")
        assert hasattr(t, "fn")
        assert hasattr(t, "schema")

    def test_duck_typed_object(self):
        class FakeTool:
            name = "fake"
            fn = lambda self, p, a: {}
            schema = {"name": "fake"}

        ft = FakeTool()
        assert isinstance(ft, ToolLike)


# =============================================================================
# Agent Initialization Tests (with mocked LLM)
# =============================================================================


class TestAgentInit:
    """Test Agent initialization with mocked dependencies."""

    @patch("agent.AsyncLiteLLMAnthropicAdapter")
    @patch("agent.set_event_emitter")
    @patch("agent.init_llm_config")
    @patch("agent.get_llm_config")
    def test_basic_init(self, mock_get_config, mock_init_config, mock_set_emitter, mock_adapter):
        """Test that Agent initializes without errors."""
        mock_model_config = MagicMock()
        mock_model_config.agent_model.model_id = "test-model"
        mock_get_config.return_value = mock_model_config

        from agent import Agent
        agent = Agent(load_tools=False)

        assert agent.threads == {"main": []}
        assert agent.objectives == {}
        # Core tools are always registered (memory, objective, notes, etc.)
        assert len(agent.tools) > 0
        assert "memory" in agent.tools
        assert agent.senders == {}

    @patch("agent.AsyncLiteLLMAnthropicAdapter")
    @patch("agent.set_event_emitter")
    @patch("agent.init_llm_config")
    @patch("agent.get_llm_config")
    def test_tool_registration(self, mock_get_config, mock_init_config, mock_set_emitter, mock_adapter):
        """Test registering a tool."""
        mock_model_config = MagicMock()
        mock_model_config.agent_model.model_id = "test-model"
        mock_get_config.return_value = mock_model_config

        from agent import Agent
        agent = Agent(load_tools=False)

        def my_fn(params, agent):
            return {"ok": True}

        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object", "properties": {}},
            fn=my_fn,
        )
        agent.register(tool)
        assert "test_tool" in agent.tools
        assert agent.tools["test_tool"].schema["name"] == "test_tool"

    @patch("agent.AsyncLiteLLMAnthropicAdapter")
    @patch("agent.set_event_emitter")
    @patch("agent.init_llm_config")
    @patch("agent.get_llm_config")
    def test_sender_registration(self, mock_get_config, mock_init_config, mock_set_emitter, mock_adapter):
        """Test registering a sender."""
        mock_model_config = MagicMock()
        mock_model_config.agent_model.model_id = "test-model"
        mock_get_config.return_value = mock_model_config

        from agent import Agent
        agent = Agent(load_tools=False)

        mock_sender = MagicMock()
        mock_sender.name = "test_channel"
        agent.register_sender("test", mock_sender)
        assert "test" in agent.senders

    @patch("agent.AsyncLiteLLMAnthropicAdapter")
    @patch("agent.set_event_emitter")
    @patch("agent.init_llm_config")
    @patch("agent.get_llm_config")
    def test_thread_management(self, mock_get_config, mock_init_config, mock_set_emitter, mock_adapter):
        """Test get_thread and clear_thread."""
        mock_model_config = MagicMock()
        mock_model_config.agent_model.model_id = "test-model"
        mock_get_config.return_value = mock_model_config

        from agent import Agent
        agent = Agent(load_tools=False)

        # Add some messages to a thread
        agent.threads["test_thread"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        thread = agent.get_thread("test_thread")
        assert len(thread) == 2

        agent.clear_thread("test_thread")
        assert agent.get_thread("test_thread") == []

    @patch("agent.AsyncLiteLLMAnthropicAdapter")
    @patch("agent.set_event_emitter")
    @patch("agent.init_llm_config")
    @patch("agent.get_llm_config")
    def test_event_emission(self, mock_get_config, mock_init_config, mock_set_emitter, mock_adapter):
        """Test that Agent can emit events (inherits EventEmitter)."""
        mock_model_config = MagicMock()
        mock_model_config.agent_model.model_id = "test-model"
        mock_get_config.return_value = mock_model_config

        from agent import Agent
        agent = Agent(load_tools=False)

        received = []
        agent.on("test_event", lambda d: received.append(d))
        agent.emit("test_event", {"data": "hello"})
        assert len(received) == 1
        assert received[0]["data"] == "hello"
