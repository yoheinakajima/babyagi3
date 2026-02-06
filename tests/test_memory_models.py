"""
Unit tests for memory/models.py

Tests cover:
- Event dataclass construction and repr
- Entity dataclass construction and repr
- Edge dataclass construction and repr
- Topic dataclass construction and repr
- EventTopic junction model
- Task model
- SummaryNode model
- AgentState model
- ToolRecord and ToolDefinition (including computed properties)
- Credential model
- Learning model
- ExtractedFeedback model
- Fact model
- Extraction models (ExtractedFact, ExtractedEntity, ExtractedEdge, ExtractedTopic)
- ExtractionResult model
- AssembledContext (to_dict, to_prompt)
"""

from datetime import datetime

import pytest

from memory.models import (
    Event,
    Entity,
    Edge,
    Topic,
    EventTopic,
    Task,
    SummaryNode,
    AgentState,
    ToolRecord,
    ToolDefinition,
    Credential,
    Learning,
    ExtractedFeedback,
    Fact,
    ExtractedFact,
    ExtractedEntity,
    ExtractedEdge,
    ExtractedTopic,
    ExtractionResult,
    AssembledContext,
    RetrievalQuery,
    RetrievalResult,
    ExtractionCall,
)


# =============================================================================
# Event Tests
# =============================================================================


class TestEvent:
    def test_minimal_creation(self):
        e = Event(
            id="evt1",
            timestamp=datetime.now(),
            channel="cli",
            direction="inbound",
            event_type="message",
        )
        assert e.id == "evt1"
        assert e.extraction_status == "pending"
        assert e.content == ""
        assert e.is_owner is False

    def test_full_creation(self):
        e = Event(
            id="evt2",
            timestamp=datetime.now(),
            channel="email",
            direction="outbound",
            event_type="tool_call",
            task_id="task1",
            tool_id="web_search",
            person_id="p1",
            is_owner=True,
            parent_event_id="evt1",
            conversation_id="conv1",
            content="Hello world",
            metadata={"key": "value"},
        )
        assert e.channel == "email"
        assert e.is_owner is True
        assert e.metadata == {"key": "value"}

    def test_repr(self):
        e = Event(
            id="abcdefgh_long",
            timestamp=datetime.now(),
            channel="cli",
            direction="inbound",
            event_type="message",
            content="Short",
        )
        r = repr(e)
        assert "abcdefgh" in r
        assert "message" in r

    def test_repr_long_content(self):
        e = Event(
            id="abcdefgh",
            timestamp=datetime.now(),
            channel="cli",
            direction="inbound",
            event_type="message",
            content="x" * 100,
        )
        r = repr(e)
        assert "..." in r


# =============================================================================
# Entity Tests
# =============================================================================


class TestEntity:
    def test_creation(self):
        e = Entity(
            id="ent1",
            name="John Smith",
            type="person",
            type_raw="entrepreneur",
        )
        assert e.name == "John Smith"
        assert e.aliases == []
        assert e.event_count == 0

    def test_with_aliases(self):
        e = Entity(
            id="ent2",
            name="OpenAI",
            type="org",
            type_raw="AI company",
            aliases=["OAI", "Open AI"],
        )
        assert len(e.aliases) == 2

    def test_repr(self):
        e = Entity(id="abcdefgh", name="Test", type="person", type_raw="human")
        r = repr(e)
        assert "Test" in r
        assert "person" in r


# =============================================================================
# Edge Tests
# =============================================================================


class TestEdge:
    def test_creation(self):
        e = Edge(
            id="edge1",
            source_entity_id="ent1",
            target_entity_id="ent2",
            relation="works at",
        )
        assert e.strength == 0.5
        assert e.is_current is True

    def test_custom_strength(self):
        e = Edge(
            id="edge2",
            source_entity_id="ent1",
            target_entity_id="ent2",
            relation="invested in",
            strength=0.9,
        )
        assert e.strength == 0.9

    def test_repr(self):
        e = Edge(
            id="abcdefgh",
            source_entity_id="ent1",
            target_entity_id="ent2",
            relation="knows",
            strength=0.75,
        )
        r = repr(e)
        assert "knows" in r
        assert "0.75" in r


# =============================================================================
# Topic Tests
# =============================================================================


class TestTopic:
    def test_creation(self):
        t = Topic(id="top1", label="AI Research")
        assert t.event_count == 0
        assert t.keywords == []

    def test_with_keywords(self):
        t = Topic(
            id="top2",
            label="Machine Learning",
            keywords=["neural networks", "deep learning", "transformers"],
        )
        assert len(t.keywords) == 3

    def test_repr(self):
        t = Topic(
            id="abcdefgh",
            label="Science",
            keywords=["physics", "chemistry", "biology", "math"],
        )
        r = repr(t)
        assert "Science" in r
        # Should show first 3 keywords
        assert "physics" in r


class TestEventTopic:
    def test_creation(self):
        et = EventTopic(event_id="evt1", topic_id="top1")
        assert et.relevance == 1.0


# =============================================================================
# Task Tests
# =============================================================================


class TestTask:
    def test_creation(self):
        t = Task(id="task1", title="Research competitors")
        assert t.status == "pending"
        assert t.outcome is None

    def test_repr_long_title(self):
        t = Task(id="abcdefgh", title="A" * 50)
        r = repr(t)
        assert "..." in r


# =============================================================================
# SummaryNode Tests
# =============================================================================


class TestSummaryNode:
    def test_creation(self):
        s = SummaryNode(
            id="sn1",
            node_type="channel",
            key="channel:email",
            label="Email",
        )
        assert s.events_since_update == 0
        assert s.summary == ""

    def test_repr_long_summary(self):
        s = SummaryNode(
            id="abcdefgh",
            node_type="root",
            key="root",
            label="Root",
            summary="x" * 100,
        )
        r = repr(s)
        assert "..." in r


# =============================================================================
# AgentState Tests
# =============================================================================


class TestAgentState:
    def test_creation(self):
        a = AgentState(id="agent1", name="TestBot")
        assert a.current_topics == []
        assert a.active_tasks == []
        assert a.settings == {}

    def test_repr(self):
        a = AgentState(id="agent1", name="TestBot", mood="neutral")
        r = repr(a)
        assert "TestBot" in r
        assert "neutral" in r


# =============================================================================
# ToolDefinition Tests
# =============================================================================


class TestToolDefinition:
    def test_creation(self):
        td = ToolDefinition(
            id="td1",
            name="my_tool",
            description="A custom tool",
        )
        assert td.tool_type == "executable"
        assert td.category == "custom"
        assert td.is_enabled is True
        assert td.version == 1

    def test_success_rate_no_usage(self):
        td = ToolDefinition(id="td1", name="tool", description="test")
        assert td.success_rate == 100.0

    def test_success_rate_with_usage(self):
        td = ToolDefinition(
            id="td1", name="tool", description="test",
            usage_count=10, success_count=8,
        )
        assert td.success_rate == 80.0

    def test_is_healthy_low_usage(self):
        td = ToolDefinition(
            id="td1", name="tool", description="test",
            usage_count=2, success_count=0,
        )
        assert td.is_healthy is True  # < 3 uses, not enough data

    def test_is_healthy_above_threshold(self):
        td = ToolDefinition(
            id="td1", name="tool", description="test",
            usage_count=10, success_count=6,
        )
        assert td.is_healthy is True  # 60% >= 50%

    def test_is_unhealthy(self):
        td = ToolDefinition(
            id="td1", name="tool", description="test",
            usage_count=10, success_count=4,
        )
        assert td.is_healthy is False  # 40% < 50%

    def test_repr(self):
        td = ToolDefinition(
            id="td1", name="my_tool", description="test",
            usage_count=5, success_count=5,
        )
        r = repr(td)
        assert "my_tool" in r
        assert "enabled" in r
        assert "healthy" in r


class TestToolRecord:
    def test_creation(self):
        tr = ToolRecord(id="tr1", name="search")
        assert tr.usage_count == 0
        assert tr.description == ""

    def test_repr(self):
        tr = ToolRecord(id="tr1", name="search", usage_count=5)
        r = repr(tr)
        assert "search" in r
        assert "5" in r


# =============================================================================
# Credential Tests
# =============================================================================


class TestCredential:
    def test_account_credential(self):
        c = Credential(
            id="cred1",
            credential_type="account",
            service="github.com",
            username="testuser",
            email="test@example.com",
        )
        assert c.credential_type == "account"
        assert c.username == "testuser"

    def test_credit_card_credential(self):
        c = Credential(
            id="cred2",
            credential_type="credit_card",
            service="stripe.com",
            card_last_four="4242",
            card_type="visa",
        )
        assert c.card_last_four == "4242"

    def test_repr_account(self):
        c = Credential(
            id="cred1",
            credential_type="account",
            service="github.com",
            username="testuser",
        )
        r = repr(c)
        assert "github.com" in r
        assert "testuser" in r

    def test_repr_credit_card(self):
        c = Credential(
            id="cred2",
            credential_type="credit_card",
            service="stripe.com",
            card_last_four="4242",
        )
        r = repr(c)
        assert "credit_card" in r
        assert "4242" in r


# =============================================================================
# Learning Tests
# =============================================================================


class TestLearning:
    def test_creation(self):
        l = Learning(
            id="learn1",
            source_type="user_feedback",
            source_event_id="evt1",
            content="User prefers concise responses",
        )
        assert l.sentiment == "neutral"
        assert l.confidence == 0.5
        assert l.tool_id is None
        assert l.recommendation is None
        assert l.superseded_by is None

    def test_creation_with_tool(self):
        l = Learning(
            id="learn2",
            source_type="tool_error_pattern",
            content="Tool X has 40% success rate",
            tool_id="tool_x",
            sentiment="negative",
            confidence=0.8,
            recommendation="Validate parameters before use",
        )
        assert l.tool_id == "tool_x"
        assert l.recommendation == "Validate parameters before use"

    def test_superseded_by(self):
        old = Learning(
            id="old1",
            source_type="user_feedback",
            content="Keep emails long and detailed",
            sentiment="positive",
        )
        new = Learning(
            id="new1",
            source_type="user_feedback",
            content="Keep emails short",
            sentiment="negative",
        )
        # Simulate supersession
        old.superseded_by = new.id
        assert old.superseded_by == "new1"
        assert new.superseded_by is None

    def test_repr(self):
        l = Learning(
            id="abcdefgh",
            source_type="user_feedback",
            source_event_id="evt1",
            content="x" * 100,
            tool_id="web_search",
        )
        r = repr(l)
        assert "..." in r
        assert "web_search" in r


class TestExtractedFeedback:
    def test_no_feedback(self):
        f = ExtractedFeedback()
        assert f.has_feedback is False
        assert f.sentiment == "neutral"

    def test_with_feedback(self):
        f = ExtractedFeedback(
            has_feedback=True,
            feedback_type="correction",
            what_was_wrong="Too verbose",
            what_to_do_instead="Be concise",
        )
        assert f.feedback_type == "correction"


# =============================================================================
# Fact Tests
# =============================================================================


class TestFact:
    def test_creation(self):
        f = Fact(
            id="fact1",
            subject_entity_id="ent1",
            predicate="invested in",
        )
        assert f.fact_type == "relation"
        assert f.confidence == 0.8
        assert f.is_current is True

    def test_repr(self):
        f = Fact(
            id="abcdefgh",
            subject_entity_id="ent1",
            predicate="works at",
            object_entity_id="ent2",
        )
        r = repr(f)
        assert "works at" in r


# =============================================================================
# Extraction Models Tests
# =============================================================================


class TestExtractionModels:
    def test_extracted_entity(self):
        e = ExtractedEntity(name="Google", type_raw="tech company")
        assert e.aliases == []
        assert e.importance == 0.5

    def test_extracted_edge(self):
        e = ExtractedEdge(source="John", target="Google", relation="works at")
        assert e.strength == 0.5
        assert e.is_current is True

    def test_extracted_topic(self):
        t = ExtractedTopic(label="AI", keywords=["ml", "deep learning"])
        assert t.relevance == 1.0

    def test_extracted_fact(self):
        f = ExtractedFact(
            subject="John",
            predicate="invested in",
            object="TechCorp",
            fact_text="John invested in TechCorp",
        )
        assert f.object_type == "value"

    def test_extraction_result(self):
        er = ExtractionResult(
            entities=[ExtractedEntity(name="Test", type_raw="thing")],
            facts=[ExtractedFact(subject="A", predicate="is", object="B")],
        )
        assert len(er.entities) == 1
        assert len(er.facts) == 1


# =============================================================================
# AssembledContext Tests
# =============================================================================


class TestAssembledContext:
    def test_empty_context(self):
        ctx = AssembledContext()
        d = ctx.to_dict()
        assert "identity" in d
        assert "state" in d
        assert "knowledge" in d
        assert "recent" in d

    def test_to_dict_excludes_none(self):
        ctx = AssembledContext()
        d = ctx.to_dict()
        assert "channel" not in d
        assert "tool" not in d
        assert "counterparty" not in d

    def test_to_dict_includes_set_values(self):
        ctx = AssembledContext(
            channel={"name": "email", "summary": "Email channel"},
            user_preferences="Prefers concise responses",
        )
        d = ctx.to_dict()
        assert "channel" in d
        assert "user_preferences" in d

    def test_to_prompt_identity(self):
        ctx = AssembledContext(
            identity={"name": "TestBot", "description": "A test bot"},
        )
        prompt = ctx.to_prompt()
        assert "TestBot" in prompt
        assert "test bot" in prompt

    def test_to_prompt_knowledge(self):
        ctx = AssembledContext(knowledge="The sky is blue")
        prompt = ctx.to_prompt()
        assert "What I Know" in prompt
        assert "sky is blue" in prompt

    def test_to_prompt_user_preferences(self):
        ctx = AssembledContext(user_preferences="Likes bullet points")
        prompt = ctx.to_prompt()
        assert "User Preferences" in prompt
        assert "bullet points" in prompt

    def test_to_prompt_learnings(self):
        ctx = AssembledContext(
            learnings=[
                {"type": "tool", "tool": "web_search", "learning": "Use specific queries"},
                {"type": "general", "learning": "Be concise", "recommendation": "Use short sentences"},
            ]
        )
        prompt = ctx.to_prompt()
        assert "web_search" in prompt
        assert "Be concise" in prompt
        assert "short sentences" in prompt


# =============================================================================
# Retrieval & Extraction Call Models
# =============================================================================


class TestRetrievalModels:
    def test_retrieval_query(self):
        rq = RetrievalQuery(id="rq1", query_text="What is AI?")
        assert rq.total_results == 0
        assert rq.was_successful is False

    def test_retrieval_result(self):
        rr = RetrievalResult(
            id="rr1",
            query_id="rq1",
            result_type="fact",
            result_id="fact1",
            retrieval_method="semantic",
        )
        assert rr.was_used is False
        assert rr.method_step == 1


class TestExtractionCall:
    def test_creation(self):
        ec = ExtractionCall(
            id="ec1",
            source_type="conversation",
            content_length=500,
            entities_extracted=3,
            facts_extracted=2,
        )
        assert ec.cost_usd == 0.0
        assert ec.timing_mode == "immediate"
