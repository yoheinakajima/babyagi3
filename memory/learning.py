"""
Self-improvement system for the memory system.

Extracts learnings from feedback and self-evaluation, retrieves relevant
learnings for context, and summarizes them into user preferences.
"""

import json
import logging
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

from metrics import LiteLLMAnthropicAdapter, track_source, get_model_for_use_case
from .embeddings import get_embedding
from .models import Event, ExtractedFeedback, Learning


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


# ═══════════════════════════════════════════════════════════
# FEEDBACK EXTRACTION
# ═══════════════════════════════════════════════════════════


class FeedbackExtractor:
    """Extracts feedback from user messages about prior AI work."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Get instrumented LLM client for metrics tracking (supports multiple providers)."""
        if self._client is None:
            self._client = LiteLLMAnthropicAdapter()
        return self._client

    @property
    def fast_model(self) -> str:
        """Get the configured fast model for quick classification tasks."""
        return get_model_for_use_case("fast")

    async def extract(
        self,
        event: Event,
        recent_events: list[Event],
    ) -> Learning | None:
        """
        Analyze a user message for feedback about prior work.

        Args:
            event: The incoming user message event
            recent_events: Recent AI actions for context

        Returns:
            Learning if feedback detected, None otherwise
        """
        # Skip very short messages
        if len(event.content.strip()) < 10:
            return None

        # Build context from recent AI actions
        recent_actions = self._format_recent_actions(recent_events)

        # Call LLM to detect feedback
        feedback = await self._llm_extract(event.content, recent_actions)

        if not feedback.has_feedback:
            return None

        # Create learning from feedback
        content = self._format_learning_content(feedback)

        # Determine category — use LLM-classified category, validate it
        valid_categories = {"general", "owner_profile", "agent_self", "tool_feedback"}
        category = feedback.category if feedback.category in valid_categories else "general"
        # Auto-classify tool feedback if tool is referenced
        if feedback.about_tool and category == "general":
            category = "tool_feedback"

        learning = Learning(
            id=generate_id(),
            source_type="user_feedback",
            source_event_id=event.id,
            content=content,
            content_embedding=get_embedding(content),
            sentiment=feedback.sentiment,
            confidence=feedback.confidence,
            category=category,
            tool_id=feedback.about_tool,
            objective_type=feedback.about_objective_type,
            applies_when=feedback.what_was_wrong,
            recommendation=feedback.what_to_do_instead,
            topic_ids=[],
            entity_ids=[feedback.about_entity_id] if feedback.about_entity_id else [],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        return learning

    def _format_recent_actions(self, events: list[Event]) -> str:
        """Format recent events for the extraction prompt."""
        if not events:
            return "No recent actions."

        lines = []
        for e in events[:10]:  # Limit to 10 most recent
            content = e.content[:300] if len(e.content) > 300 else e.content
            if e.tool_id:
                lines.append(f"- Tool '{e.tool_id}': {content}")
            elif e.event_type == "message" and e.direction == "outbound":
                lines.append(f"- Response: {content}")
            else:
                lines.append(f"- {e.event_type}: {content}")

        return "\n".join(lines)

    def _format_learning_content(self, feedback: ExtractedFeedback) -> str:
        """Format feedback into a learning content string."""
        parts = []

        if feedback.about_tool:
            parts.append(f"For tool '{feedback.about_tool}':")
        elif feedback.about_objective_type:
            parts.append(f"For {feedback.about_objective_type} tasks:")

        if feedback.what_was_wrong:
            parts.append(f"Issue: {feedback.what_was_wrong}")

        if feedback.what_to_do_instead:
            parts.append(f"Preference: {feedback.what_to_do_instead}")

        if not parts:
            return f"User feedback ({feedback.feedback_type})"

        return " ".join(parts)

    async def _llm_extract(
        self, content: str, recent_actions: str
    ) -> ExtractedFeedback:
        """Call LLM to extract feedback from message."""
        prompt = f"""Analyze this user message for feedback about prior AI work.

MESSAGE:
{content}

RECENT AI ACTIONS:
{recent_actions}

Determine if the message contains feedback (correction, praise, preference, complaint) about:
- A specific tool that was used
- A type of task/objective
- General working style
- Owner profile info (facts about themselves: location, interests, schedule, role, etc.)
- Agent self-improvement (how the agent should behave, its role, boundaries)

Return JSON:
{{
    "has_feedback": true/false,
    "feedback_type": "correction" | "praise" | "preference" | "complaint" | "profile_info" | null,
    "category": "general" | "owner_profile" | "agent_self" | "tool_feedback",
    "about_tool": "tool_name or null",
    "about_objective_type": "research/code/email/communication/etc or null",
    "what_was_wrong": "description of issue or null",
    "what_to_do_instead": "preferred approach or null",
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0-1.0
}}

Category guide:
- "owner_profile": facts about the owner (interests, timezone, role, preferences, schedule)
- "agent_self": how the agent should behave, its boundaries, role definition
- "tool_feedback": feedback about a specific tool's usage
- "general": everything else (work style, communication, general preferences)

Examples of feedback:
- "Actually, I prefer shorter emails" → preference, category=general
- "I'm a VC focused on AI startups" → profile_info, category=owner_profile
- "Don't spend money without asking me" → preference, category=agent_self
- "Don't use that API, use X instead" → correction, category=tool_feedback
- "I'm usually free after 3pm Pacific" → profile_info, category=owner_profile
- "You should always check my calendar before scheduling" → preference, category=agent_self

Return only valid JSON, no other text."""

        try:
            with track_source("learning"):
                response = self.client.messages.create(
                    model=self.fast_model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}],
                )

            response_text = response.content[0].text

            # Parse JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                data = json.loads(response_text[start:end])
            else:
                return ExtractedFeedback()

            return ExtractedFeedback(
                has_feedback=data.get("has_feedback", False),
                feedback_type=data.get("feedback_type"),
                category=data.get("category", "general"),
                about_tool=data.get("about_tool"),
                about_objective_type=data.get("about_objective_type"),
                what_was_wrong=data.get("what_was_wrong"),
                what_to_do_instead=data.get("what_to_do_instead"),
                sentiment=data.get("sentiment", "neutral"),
                confidence=data.get("confidence", 0.5),
            )

        except Exception as e:
            logger.warning("Feedback extraction error: %s", e)
            return ExtractedFeedback()


# ═══════════════════════════════════════════════════════════
# OBJECTIVE EVALUATION
# ═══════════════════════════════════════════════════════════


class ObjectiveEvaluator:
    """Evaluates completed objectives to generate learnings."""

    # Objective type inference keywords
    OBJECTIVE_TYPES = {
        "research": ["research", "find", "search", "look up", "investigate", "analyze"],
        "code": ["code", "implement", "build", "create", "develop", "fix", "debug", "program"],
        "email": ["email", "send", "reply", "respond", "message", "write to"],
        "communication": ["communicate", "tell", "inform", "notify", "update", "reach out"],
        "data": ["data", "spreadsheet", "csv", "database", "query", "export"],
        "document": ["document", "write", "draft", "create doc", "report"],
        "schedule": ["schedule", "calendar", "meeting", "appointment", "remind"],
    }

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Get instrumented LLM client for metrics tracking (supports multiple providers)."""
        if self._client is None:
            self._client = LiteLLMAnthropicAdapter()
        return self._client

    @property
    def fast_model(self) -> str:
        """Get the configured fast model for quick classification tasks."""
        return get_model_for_use_case("fast")

    async def evaluate(
        self,
        objective_id: str,
        goal: str,
        status: str,
        result: str,
        events: list[Event],
    ) -> list[Learning]:
        """
        Evaluate a completed objective to generate learnings.

        Args:
            objective_id: ID of the objective
            goal: The objective's goal text
            status: Final status (completed/failed)
            result: Result or error message
            events: Events from this objective's execution

        Returns:
            List of learnings extracted from the evaluation
        """
        # Skip very short objectives
        if len(events) < 3:
            return []

        # Format the steps taken
        steps = self._format_steps(events)
        duration = self._calculate_duration(events)
        tools_used = self._extract_tools_used(events)
        objective_type = self._infer_objective_type(goal)

        # Call LLM for evaluation
        evaluation = await self._llm_evaluate(
            goal, status, duration, steps, result, tools_used
        )

        if not evaluation.get("should_remember", False):
            return []

        learnings = []
        for item in evaluation.get("learnings", []):
            insight = item.get("insight", "")
            if not insight:
                continue

            applies_to = item.get("applies_to", "general")

            # Determine category for self-evaluation learnings
            if applies_to not in ["objective_type", "general"]:
                eval_category = "tool_feedback"
            else:
                eval_category = "agent_self"

            learning = Learning(
                id=generate_id(),
                source_type="self_evaluation",
                source_event_id=events[-1].id if events else None,
                content=insight,
                content_embedding=get_embedding(insight),
                sentiment=item.get("sentiment", "neutral"),
                confidence=evaluation.get("overall_score", 5) / 10.0,
                category=eval_category,
                tool_id=applies_to if applies_to not in ["objective_type", "general"] else None,
                objective_type=objective_type if applies_to == "objective_type" else None,
                recommendation=item.get("recommendation"),
                topic_ids=[],
                entity_ids=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            learnings.append(learning)

        return learnings

    def _format_steps(self, events: list[Event]) -> str:
        """Format events into a step-by-step description."""
        steps = []
        for i, e in enumerate(events[:20], 1):  # Limit to 20 steps
            content = e.content[:200] if len(e.content) > 200 else e.content
            if e.tool_id:
                steps.append(f"{i}. Used tool '{e.tool_id}': {content}")
            elif e.event_type == "message":
                direction = "Received" if e.direction == "inbound" else "Sent"
                steps.append(f"{i}. {direction} message: {content}")
            else:
                steps.append(f"{i}. {e.event_type}: {content}")
        return "\n".join(steps)

    def _calculate_duration(self, events: list[Event]) -> str:
        """Calculate duration from first to last event."""
        if len(events) < 2:
            return "unknown"

        first = events[0].timestamp
        last = events[-1].timestamp

        if first and last:
            delta = last - first
            minutes = delta.total_seconds() / 60
            if minutes < 1:
                return f"{int(delta.total_seconds())} seconds"
            elif minutes < 60:
                return f"{int(minutes)} minutes"
            else:
                return f"{int(minutes / 60)} hours"

        return "unknown"

    def _extract_tools_used(self, events: list[Event]) -> list[str]:
        """Extract unique tools used in the objective."""
        tools = set()
        for e in events:
            if e.tool_id:
                tools.add(e.tool_id)
        return list(tools)

    def _infer_objective_type(self, goal: str) -> str:
        """Infer objective type from goal text."""
        goal_lower = goal.lower()

        for obj_type, keywords in self.OBJECTIVE_TYPES.items():
            for keyword in keywords:
                if keyword in goal_lower:
                    return obj_type

        return "general"

    async def _llm_evaluate(
        self,
        goal: str,
        status: str,
        duration: str,
        steps: str,
        result: str,
        tools_used: list[str],
    ) -> dict:
        """Call LLM to evaluate the objective."""
        prompt = f"""Evaluate this completed AI objective for learnings.

OBJECTIVE: {goal}
STATUS: {status}
DURATION: {duration}
TOOLS USED: {', '.join(tools_used) if tools_used else 'none'}

STEPS TAKEN:
{steps}

RESULT:
{result[:500] if result else 'No result recorded'}

Consider:
1. Was the approach efficient?
2. Were any tools used incorrectly or inefficiently?
3. What could be done better next time?
4. Were there any errors, retries, or wasted steps?

Return JSON:
{{
    "overall_score": 1-10,
    "learnings": [
        {{
            "insight": "what we learned",
            "applies_to": "tool_name" | "objective_type" | "general",
            "recommendation": "what to do next time",
            "sentiment": "positive" | "negative" | "neutral"
        }}
    ],
    "tools_used_well": ["tool1"],
    "tools_used_poorly": ["tool2"],
    "should_remember": true/false
}}

Only include meaningful learnings. Set should_remember=false if this was straightforward with nothing notable.
Return only valid JSON, no other text."""

        try:
            with track_source("learning"):
                response = self.client.messages.create(
                    model=self.fast_model,
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}],
                )

            response_text = response.content[0].text

            # Parse JSON from response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response_text[start:end])

            return {"should_remember": False}

        except Exception as e:
            logger.warning("Objective evaluation error: %s", e)
            return {"should_remember": False}


# ═══════════════════════════════════════════════════════════
# LEARNING RETRIEVAL
# ═══════════════════════════════════════════════════════════


class LearningRetriever:
    """Retrieves relevant learnings for context assembly."""

    def __init__(self, store):
        self.store = store

    def get_for_tool(self, tool_name: str, limit: int = 3) -> list[Learning]:
        """
        Get learnings specific to a tool.

        Prioritizes negative sentiment (corrections) over positive.
        """
        return self.store.get_learnings_for_tool(tool_name, limit=limit)

    def get_for_objective(self, goal: str, limit: int = 5) -> list[Learning]:
        """
        Get learnings relevant to an objective via vector search.

        Searches by goal text similarity.
        """
        embedding = get_embedding(goal)
        return self.store.search_learnings(embedding=embedding, limit=limit)

    def get_for_objective_type(
        self, objective_type: str, limit: int = 3
    ) -> list[Learning]:
        """Get learnings for a specific objective type."""
        return self.store.find_learnings(
            objective_type=objective_type,
            limit=limit,
        )

    def get_user_preferences(self) -> str:
        """
        Get summarized user preferences.

        Returns the summary from the user_preferences summary node.
        """
        node = self.store.get_summary_node("user_preferences")
        return node.summary if node else ""

    def get_recent_learnings(self, limit: int = 5) -> list[Learning]:
        """Get most recent learnings regardless of type."""
        return self.store.find_learnings(limit=limit)

    def get_by_category(self, category: str, limit: int = 5) -> list[Learning]:
        """Get learnings for a specific category (owner_profile, agent_self, etc.)."""
        return self.store.find_learnings(category=category, limit=limit)

    def format_for_context(self, learnings: list[Learning]) -> list[dict]:
        """Format learnings for inclusion in AssembledContext."""
        result = []
        for learning in learnings:
            entry = {
                "learning": learning.content,
                "type": "tool" if learning.tool_id else learning.category,
            }
            if learning.tool_id:
                entry["tool"] = learning.tool_id
            if learning.recommendation:
                entry["recommendation"] = learning.recommendation

            # Record that this learning was applied
            self.store.record_learning_applied(learning.id)

            result.append(entry)

        return result


# ═══════════════════════════════════════════════════════════
# PREFERENCE SUMMARIZATION
# ═══════════════════════════════════════════════════════════


class PreferenceSummarizer:
    """Summarizes learnings into user preferences."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        """Get instrumented LLM client for metrics tracking (supports multiple providers)."""
        if self._client is None:
            self._client = LiteLLMAnthropicAdapter()
        return self._client

    @property
    def model(self) -> str:
        """Get the configured model for memory operations."""
        return get_model_for_use_case("memory")

    async def refresh_preferences(self, store) -> str:
        """
        Regenerate the user preferences summary from all learnings.

        Args:
            store: MemoryStore instance

        Returns:
            New preferences summary text
        """
        # Get all learnings, prioritize negatives and recent
        learnings = store.get_all_learnings(limit=50)

        if not learnings:
            return "No user preferences recorded yet."

        # Group learnings by category
        grouped = self._group_learnings(learnings)

        # Format for prompt
        learnings_text = self._format_learnings(grouped)

        # Generate summary
        return await self._llm_summarize(learnings_text)

    def _group_learnings(self, learnings: list[Learning]) -> dict:
        """Group learnings by category for summarization.

        Uses the Learning.category field first, then falls back to
        heuristic grouping for uncategorized learnings.
        """
        groups = {
            "owner_profile": [],
            "agent_self": [],
            "tool_preferences": [],
            "communication_style": [],
            "work_approach": [],
            "general": [],
        }

        for learning in learnings:
            # Use explicit category first
            if learning.category == "owner_profile":
                groups["owner_profile"].append(learning)
            elif learning.category == "agent_self":
                groups["agent_self"].append(learning)
            elif learning.category == "tool_feedback" or learning.tool_id:
                groups["tool_preferences"].append(learning)
            # Fall back to heuristic grouping
            elif learning.objective_type in ["email", "communication"]:
                groups["communication_style"].append(learning)
            elif learning.objective_type in ["research", "code", "data"]:
                groups["work_approach"].append(learning)
            else:
                groups["general"].append(learning)

        return groups

    def _format_learnings(self, grouped: dict) -> str:
        """Format grouped learnings for the summary prompt."""
        sections = []

        for category, learnings in grouped.items():
            if not learnings:
                continue

            category_name = category.replace("_", " ").title()
            items = []
            for l in learnings[:10]:  # Limit per category
                sentiment_marker = "[+]" if l.sentiment == "positive" else "[-]" if l.sentiment == "negative" else "[~]"
                items.append(f"  {sentiment_marker} {l.content}")
                if l.recommendation:
                    items.append(f"      → {l.recommendation}")

            sections.append(f"{category_name}:\n" + "\n".join(items))

        return "\n\n".join(sections)

    async def _llm_summarize(self, learnings_text: str) -> str:
        """Call LLM to generate preferences summary."""
        prompt = f"""Summarize these learnings into concise, actionable context.

LEARNINGS:
{learnings_text}

Create a brief summary (8-15 bullet points) organized into these sections:

**Owner Profile** (who they are, interests, schedule, location, role):
- Only include facts explicitly stated or clearly inferred
- Example: "Based in San Francisco, PST timezone"
- Example: "Focused on AI startups and venture capital"

**Agent Rules** (how the agent should behave, boundaries, role):
- Example: "Always ask before spending any money"
- Example: "Check calendar before scheduling meetings"

**Communication Preferences** (tone, length, format, style):
- Example: "Keep emails under 3 paragraphs"

**Tool & Work Preferences** (how to approach tasks):
- Example: "Prefer Python over JavaScript for scripts"

Only include sections that have actual learnings. Be specific and practical.
Format as actionable instructions the AI assistant should follow.

Summary:"""

        try:
            with track_source("learning"):
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=600,
                    messages=[{"role": "user", "content": prompt}],
                )

            return response.content[0].text.strip()

        except Exception as e:
            logger.warning("Preference summarization error: %s", e)
            return "Unable to generate preferences summary."


# ═══════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════


def ensure_user_preferences_node(store):
    """
    Ensure the user_preferences summary node exists.

    Call this during memory system initialization.
    """
    existing = store.get_summary_node("user_preferences")
    if existing is None:
        store.create_summary_node(
            node_type="preferences",
            key="user_preferences",
            label="User Preferences",
            parent_key="root",
            summary="No user preferences recorded yet.",
        )
