"""
Self-improvement system for the memory system.

Extracts learnings from feedback, self-evaluation, and tool error patterns.
Retrieves relevant learnings with age-based decay weighting.
Resolves contradictions by superseding (never deleting) old learnings.
Re-boosts decay when a learning is surfaced and proves useful.
"""

import json
import logging
import math
from datetime import datetime
from uuid import uuid4

logger = logging.getLogger(__name__)

from metrics import LiteLLMAnthropicAdapter, track_source, get_model_for_use_case
from .embeddings import get_embedding
from .models import Event, ExtractedFeedback, Learning


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


# Half-life in days for learning decay.  A learning's weight drops to 50%
# after this many days — unless it gets re-boosted by being surfaced in
# context, which resets updated_at and the decay clock.
DECAY_HALF_LIFE_DAYS = 14

# Similarity threshold for contradiction detection.  Two learnings about
# the same tool/topic with cosine similarity above this are considered
# the same topic — the old one gets superseded (kept, but filtered from
# retrieval).
CONTRADICTION_SIMILARITY_THRESHOLD = 0.82


def decay_weight(learning: Learning, now: datetime | None = None) -> float:
    """Calculate age-based decay weight for a learning (0-1).

    Uses updated_at (not created_at) so that learnings which are
    re-boosted by being surfaced in context stay fresh.
    """
    now = now or datetime.now()
    anchor = learning.updated_at or learning.created_at
    if not anchor:
        return 0.5
    age_days = max((now - anchor).total_seconds() / 86400, 0)
    return math.exp(-0.693 * age_days / DECAY_HALF_LIFE_DAYS)  # ln(2) ≈ 0.693


# ═══════════════════════════════════════════════════════════
# CONTRADICTION RESOLUTION
# ═══════════════════════════════════════════════════════════


def resolve_contradictions(new_learning: Learning, store) -> list[str]:
    """Mark existing learnings that the new one supersedes.

    Two learnings contradict when they are about the same topic
    (high embedding similarity + same tool_id) but express opposite
    sentiment.  The old one is *not* deleted — it stays in the DB
    with superseded_by pointing to the new one, preserving history
    for flip-flop tracking.

    Returns list of superseded learning IDs.
    """
    if not new_learning.content_embedding:
        return []

    # Search for similar existing learnings (already excludes superseded)
    similar = store.search_learnings(
        new_learning.content_embedding,
        tool_id=new_learning.tool_id,
        limit=5,
    )

    superseded = []
    for existing in similar:
        if existing.id == new_learning.id:
            continue
        if not existing.content_embedding:
            continue

        similarity = store._cosine_similarity(
            new_learning.content_embedding, existing.content_embedding
        )
        if similarity < CONTRADICTION_SIMILARITY_THRESHOLD:
            continue

        # Same topic area — check for contradiction
        same_scope = new_learning.tool_id == existing.tool_id
        opposite_sentiment = (
            new_learning.sentiment != existing.sentiment
            and new_learning.sentiment != "neutral"
            and existing.sentiment != "neutral"
        )

        if same_scope and opposite_sentiment:
            store.supersede_learning(existing.id, new_learning.id)
            superseded.append(existing.id)
            logger.debug(
                "Superseded learning %s (sim=%.2f, %s→%s)",
                existing.id[:8],
                similarity,
                existing.sentiment,
                new_learning.sentiment,
            )

    return superseded


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

        learning = Learning(
            id=generate_id(),
            source_type="user_feedback",
            source_event_id=event.id,
            content=content,
            content_embedding=get_embedding(content),
            sentiment=feedback.sentiment,
            confidence=feedback.confidence,
            tool_id=feedback.about_tool,
            objective_type=feedback.about_objective_type,
            recommendation=feedback.what_to_do_instead,
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
- General working style or behavior

Return JSON:
{{
    "has_feedback": true/false,
    "feedback_type": "correction" | "praise" | "preference" | "complaint" | "profile_info" | null,
    "about_tool": "tool_name or null",
    "about_objective_type": "research/code/email/communication/etc or null",
    "what_was_wrong": "description of issue or null",
    "what_to_do_instead": "preferred approach or null",
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0-1.0
}}

Examples of feedback:
- "Actually, I prefer shorter emails" → preference
- "Don't use that API, use X instead" → correction, about_tool
- "I'm usually free after 3pm Pacific" → profile_info
- "You should always check my calendar before scheduling" → preference

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

            # Determine tool_id from applies_to
            tool_id = applies_to if applies_to not in ["objective_type", "general"] else None

            learning = Learning(
                id=generate_id(),
                source_type="self_evaluation",
                source_event_id=events[-1].id if events else None,
                content=insight,
                content_embedding=get_embedding(insight),
                sentiment=item.get("sentiment", "neutral"),
                confidence=evaluation.get("overall_score", 5) / 10.0,
                tool_id=tool_id,
                objective_type=objective_type if applies_to == "objective_type" else None,
                recommendation=item.get("recommendation"),
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
# TOOL ERROR ANALYSIS → AUTO-FIX TASKS
# ═══════════════════════════════════════════════════════════


class ToolErrorAnalyzer:
    """Converts tool error statistics into learnings and fix tasks.

    When a tool is unhealthy, creates both:
    - A learning (so the agent knows about the problem in context)
    - A fix task (so the agent proactively works on repairing it)
    """

    def analyze_and_fix(self, store) -> list[Learning]:
        """Create learnings + fix tasks from unhealthy/problematic tool stats.

        Returns the learnings created (tasks are created as a side-effect).
        """
        try:
            unhealthy = store.get_unhealthy_tools()
            problematic = store.get_problematic_tools(error_threshold=3)
        except Exception as e:
            logger.warning("Could not fetch tool stats for learning: %s", e)
            return []

        # Deduplicate by name
        seen = set()
        tools = []
        for t in unhealthy + problematic:
            if t.name not in seen:
                seen.add(t.name)
                tools.append(t)

        learnings = []
        for tool in tools:
            # Skip if we already have a recent active learning about this tool
            existing = store.find_learnings(
                tool_id=tool.name, source_type="tool_error_pattern", limit=1
            )
            if existing:
                age_days = (datetime.now() - existing[0].created_at).total_seconds() / 86400
                if age_days < 7:
                    continue  # Recent enough, skip

            content = (
                f"Tool '{tool.name}' has a {tool.success_rate:.0f}% success rate "
                f"({tool.error_count} errors out of {tool.usage_count} uses)."
            )
            if tool.last_error:
                content += f" Last error: {tool.last_error[:200]}"

            recommendation = (
                f"Investigate and fix '{tool.name}'. "
                f"Check parameters, dependencies, and error patterns."
            )

            learning = Learning(
                id=generate_id(),
                source_type="tool_error_pattern",
                content=content,
                content_embedding=get_embedding(content),
                sentiment="negative",
                confidence=min(tool.usage_count / 10.0, 1.0),
                tool_id=tool.name,
                recommendation=recommendation,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            learnings.append(learning)

            # Create a fix task so the agent acts on it
            try:
                task_title = f"Fix tool '{tool.name}' ({tool.success_rate:.0f}% success rate)"
                task_desc = (
                    f"Tool '{tool.name}' is failing frequently.\n"
                    f"Success rate: {tool.success_rate:.0f}% "
                    f"({tool.error_count} errors / {tool.usage_count} uses)\n"
                )
                if tool.last_error:
                    task_desc += f"Last error: {tool.last_error[:500]}\n"
                task_desc += (
                    "\nInvestigate the error pattern and fix the root cause. "
                    "Check parameter validation, API changes, or dependency issues."
                )

                store.create_task(
                    title=task_title,
                    description=task_desc,
                    type_raw="tool_fix",
                    type_cluster="maintenance",
                )
                logger.debug("Created fix task for tool '%s'", tool.name)
            except Exception as e:
                logger.debug("Could not create fix task for '%s': %s", tool.name, e)

        return learnings


# ═══════════════════════════════════════════════════════════
# LEARNING RETRIEVAL (with decay + re-boost)
# ═══════════════════════════════════════════════════════════


class LearningRetriever:
    """Retrieves relevant learnings for context assembly.

    All retrieval methods weight results by age-based decay so that
    recent learnings dominate while old ones fade.  When a learning
    is surfaced, its updated_at is touched to re-boost its decay
    weight — frequently useful learnings stay alive indefinitely.
    """

    def __init__(self, store):
        self.store = store

    def get_for_tool(self, tool_name: str, limit: int = 3) -> list[Learning]:
        """Get learnings specific to a tool, weighted by decay."""
        learnings = self.store.get_learnings_for_tool(tool_name, limit=limit * 2)
        ranked = self._rank_by_decay(learnings)[:limit]
        self._touch_all(ranked)
        return ranked

    def get_for_objective(self, goal: str, limit: int = 5) -> list[Learning]:
        """Get learnings relevant to an objective via vector search + decay."""
        embedding = get_embedding(goal)
        learnings = self.store.search_learnings(embedding=embedding, limit=limit * 2)
        ranked = self._rank_by_decay(learnings)[:limit]
        self._touch_all(ranked)
        return ranked

    def get_for_objective_type(
        self, objective_type: str, limit: int = 3
    ) -> list[Learning]:
        """Get learnings for a specific objective type."""
        learnings = self.store.find_learnings(
            objective_type=objective_type,
            limit=limit * 2,
        )
        ranked = self._rank_by_decay(learnings)[:limit]
        self._touch_all(ranked)
        return ranked

    def get_user_preferences(self) -> str:
        """Get summarized user preferences."""
        node = self.store.get_summary_node("user_preferences")
        return node.summary if node else ""

    def get_recent_learnings(self, limit: int = 5) -> list[Learning]:
        """Get most recent learnings regardless of type."""
        return self.store.find_learnings(limit=limit)

    def format_for_context(self, learnings: list[Learning]) -> list[dict]:
        """Format learnings for inclusion in AssembledContext."""
        result = []
        for learning in learnings:
            entry = {
                "learning": learning.content,
                "type": "tool" if learning.tool_id else "general",
            }
            if learning.tool_id:
                entry["tool"] = learning.tool_id
            if learning.recommendation:
                entry["recommendation"] = learning.recommendation

            result.append(entry)

        return result

    def _rank_by_decay(self, learnings: list[Learning]) -> list[Learning]:
        """Re-rank learnings by confidence * decay_weight."""
        now = datetime.now()
        scored = [
            (l.confidence * decay_weight(l, now), l)
            for l in learnings
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [l for _, l in scored]

    def _touch_all(self, learnings: list[Learning]):
        """Re-boost decay for learnings that were just surfaced."""
        for l in learnings:
            try:
                self.store.touch_learning(l.id)
            except Exception:
                pass  # Non-critical — don't break retrieval


# ═══════════════════════════════════════════════════════════
# PREFERENCE SUMMARIZATION
# ═══════════════════════════════════════════════════════════


class PreferenceSummarizer:
    """Summarizes active learnings into a user preferences snapshot."""

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
        Regenerate the user preferences summary from active learnings.

        No learnings are deleted — decay and supersession handle relevance.

        Returns:
            New preferences summary text
        """
        learnings = store.get_all_learnings(limit=50)

        if not learnings:
            return "No user preferences recorded yet."

        learnings_text = self._format_learnings(learnings)
        return await self._llm_summarize(learnings_text)

    def _format_learnings(self, learnings: list[Learning]) -> str:
        """Format learnings for the summary prompt."""
        items = []
        for l in learnings:
            sentiment_marker = (
                "[+]" if l.sentiment == "positive"
                else "[-]" if l.sentiment == "negative"
                else "[~]"
            )
            line = f"  {sentiment_marker} {l.content}"
            if l.recommendation:
                line += f"\n      → {l.recommendation}"
            if l.tool_id:
                line += f"  [tool: {l.tool_id}]"
            items.append(line)

        return "\n".join(items)

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
