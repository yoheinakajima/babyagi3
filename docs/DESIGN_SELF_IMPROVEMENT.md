# Self-Improvement System Design

## Overview

This document describes an elegant self-improvement system that enables the agent to:
1. **Capture feedback** from user messages about prior task performance
2. **Self-evaluate** long-running objectives to generate internal feedback
3. **Store learnings** tied to objectives, tools, and topics
4. **Retrieve relevant learnings** via vector similarity when running similar tasks
5. **Summarize learnings** into user preferences that are always in context

The design follows existing patterns in the codebase, treating learnings as first-class graph citizens alongside Entities, Edges, and Topics.

---

## Design Principles

1. **Everything is an Event** - Learnings originate from events (messages, objective completions)
2. **Graph Integration** - Learnings connect to entities (tools, topics, people) via edges
3. **Pre-computed Summaries** - User preferences are summarized, not computed at context time
4. **Vector Search** - Relevant learnings retrieved by embedding similarity
5. **Minimal Code** - Leverage existing extraction pipeline patterns

---

## Data Model

### Learning (New Model)

```python
@dataclass
class Learning:
    """A piece of learned knowledge from feedback or self-evaluation."""

    id: str

    # Source
    source_type: str  # "user_feedback", "self_evaluation", "observation"
    source_event_id: str  # Event that triggered this learning

    # Content
    content: str  # The actual learning/insight
    content_embedding: list[float] | None  # For vector search

    # Classification
    sentiment: str  # "positive", "negative", "neutral"
    confidence: float  # 0-1, how confident we are in this learning

    # Associations (what this learning is about)
    tool_id: str | None  # If about a specific tool
    topic_ids: list[str]  # Related topics
    objective_type: str | None  # Type of objective (e.g., "research", "code", "email")
    entity_ids: list[str]  # Related entities (people, orgs)

    # Actionable insight
    applies_when: str | None  # Condition when this learning applies
    recommendation: str | None  # What to do differently

    # Stats
    times_applied: int = 0  # How often this learning was used
    last_applied_at: datetime | None = None

    # Provenance
    created_at: datetime
    updated_at: datetime
```

### Database Schema Addition

```sql
CREATE TABLE IF NOT EXISTS learnings (
    id TEXT PRIMARY KEY,

    -- Source
    source_type TEXT NOT NULL,  -- "user_feedback", "self_evaluation", "observation"
    source_event_id TEXT NOT NULL,

    -- Content
    content TEXT NOT NULL,
    content_embedding BLOB,

    -- Classification
    sentiment TEXT NOT NULL,  -- "positive", "negative", "neutral"
    confidence REAL DEFAULT 0.5,

    -- Associations
    tool_id TEXT,
    topic_ids TEXT,  -- JSON array
    objective_type TEXT,
    entity_ids TEXT,  -- JSON array

    -- Actionable insight
    applies_when TEXT,
    recommendation TEXT,

    -- Stats
    times_applied INTEGER DEFAULT 0,
    last_applied_at TEXT,

    -- Timestamps
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (source_event_id) REFERENCES events(id),
    FOREIGN KEY (tool_id) REFERENCES tool_definitions(name)
);

CREATE INDEX idx_learnings_tool ON learnings(tool_id);
CREATE INDEX idx_learnings_objective_type ON learnings(objective_type);
CREATE INDEX idx_learnings_sentiment ON learnings(sentiment);
CREATE INDEX idx_learnings_source_type ON learnings(source_type);
```

### Summary Node Extension

Add new node types to the summary tree:

```python
# New node_type values:
# - "learning:tool:{tool_name}" - Learnings about a specific tool
# - "learning:objective_type:{type}" - Learnings about an objective type
# - "learning:topic:{topic_id}" - Learnings about a topic
# - "user_preferences" - Master summary of all learnings (always in context)
```

---

## Feedback Extraction

### From User Messages

When a message arrives, analyze if it contains feedback about prior work:

```python
class FeedbackExtractor:
    """Extracts feedback from user messages."""

    FEEDBACK_PROMPT = """Analyze this message for feedback about prior AI work.

MESSAGE:
{content}

RECENT CONTEXT (last 5 AI actions):
{recent_actions}

Return JSON:
{
    "has_feedback": true/false,
    "feedback_type": "correction" | "praise" | "preference" | "complaint" | null,
    "about_tool": "tool_name or null",
    "about_objective_type": "research/code/email/etc or null",
    "what_was_wrong": "description or null",
    "what_to_do_instead": "description or null",
    "sentiment": "positive" | "negative" | "neutral",
    "confidence": 0.0-1.0
}

Examples of feedback:
- "Actually, I prefer shorter emails" → preference about email style
- "That's not how I want the code formatted" → correction about code tool
- "Great research, exactly what I needed" → praise about research
- "Don't use that API, use X instead" → correction about tool usage
"""

    async def extract(self, event: Event, recent_events: list[Event]) -> Learning | None:
        # Build context from recent AI actions
        recent_actions = self._format_recent_actions(recent_events)

        # Call LLM to detect feedback
        result = await self._llm_extract(event.content, recent_actions)

        if not result.get("has_feedback"):
            return None

        # Create learning
        return Learning(
            id=generate_id(),
            source_type="user_feedback",
            source_event_id=event.id,
            content=self._format_learning(result),
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            tool_id=result.get("about_tool"),
            objective_type=result.get("about_objective_type"),
            applies_when=result.get("what_was_wrong"),
            recommendation=result.get("what_to_do_instead"),
            topic_ids=[],
            entity_ids=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
```

### Integration Hook

Add to `setup_memory_hooks`:

```python
@agent.on("message_received")
async def on_message_received(event):
    """Extract feedback from incoming messages."""
    if not event.get("is_owner", False):
        return  # Only extract feedback from owner messages

    # Get recent AI actions for context
    recent = memory.store.get_recent_events(
        limit=5,
        direction="outbound",
        event_types=["message", "tool_result"]
    )

    extractor = FeedbackExtractor()
    learning = await extractor.extract(event, recent)

    if learning:
        memory.store.create_learning(learning)
        # Trigger preference summary update
        memory.store.increment_staleness("user_preferences")
```

---

## Self-Evaluation for Objectives

### Objective Evaluator

When an objective completes, evaluate performance:

```python
class ObjectiveEvaluator:
    """Evaluates completed objectives to generate learnings."""

    EVALUATION_PROMPT = """Evaluate this completed objective.

OBJECTIVE: {goal}
STATUS: {status}
DURATION: {duration}
STEPS TAKEN: {steps}
RESULT: {result}

Consider:
1. Was the approach efficient?
2. Were any tools used incorrectly?
3. What could be done better next time?
4. Were there any errors or retries?

Return JSON:
{
    "overall_score": 1-10,
    "learnings": [
        {
            "insight": "what we learned",
            "applies_to": "tool_name" | "objective_type" | "general",
            "recommendation": "what to do next time",
            "sentiment": "positive" | "negative" | "neutral"
        }
    ],
    "tools_used_well": ["tool1", "tool2"],
    "tools_used_poorly": ["tool3"],
    "should_remember": true/false
}
"""

    async def evaluate(
        self,
        objective_id: str,
        goal: str,
        status: str,
        result: str,
        events: list[Event]
    ) -> list[Learning]:
        # Format the steps taken
        steps = self._format_steps(events)
        duration = self._calculate_duration(events)

        # Call LLM for evaluation
        evaluation = await self._llm_evaluate(goal, status, duration, steps, result)

        if not evaluation.get("should_remember"):
            return []

        learnings = []
        for item in evaluation.get("learnings", []):
            learning = Learning(
                id=generate_id(),
                source_type="self_evaluation",
                source_event_id=events[-1].id if events else None,
                content=item["insight"],
                sentiment=item["sentiment"],
                confidence=evaluation["overall_score"] / 10.0,
                tool_id=item["applies_to"] if item["applies_to"] not in ["objective_type", "general"] else None,
                objective_type=self._infer_objective_type(goal),
                recommendation=item.get("recommendation"),
                topic_ids=[],
                entity_ids=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            learnings.append(learning)

        return learnings
```

### Integration Hook

```python
@agent.on("objective_end")
async def on_objective_end(event):
    """Evaluate completed objectives for learnings."""
    if event["status"] not in ["completed", "failed"]:
        return

    # Get events from this objective
    objective_events = memory.store.get_recent_events(
        limit=50,
        task_id=event["id"]
    )

    evaluator = ObjectiveEvaluator()
    learnings = await evaluator.evaluate(
        objective_id=event["id"],
        goal=event["goal"],
        status=event["status"],
        result=event.get("result", ""),
        events=objective_events
    )

    for learning in learnings:
        memory.store.create_learning(learning)

    # Trigger preference summary update
    if learnings:
        memory.store.increment_staleness("user_preferences")
```

---

## Learning Retrieval

### Vector-Based Retrieval

When assembling context, retrieve relevant learnings:

```python
class LearningRetriever:
    """Retrieves relevant learnings for context."""

    def get_for_tool(self, tool_name: str, limit: int = 3) -> list[Learning]:
        """Get learnings specific to a tool."""
        return self.store.find_learnings(
            tool_id=tool_name,
            sentiment="negative",  # Prioritize corrections
            limit=limit
        )

    def get_for_objective(self, goal: str, limit: int = 5) -> list[Learning]:
        """Get learnings relevant to an objective via vector search."""
        embedding = get_embedding(goal)
        return self.store.search_learnings(
            embedding=embedding,
            limit=limit
        )

    def get_for_topic(self, topic_id: str, limit: int = 3) -> list[Learning]:
        """Get learnings related to a topic."""
        return self.store.find_learnings(
            topic_id=topic_id,
            limit=limit
        )

    def get_user_preferences(self) -> str:
        """Get summarized user preferences."""
        node = self.store.get_summary_node("user_preferences")
        return node.summary if node else ""
```

### Store Methods

```python
# In MemoryStore class:

def search_learnings(
    self,
    embedding: list[float],
    tool_id: str | None = None,
    objective_type: str | None = None,
    limit: int = 10
) -> list[Learning]:
    """Search learnings by vector similarity."""
    # Use cosine similarity on content_embedding
    # Filter by tool_id or objective_type if provided
    pass

def find_learnings(
    self,
    tool_id: str | None = None,
    topic_id: str | None = None,
    objective_type: str | None = None,
    sentiment: str | None = None,
    limit: int = 10
) -> list[Learning]:
    """Find learnings by filters."""
    pass

def create_learning(self, learning: Learning) -> Learning:
    """Create a new learning."""
    # Generate embedding
    learning.content_embedding = get_embedding(learning.content)
    # Insert into database
    pass
```

---

## Context Assembly Integration

### Extended AssembledContext

```python
@dataclass
class AssembledContext:
    # ... existing fields ...

    # NEW: Learnings context
    learnings: list[dict] = field(default_factory=list)
    user_preferences: str = ""
```

### Context Builder Updates

```python
def assemble_context(...) -> AssembledContext:
    # ... existing code ...

    # Always include user preferences
    prefs_node = store.get_summary_node("user_preferences")
    if prefs_node and prefs_node.summary:
        ctx.user_preferences = prefs_node.summary

    # Include relevant learnings based on event
    if event:
        retriever = LearningRetriever(store)

        # If using a tool, get tool-specific learnings
        if event.tool_id:
            tool_learnings = retriever.get_for_tool(event.tool_id)
            ctx.learnings.extend([
                {"type": "tool", "tool": event.tool_id, "learning": l.content}
                for l in tool_learnings
            ])

        # If starting an objective, get similar objective learnings
        if event.event_type == "objective_start":
            obj_learnings = retriever.get_for_objective(event.content)
            ctx.learnings.extend([
                {"type": "objective", "learning": l.content, "recommendation": l.recommendation}
                for l in obj_learnings
            ])

    return ctx
```

### Prompt Formatting

```python
def to_prompt(self) -> str:
    sections = []

    # ... existing sections ...

    # User Preferences (always shown)
    if self.user_preferences:
        sections.append("\n## User Preferences")
        sections.append(self.user_preferences)

    # Relevant Learnings (context-specific)
    if self.learnings:
        sections.append("\n## Relevant Learnings")
        for learning in self.learnings:
            if learning["type"] == "tool":
                sections.append(f"- **{learning['tool']}**: {learning['learning']}")
            else:
                sections.append(f"- {learning['learning']}")
                if learning.get("recommendation"):
                    sections.append(f"  → {learning['recommendation']}")

    return "\n".join(sections)
```

---

## Preference Summarization

### Summary Node for User Preferences

The `user_preferences` summary node is a special leaf node that aggregates all learnings:

```python
class PreferenceSummarizer:
    """Summarizes learnings into user preferences."""

    SUMMARY_PROMPT = """Summarize these learnings into user preferences.

LEARNINGS (sorted by recency):
{learnings_text}

Create a concise summary (5-10 bullet points) that captures:
1. Communication preferences (tone, length, format)
2. Tool usage preferences (which tools, how to use them)
3. Work style preferences (thoroughness, speed, approach)
4. Domain-specific preferences

Format as actionable instructions I can follow.
Example: "- Prefer concise emails under 3 paragraphs"

Summary:"""

    async def refresh_preferences(self, store) -> str:
        """Regenerate the user preferences summary."""
        # Get all learnings, prioritize negatives (corrections) and recent
        learnings = store.find_learnings(
            limit=50,
            order_by="created_at DESC"
        )

        # Group by category
        grouped = self._group_learnings(learnings)

        # Format for prompt
        learnings_text = self._format_learnings(grouped)

        # Generate summary
        response = await self._llm_summarize(learnings_text)

        return response.strip()
```

### Summary Manager Extension

```python
# In SummaryManager._refresh_leaf_node:

async def _refresh_leaf_node(self, node: SummaryNode):
    if node.key == "user_preferences":
        # Special handling for user preferences
        summarizer = PreferenceSummarizer()
        node.summary = await summarizer.refresh_preferences(self.store)
        node.summary_embedding = get_embedding(node.summary)
        node.summary_updated_at = datetime.now()
        node.events_since_update = 0
        self.store.update_summary_node(node)
        return

    # ... existing leaf node handling ...
```

---

## Integration Points

### 1. Message Processing

```
User Message
    ↓
log_event() → Event
    ↓
FeedbackExtractor.extract()
    ↓
If feedback found → create_learning()
    ↓
Increment staleness("user_preferences")
```

### 2. Objective Execution

```
Objective Starts
    ↓
LearningRetriever.get_for_objective()
    ↓
Add to context.learnings
    ↓
... objective runs ...
    ↓
Objective Ends
    ↓
ObjectiveEvaluator.evaluate()
    ↓
create_learning() for each insight
    ↓
Increment staleness("user_preferences")
```

### 3. Tool Usage

```
Tool Call
    ↓
LearningRetriever.get_for_tool()
    ↓
Add to context.learnings
    ↓
Tool executes
    ↓
(Success/Error tracked in tool_definitions)
```

### 4. Context Assembly

```
assemble_context()
    ↓
get_summary_node("user_preferences") → Always in context
    ↓
If tool_id → get_for_tool() → Add learnings
    ↓
If objective → get_for_objective() → Add learnings
    ↓
Format into prompt
```

---

## File Changes Summary

| File | Changes |
|------|---------|
| `memory/models.py` | Add `Learning` dataclass |
| `memory/store.py` | Add `learnings` table, CRUD methods, search methods |
| `memory/extraction.py` | Add `FeedbackExtractor` class |
| `memory/learning.py` | **NEW**: `LearningRetriever`, `ObjectiveEvaluator`, `PreferenceSummarizer` |
| `memory/context.py` | Add `learnings` and `user_preferences` to `AssembledContext` |
| `memory/summaries.py` | Handle `user_preferences` node type |
| `memory/integration.py` | Add hooks for feedback extraction and objective evaluation |

---

## Token Budget Considerations

```python
# In ContextConfig:
token_budgets: dict = field(
    default_factory=lambda: {
        # ... existing ...
        "user_preferences": 300,  # Always included
        "learnings": 200,  # Context-specific learnings
    }
)
```

---

## Example Flow

### User Provides Feedback

```
User: "Actually, when you send emails, please keep them under 3 paragraphs. That last one was too long."

→ FeedbackExtractor detects:
  - has_feedback: true
  - feedback_type: "preference"
  - about_tool: "send_message"
  - what_was_wrong: "Email was too long"
  - what_to_do_instead: "Keep emails under 3 paragraphs"
  - sentiment: "negative"

→ Learning created:
  - content: "User prefers emails under 3 paragraphs"
  - tool_id: "send_message"
  - recommendation: "Keep email body concise, max 3 paragraphs"

→ user_preferences summary updated:
  "- Keep emails under 3 paragraphs..."
```

### Objective Self-Evaluation

```
Objective: "Research competitors in the CRM space"
Status: completed
Duration: 45 minutes

→ ObjectiveEvaluator analyzes:
  - Steps: web_search (3x), browse_page (5x), memory.store
  - Retries: 2 (rate limited)

→ Learnings generated:
  1. "For competitor research, start with industry reports before individual companies"
  2. "Space out web searches to avoid rate limits"

→ user_preferences updated to include research approach preferences
```

---

## Benefits

1. **Continuous Improvement** - Agent learns from every interaction
2. **Personalization** - Preferences adapt to individual users
3. **Efficiency** - Pre-computed summaries, fast retrieval
4. **Transparency** - Learnings are inspectable entities
5. **Graceful** - Works without breaking existing functionality
6. **Graph-Native** - Follows existing entity/edge/topic patterns

---

## Future Extensions

1. **Learning Decay** - Reduce confidence over time without reinforcement
2. **Learning Conflicts** - Detect and resolve contradictory learnings
3. **Multi-User** - Per-user preference profiles
4. **Learning Export** - Share learnings across agent instances
5. **Active Learning** - Ask clarifying questions about ambiguous feedback
