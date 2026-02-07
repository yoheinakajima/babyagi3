"""
Context window budget management.

Prevents context overflow by tracking and enforcing token limits across
all components sent to the LLM: system prompt, tool schemas, and thread
messages. Provides thread trimming, tool result truncation, and
graceful recovery when limits are approached or exceeded.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Conservative estimate: 1 token ≈ 4 characters.
# Overestimates slightly, which is safer than underestimating.
CHARS_PER_TOKEN = 4

# Maximum characters for a single tool result before truncation.
MAX_TOOL_RESULT_CHARS = 12_000  # ~3,000 tokens

# Threshold above which tool results get LLM summarization instead of
# brute truncation (when a summarizer is available).
SUMMARIZE_THRESHOLD_CHARS = 8_000  # ~2,000 tokens


@dataclass
class ContextBudget:
    """Centralized token budget for a single LLM API call.

    Allocates the model's context window across system prompt, tool schemas,
    and thread messages. Thread gets whatever remains after the fixed-cost
    components (system prompt + tool schemas) are accounted for.
    """

    model_limit: int = 200_000
    reserved_for_response: int = 8_096

    # Hard caps per component (tokens). These are ceilings, not allocations.
    system_prompt_max: int = 15_000
    tool_schemas_max: int = 30_000

    # Minimum thread budget — if we can't fit at least this many tokens of
    # thread, something is very wrong and we should aggressively cut tools.
    thread_min: int = 10_000

    def available_tokens(self) -> int:
        return self.model_limit - self.reserved_for_response

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(text) // CHARS_PER_TOKEN

    def estimate_messages_tokens(self, messages: list) -> int:
        total = 0
        for msg in messages:
            total += self.estimate_tokens(json.dumps(msg, default=str))
        return total

    def estimate_schemas_tokens(self, schemas: list) -> int:
        total = 0
        for s in schemas:
            total += self.estimate_tokens(json.dumps(s, default=str))
        return total

    def fits(self, system_prompt: str, tool_schemas: list, thread: list) -> bool:
        """Check whether the full payload fits within the context window."""
        sp = self.estimate_tokens(system_prompt)
        ts = self.estimate_schemas_tokens(tool_schemas)
        th = self.estimate_messages_tokens(thread)
        return (sp + ts + th) <= self.available_tokens()

    def trim_thread(self, thread: list, system_prompt: str, tool_schemas: list) -> list:
        """Trim thread to fit within budget, preserving newest messages.

        Strategy:
        1. Keep the first user message (establishes conversation context).
        2. Keep the most recent messages (the active tool loop / latest query).
        3. Drop the middle (oldest turns) and insert a marker.

        Returns a new list — does not mutate the original.
        """
        available = self.available_tokens()
        sp_tokens = self.estimate_tokens(system_prompt)
        ts_tokens = self.estimate_schemas_tokens(tool_schemas)
        thread_budget = available - sp_tokens - ts_tokens

        if thread_budget < self.thread_min:
            # System prompt + tools already too big — caller should reduce tools
            thread_budget = self.thread_min

        current_tokens = self.estimate_messages_tokens(thread)
        if current_tokens <= thread_budget:
            return thread  # Already fits

        logger.warning(
            "Thread trimming: %d tokens exceeds budget of %d (available=%d, system=%d, tools=%d). Trimming.",
            current_tokens, thread_budget, available, sp_tokens, ts_tokens,
        )

        # Binary approach: keep head and expand tail until we fit
        marker = {"role": "user", "content": "[Earlier messages trimmed to fit context window. Ask user to repeat if needed.]"}
        marker_tokens = self.estimate_tokens(json.dumps(marker, default=str))

        # Always keep the first message
        head = thread[:1] if thread else []
        head_tokens = self.estimate_messages_tokens(head)

        # Fill from the end
        remaining_budget = thread_budget - head_tokens - marker_tokens
        tail = []
        tail_tokens = 0

        for msg in reversed(thread[1:]):
            msg_tokens = self.estimate_tokens(json.dumps(msg, default=str))
            if tail_tokens + msg_tokens > remaining_budget:
                break
            tail.insert(0, msg)
            tail_tokens += msg_tokens

        # If we couldn't even fit the most recent message, force-include it
        if not tail and len(thread) > 1:
            tail = thread[-1:]

        trimmed = head + [marker] + tail
        logger.info(
            "Thread trimmed: %d messages -> %d messages (dropped %d)",
            len(thread), len(trimmed), len(thread) - len(trimmed),
        )
        return trimmed

    def emergency_trim(self, thread: list) -> list:
        """Aggressive trim for recovery after a context overflow error.

        Keeps only the first user message and the last 4 messages.
        """
        if len(thread) <= 5:
            return thread
        marker = {"role": "user", "content": "[Context was too large. Earlier messages dropped. Please repeat key details if needed.]"}
        return thread[:1] + [marker] + thread[-4:]

    def reduce_tool_schemas(self, schemas: list, max_count: int = 20) -> list:
        """Reduce tool schemas to fit budget by keeping only max_count."""
        if len(schemas) <= max_count:
            return schemas
        logger.warning("Reducing tool schemas from %d to %d", len(schemas), max_count)
        return schemas[:max_count]


def truncate_tool_result(result_json: str, tool_name: str = "") -> tuple[str, bool]:
    """Truncate a tool result if it exceeds the character limit.

    Returns:
        (possibly_truncated_json, was_truncated)
    """
    if len(result_json) <= MAX_TOOL_RESULT_CHARS:
        return result_json, False

    original_len = len(result_json)
    # Try to truncate at a reasonable JSON boundary
    truncated = result_json[:MAX_TOOL_RESULT_CHARS]

    # Wrap in a structure that tells the LLM what happened
    result = json.dumps({
        "_truncated": True,
        "_original_chars": original_len,
        "_tool": tool_name,
        "_note": "Result was too large and has been truncated. Key data shown below. If you need specific details not shown, call the tool again with more specific parameters.",
        "partial_data": truncated,
    })
    return result, True


async def summarize_tool_result(
    result_json: str,
    tool_name: str,
    client,
    model: str,
    task_context: str | None = None,
) -> str | None:
    """Summarize a large tool result using a fast LLM.

    Args:
        result_json: The raw JSON result string from the tool.
        tool_name: Name of the tool that produced the result.
        client: Anthropic client for LLM calls.
        model: Model ID for the summarization call.
        task_context: Optional description of the current task/objective goal.
            When provided, the summarizer prioritizes data relevant to this goal
            instead of applying generic heuristics that may discard critical fields.

    Returns the summarized JSON string, or None if summarization fails.
    """
    if len(result_json) <= SUMMARIZE_THRESHOLD_CHARS:
        return None

    try:
        # Truncate the input to the summarizer too — don't blow up the
        # summarization call itself. 40K chars ≈ 10K tokens input.
        summarizer_input = result_json[:40_000]

        # Build context-aware system prompt so the summarizer knows what
        # data the agent actually needs.  Without this, the summarizer
        # uses generic heuristics ("keep IDs and counts") which drops
        # domain-specific fields the agent requires (e.g. company names,
        # valuations, deal amounts) leading to hallucinated responses.
        if task_context:
            system_msg = (
                "You summarize API/tool results concisely while preserving data relevant to the current task. "
                f"Current task context: {task_context}\n"
                "Prioritize keeping fields and values that are relevant to the task above. "
                "Also keep key data (IDs, names, counts, statuses). Output valid JSON. "
                "For lists, try to include ALL items (not just the first few) with their task-relevant fields."
            )
        else:
            system_msg = "You summarize API/tool results concisely. Keep key data (IDs, names, counts, statuses). Output valid JSON."

        if task_context:
            user_msg = (
                f"Summarize this {tool_name} tool result into compact JSON, keeping all fields relevant to the task: {task_context}\n"
                f"For lists, include ALL items with their key fields (not just the first few). "
                f"Keep IDs, names, and any metrics/fields related to the task.\n\n{summarizer_input}"
            )
        else:
            user_msg = (
                f"Summarize this {tool_name} tool result into a compact JSON with the most important fields and values. "
                f"If it's a list, include the count and first few items. "
                f"Keep IDs, names, and key metrics.\n\n{summarizer_input}"
            )

        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_msg,
            messages=[{
                "role": "user",
                "content": user_msg,
            }],
        )

        summary_text = "".join(
            b.text for b in response.content if hasattr(b, "text")
        )
        # Wrap summary with metadata
        result = json.dumps({
            "_summarized": True,
            "_original_chars": len(result_json),
            "_tool": tool_name,
            "summary": summary_text,
        })
        logger.info("Summarized %s result: %d chars -> %d chars", tool_name, len(result_json), len(result))
        return result

    except Exception as e:
        logger.warning("LLM summarization failed for %s, falling back to truncation: %s", tool_name, e)
        return None
