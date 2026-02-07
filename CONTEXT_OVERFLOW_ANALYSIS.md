# Context Window Overflow Analysis & Solution Proposals

## Incident Summary

209,649 tokens were sent to Claude (200,000 max) during a multi-step CLI session involving Attio Composio tools. The agent entered the tool loop in `agent.py:run_async()` (line 872), calling tools like `attio_list_list_entries`, `attio_get_company`, `web_search`, etc. Each raw tool result was appended to the thread unbounded. After many iterations, the accumulated thread exceeded the context window.

---

## Root Cause Map

| # | Gap | Location | What Happens |
|---|-----|----------|--------------|
| 1 | **No tool result truncation** | `agent.py:946-958` | `json.dumps(result)` serialized in full, appended raw. A single Attio API response can be 20K+ tokens. |
| 2 | **No thread trimming before LLM call** | `agent.py:873-879` | `messages=thread` sent as-is. No pre-flight token check, no pruning of old messages. Thread grows unbounded within the `while True` loop. |
| 3 | **No catch-and-recover for context overflow** | `listeners/cli.py:146-148` | `ContextWindowExceededError` is caught by the generic `except Exception`, logged, and surfaced to user. No retry, no trim, no recovery. |
| 4 | **Composio loads too many tools** | `tools/skills.py:1197` | `limit=500` per app. Attio alone registers ~86 tools. Each tool schema costs ~100-150 tokens. |
| 5 | **Tool selection fallback sends ALL tools** | `agent.py:1594-1595` | If `_current_tool_selection is None` (memory system not initialized), ALL registered tool schemas are sent. |
| 6 | **Memory context only protects system prompt** | `memory/context.py:53` | `max_context_tokens: 4000` controls only the memory section of the system prompt. Thread and tool schemas are unprotected. |

### What IS Protected Today

| Component | Budget | Mechanism |
|-----------|--------|-----------|
| Memory context sections | ~4,000 tokens total | `_truncate_to_budget()` in `memory/context.py:68-75` |
| Tool selection (when active) | 25 tools max | `ToolContextConfig.max_active_tools` in `memory/tool_context.py:34` |

### What IS NOT Protected

| Component | Typical Size | Worst Case |
|-----------|-------------|------------|
| Thread (message history) | 10K-50K | **150K+** (long sessions with tool-heavy loops) |
| Individual tool results | 500-5K | **30K+** (Attio list responses) |
| Tool schemas (fallback) | 5K | **50K+** (500 Composio tools loaded) |
| System prompt (base) | 2K-5K | 10K (with large status summaries) |

---

## Proposed Solutions

### Solution 1: Tool Result Truncation (Quick Win)

**What:** Cap individual tool results before appending to thread. Store the full result separately; put a truncated version (or summary) in the thread.

**Where:** `agent.py:944-958` — after `json.dumps(result)`, check length and truncate.

**Implementation sketch:**
```python
MAX_TOOL_RESULT_CHARS = 12000  # ~3,000 tokens

result_json = json.dumps(result, default=json_serialize)
if len(result_json) > MAX_TOOL_RESULT_CHARS:
    # Store full result for reference
    self._store_full_result(block.id, result_json)
    # Truncate for the thread
    result_json = result_json[:MAX_TOOL_RESULT_CHARS] + \
        f"\n... [TRUNCATED — full result: {len(result_json)} chars. " \
        f"Use memory tool to retrieve if needed.]"
```

**Estimated Lines of Code:** ~15-25 lines in `agent.py`

| Pros | Cons |
|------|------|
| Simplest fix, directly addresses the #1 cause | Truncation can lose critical data mid-JSON |
| No architectural changes needed | Needs a retrieval mechanism for full results |
| Immediate protection against giant API responses | Hard to pick the "right" truncation point in structured data |

**Impact:** High — prevents the single largest contributor to overflow (giant tool results).

**Recommendation:** **Do this first.** It's the highest-ROI change.

---

### Solution 2: Thread Trimming Before LLM Calls (Critical)

**What:** Before every `client.messages.create()` call, estimate the total token count (system prompt + tools + thread). If over budget, prune the oldest messages from the thread while preserving the most recent N messages and all system messages.

**Where:** `agent.py:872-879` — add a pre-flight check before the API call.

**Implementation sketch:**
```python
MODEL_CONTEXT_LIMIT = 190_000  # Leave 10K headroom
CHARS_PER_TOKEN = 4

def _trim_thread(self, thread, system_prompt, tool_schemas):
    total_chars = len(system_prompt) + sum(len(json.dumps(s)) for s in tool_schemas)
    for msg in thread:
        total_chars += len(json.dumps(msg))

    estimated_tokens = total_chars // CHARS_PER_TOKEN
    if estimated_tokens <= MODEL_CONTEXT_LIMIT:
        return thread

    # Keep first user message + last N messages, drop middle
    preserved_head = thread[:1]
    preserved_tail = thread[-10:]
    trimmed = preserved_head + [
        {"role": "user", "content": "[Earlier messages trimmed to fit context window]"}
    ] + preserved_tail
    return trimmed
```

**Estimated Lines of Code:** ~40-60 lines in `agent.py` (new method + call site)

| Pros | Cons |
|------|------|
| Prevents overflow regardless of conversation length | Loses older context, which may be needed |
| Deterministic — no LLM call required for summarization | Token estimation is approximate (4 chars/token) |
| Works even if tool result truncation is in place | Thread trimming mid-loop can confuse the LLM |

**Impact:** High — provides a hard ceiling on thread size.

**Recommendation:** **Do this second.** It's the safety net that prevents overflow in all cases.

---

### Solution 3: Catch-and-Recover for Context Overflow (Safety Net)

**What:** Catch `ContextWindowExceededError` (or the API error code) in the tool loop, trim the thread aggressively, and retry the API call.

**Where:** `agent.py:872-879` (wrap the API call) and `listeners/cli.py:136-148` (outer handler).

**Implementation sketch:**
```python
# In agent.py run_async(), inside the while True loop:
try:
    response = await self.client.messages.create(...)
except Exception as e:
    if "context_length_exceeded" in str(e) or "context_window" in str(e):
        logger.warning("Context overflow, trimming thread and retrying...")
        thread = self._emergency_trim(thread)
        # Also reduce tool schemas
        tool_schemas = self._reduce_tool_schemas()
        response = await self.client.messages.create(
            model=self.model, max_tokens=8096,
            system=system_prompt, tools=tool_schemas,
            messages=thread
        )
    else:
        raise
```

**Estimated Lines of Code:** ~30-40 lines in `agent.py`, ~10 lines in `listeners/cli.py`

| Pros | Cons |
|------|------|
| Prevents user-facing crashes entirely | Reactive, not preventive — the overflow already happened |
| Works even when estimation is wrong | Retry may also fail if trimming isn't aggressive enough |
| Simple error-handling pattern | Adds complexity to the loop; risk of infinite retry |

**Impact:** Medium — doesn't prevent the issue but ensures graceful recovery.

**Recommendation:** **Do this third.** Defense in depth — catches anything solutions 1 and 2 miss.

---

### Solution 4: Composio Tool Limiting & Lazy Loading (Structural)

**What:** Instead of loading all 86+ Attio actions at once, load only a curated subset (10-15 most common actions per app). Provide a "discover more tools" meta-tool that loads additional actions on demand.

**Where:** `tools/skills.py:1192-1197` and `agent.py:1574-1595`

**Implementation sketch:**
```python
# In tools/skills.py — cap default actions per app
DEFAULT_ACTION_LIMIT = 15  # Instead of 500

# Provide a meta-tool for discovering more
async def discover_tools(input, agent):
    """Search for additional tools in an app."""
    app = input["app"]
    query = input.get("query", "")
    all_actions = client.tools.get_raw_composio_tools(toolkits=[app], limit=500)
    # Return descriptions only (not full schemas)
    return [{"name": a.slug, "description": a.description} for a in all_actions]
```

**Estimated Lines of Code:** ~40-60 lines across `tools/skills.py` and a new meta-tool

| Pros | Cons |
|------|------|
| Reduces baseline token cost by 70%+ per app | May miss the exact tool needed for a task |
| Faster API calls (fewer tool schemas to process) | Requires LLM to use meta-tool (extra round-trip) |
| More aligned with how tool_context.py intends to work | Requires curating "default" action lists per app |

**Impact:** Medium-High — reduces chronic token pressure, not just overflow events.

**Recommendation:** **Do this as a structural improvement.** It fixes the root cause of "too many tools" rather than treating the symptom.

---

### Solution 5: LLM-Based Tool Result Summarization (Ambitious)

**What:** When a tool result exceeds a threshold, call a fast/cheap model (e.g., Haiku) to summarize it before inserting into the thread. Store the full result in a side-store accessible via the memory tool.

**Where:** `agent.py:944-958` — replace simple truncation with an LLM summarization call.

**Implementation sketch:**
```python
MAX_RESULT_TOKENS = 2000
SUMMARIZE_THRESHOLD = 4000  # chars

if len(result_json) > SUMMARIZE_THRESHOLD:
    # Store full result
    result_id = self._store_full_result(block.id, result_json)
    # Summarize with fast model
    summary = await self._summarize_result(
        tool_name=block.name,
        result_json=result_json,
        max_tokens=MAX_RESULT_TOKENS
    )
    result_json = json.dumps({
        "summary": summary,
        "full_result_id": result_id,
        "original_size": len(result_json),
        "note": "Full result stored. Use memory tool to retrieve."
    })
```

**Estimated Lines of Code:** ~60-80 lines (new method + call site + side-store)

| Pros | Cons |
|------|------|
| Preserves semantic content (unlike truncation) | Adds latency per large tool result (LLM call) |
| LLM can extract what's relevant to the current task | Adds cost (Haiku calls per tool result) |
| Full result still available for retrieval | Summarization can miss details the agent needs |
| Better UX than raw truncation | More complex; harder to debug |

**Impact:** High — best quality context preservation, but highest implementation cost.

**Recommendation:** **Implement after Solutions 1-3 are in place.** This replaces the simple truncation (Solution 1) with an intelligent version.

---

### Solution 6: Pre-Flight Token Budget System (Comprehensive)

**What:** Create a centralized `ContextBudget` class that tracks and enforces token limits across all components: system prompt, tool schemas, thread messages, and tool results. Every component asks the budget for its allocation before assembling the API call.

**Where:** New file `context_budget.py`, integrated into `agent.py:872-879`

**Implementation sketch:**
```python
@dataclass
class ContextBudget:
    model_limit: int = 200_000
    reserved_for_response: int = 8_096

    # Allocations (in priority order)
    system_prompt_max: int = 10_000
    tool_schemas_max: int = 30_000
    thread_max: int = 0  # Computed as remainder

    def allocate(self, system_prompt, tool_schemas, thread):
        available = self.model_limit - self.reserved_for_response
        sp_tokens = self._count(system_prompt)
        ts_tokens = self._count_schemas(tool_schemas)
        thread_budget = available - sp_tokens - ts_tokens
        return self._fit_thread(thread, thread_budget)
```

**Estimated Lines of Code:** ~100-150 lines (new module + integration)

| Pros | Cons |
|------|------|
| Single source of truth for all token accounting | Largest implementation effort |
| Prevents overflow by construction, not reaction | Token counting is still approximate without tiktoken |
| Makes limits visible and configurable | Requires refactoring the API call site |
| Can be tuned per model (200K vs 100K vs etc.) | Over-engineering risk if simpler solutions suffice |

**Impact:** Highest — eliminates the class of bugs entirely.

**Recommendation:** **Design target for v2.** Get Solutions 1-4 working first, then consolidate into this.

---

## Recommended Implementation Order

| Priority | Solution | Impact | Cost (LOC) | Addresses |
|----------|----------|--------|------------|-----------|
| **P0** | 1. Tool result truncation | High | ~20 lines | Giant tool results blowing up a single turn |
| **P0** | 2. Thread trimming | High | ~50 lines | Long conversations accumulating past the limit |
| **P1** | 3. Error recovery | Medium | ~40 lines | Graceful handling when estimation is wrong |
| **P1** | 4. Composio tool limiting | Medium-High | ~50 lines | Too many tool schemas eating the budget |
| **P2** | 5. LLM summarization | High | ~70 lines | Quality improvement over raw truncation |
| **P2** | 6. Pre-flight budget system | Highest | ~130 lines | Systemic fix, eliminates the bug class |

### Quick-Win Stack (P0 — do now)

Solutions 1 + 2 + 3 together are ~110 lines of code and cover 95%+ of overflow scenarios:
- Solution 1 prevents any single tool result from being huge
- Solution 2 prevents the thread from growing past the limit
- Solution 3 catches anything the first two miss

### Structural Stack (P1 — do next sprint)

Solution 4 reduces chronic token pressure from Composio tools, making the system breathe easier even before overflow is a risk.

### Quality Stack (P2 — planned improvement)

Solutions 5 and 6 replace brute-force truncation with intelligent summarization and centralized budget management.
