# Prompting Tips Analysis: Applicability to BabyAGI 3

## Overview

Analysis of external prompting best practices against the current BabyAGI 3 system.
Identifies gaps and concrete changes with estimated impact and effort.

## Current Strengths

BabyAGI 3 already has strong foundations in several areas:

- **Self-improvement pipeline** (`memory/learning.py`): FeedbackExtractor → ObjectiveEvaluator → PreferenceSummarizer. User corrections are captured, stored with embeddings, and surfaced contextually.
- **Objective concurrency** (`agent.py`): Priority queues, budget caps, token limits, cancellation tokens, retry with exponential backoff, max 5 concurrent.
- **Context-aware tool selection** (`memory/tool_context.py`): Dynamically selects relevant tools per request, always includes core 6, inventory summary for discoverability.
- **Persistent memory** (`memory/store.py`): SQLite with vector search, knowledge graph, hierarchical summaries.

## Identified Gaps

The biggest gaps are in **planning before execution**, **verification before completion**, and **error-aware retries**.

## Suggested Changes

| # | Change | Description | Where | What Already Exists | Impact | Lines to Add | Complexity | Priority |
|---|--------|-------------|-------|---------------------|--------|-------------|-----------|----------|
| 1 | **Planning step in objective prompt** | Objective prompt (`agent.py:996-998`) is bare: "Complete this objective. Work autonomously." Add planning instructions. | `agent.py:996-998` | Nothing — objectives jump straight to execution | **High** | ~15 | Low | P0 |
| 2 | **Verification before completion** | Objective marked `completed` when `run_async()` returns. Add verification instruction to prompt. | `agent.py:996-998` | `ObjectiveEvaluator` is post-mortem only, not a pre-completion gate | **High** | ~8 | Low | P0 |
| 3 | **Error-aware retry prompt** | Failed retries reuse the same prompt. Inject previous error so the agent can adjust approach. | `agent.py:1040-1058` | Retries exist but are blind — same prompt each time | **High** | ~20 | Medium | P0 |
| 4 | **Decomposition instruction** | Instruct agent to break complex objectives into sub-objectives. No guidance on when to decompose. | `agent.py:1398-1407` | Objective nesting is possible but never instructed | **Medium** | ~10 | Low | P1 |
| 5 | **Session-start learning review** | Load high-confidence negative learnings at conversation start, not just reactively. | `memory/context.py` | Learnings aggregated into `user_preferences` but context-specific ones are reactive only | **Medium** | ~15 | Medium | P1 |
| 6 | **Re-plan on failure instruction** | Add circuit-breaker: "If approach isn't working after 2-3 failed steps, stop and re-plan." | `agent.py:996-998` | No instruction to pause and reassess within a single run | **Medium** | ~5 | Low | P1 |
| 7 | **Core working principles** | Add "simplicity first, minimal impact, find root causes" to base prompt. | `agent.py:1266` | Prompt covers what agent can do but not how it should approach work | **Low-Medium** | ~8 | Low | P2 |
| 8 | **Structured plan tracking** | Have objectives write plans to notes tool, check off steps as they go. | `agent.py:996-998` + notes tool | Notes tool exists but is disconnected from objectives | **Low-Medium** | ~12 | Medium | P2 |
| 9 | **Elegance self-check** | "For non-trivial solutions, consider if there's a simpler approach before finalizing." | `agent.py:996-998` | Nothing | **Low** | ~4 | Low | P3 |
| 10 | **Learning application counter** | Increment `times_applied` when learnings are surfaced in context. Enable stale learning pruning. | `memory/learning.py` + `memory/context.py` | `times_applied` field exists but is always 0 | **Low** | ~10 | Low | P3 |

## Proposed Implementations for Top Changes

### Change #1 + #2 + #6: Enhanced Objective Prompt

**File:** `agent.py:996-998`

Current:
```python
prompt = f"""Complete this objective: {obj.goal}

Work autonomously. Use tools as needed. When done, provide a final summary."""
```

Proposed:
```python
prompt = f"""Complete this objective: {obj.goal}

APPROACH:
1. PLAN: Break this into concrete steps. Identify what tools and information you need.
2. EXECUTE: Work through your plan step by step.
3. VERIFY: Before finishing, confirm your work actually achieves the goal. Check outputs and results.
4. If something isn't working after 2-3 attempts at a step, stop and re-plan with a different approach.

Work autonomously. When done, provide a summary of what was accomplished and how it was verified."""
```

### Change #3: Error-Aware Retry

**File:** `agent.py` in `run_objective()`, around line 1054-1057

Current: retry blindly re-runs same prompt.

Proposed:
```python
# Before retry, build error-aware prompt
error_history = getattr(obj, '_error_history', [])
error_history.append(obj.last_error)
obj._error_history = error_history

# Retry prompt includes error context
error_context = "\n".join(f"- Attempt {i+1}: {e}" for i, e in enumerate(error_history))
retry_prompt = f"""Complete this objective: {obj.goal}

PREVIOUS ATTEMPTS FAILED:
{error_context}

Adjust your approach to avoid these errors. Consider a different strategy.

APPROACH:
1. PLAN: Given the previous failures, plan a new approach.
2. EXECUTE: Work through your revised plan.
3. VERIFY: Confirm your work achieves the goal.

Provide a summary when done."""
```

This would require storing the retry prompt on the Objective and using it in the next `run_async()` call.

### Change #4: Decomposition Instruction

**File:** `agent.py:1398-1407` (OBJECTIVE BEST PRACTICES section)

Add:
```
- **Decompose complex work**: If an objective involves multiple independent phases
  (e.g., research → plan → execute), consider spawning sub-objectives for parts
  that can run in parallel. Keep each sub-objective focused on one deliverable.
```

## Cost/Benefit Summary

| Category | Current State | Gap Size | Effort |
|----------|--------------|----------|--------|
| Self-Improvement | Strong (learning.py) | Small (`times_applied`, session review) | Low |
| Subagent/Decomposition | Infrastructure exists | No guidance to use it | Low |
| Planning Before Execution | Missing | Large | Low |
| Verification Before Completion | Post-mortem only | Large | Low |
| Error-Aware Retry | Blind retry | Large | Medium |
| Core Principles | Absent | Moderate | Low |
| Task Tracking | Basic notes | Moderate | Medium |

**Highest ROI: Changes #1-3** — address the biggest behavioral gaps with minimal code.
