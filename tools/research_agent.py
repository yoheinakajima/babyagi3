"""
Research Agent - Specialized agent for long-running batch data operations.

The main agent delegates research tasks here via the `research_task` tool.
This keeps the main agent simple while providing deep research capabilities.

The research agent has:
- All research tools (data_collection, batch_next, checkpoint, etc.)
- Detailed knowledge of research patterns
- Examples for common scenarios (VC lists, CRM sync, CSV enrichment)
"""

from tools import tool, tool_error

import logging

logger = logging.getLogger(__name__)

# Research agent system prompt - provides context and patterns
RESEARCH_AGENT_PROMPT = '''You are a specialized Research Agent for long-running batch data operations.

## Your Capabilities

You have access to these research tools:
- data_collection: Create/manage collections with schema and auto-deduplication
- import_collection: Bulk import from CSV or API responses
- batch_next: Get next item(s) for processing (supports batch_size for parallel)
- update_collection_item: Update item after enrichment
- export_collection: Export to CSV
- checkpoint: Save/restore progress for resumable tasks
- cursor_state: Track API pagination state
- rate_limit: Throttle API calls
- pace_work: Yield to higher priority tasks
- reset_collection_items: Reset failed/stuck items
- research_progress: Overview of all collections

## Common Patterns

### Pattern 1: Build List from Web Search (VC Research)
```
# 1. Create collection with schema
data_collection(action="create", name="vc_list",
    schema={"name": "string", "website": "string", "focus": "string", "stage": "string"},
    key_field="name")

# 2. Set up rate limiting
rate_limit(action="create", name="web_search", calls_per_minute=10)

# 3. Iteratively search and add (auto-dedupes)
search_queries = ["top VC firms 2024", "enterprise software VCs", "AI focused venture capital", ...]
for query in search_queries:
    rate_limit(action="check", name="web_search")
    results = web_search(query=query)
    data_collection(action="add", name="vc_list", source=f"search:{query}",
                   items=[parse_vc_from_result(r) for r in results])

    # Checkpoint progress
    checkpoint(action="save", task_id="vc_research", collection_name="vc_list",
              phase="collecting", progress={"queries_completed": i, "total_queries": len(search_queries)})

# 4. Enrich each VC
while True:
    item = batch_next(collection_name="vc_list")
    if item.get("done"): break

    pace_work(delay_seconds=2)  # Be nice to APIs

    # Research this VC
    details = web_search(query=f"{item['name']} VC investment focus portfolio")
    update_collection_item(collection_name="vc_list", item_id=item["_id"],
                          updates={"focus": extracted_focus, "stage": extracted_stage},
                          status="enriched")

# 5. Export
export_collection(name="vc_list", output_path="~/Downloads/vcs.csv")
checkpoint(action="delete", task_id="vc_research")
```

### Pattern 2: CRM Pagination (HubSpot/Salesforce)
```
# Load existing cursor (survives restarts)
state = cursor_state(action="load", name="hubspot_contacts")
cursor = state.get("cursor_value") if state["found"] else None
page = state.get("page_number", 0) if state["found"] else 0
total = state.get("total_fetched", 0) if state["found"] else 0

while True:
    # Fetch page from CRM
    response = composio_action(action="HUBSPOT_LIST_CONTACTS", params={"after": cursor, "limit": 100})

    # Import entire page
    import_collection(name="contacts", items=response["results"],
                     key_field="email", source=f"hubspot_page_{page}")

    # Save cursor for resume
    cursor_state(action="save", name="hubspot_contacts",
                cursor_value=response.get("paging", {}).get("next", {}).get("after"),
                page_number=page + 1,
                total_fetched=total + len(response["results"]),
                has_more=response.get("paging") is not None)

    if not response.get("paging"):
        break
    cursor = response["paging"]["next"]["after"]
    page += 1
    total += len(response["results"])

cursor_state(action="delete", name="hubspot_contacts")
```

### Pattern 3: CSV Enrichment
```
# One-liner import
import_collection(name="leads", csv_path="~/Downloads/leads.csv",
                 key_field="email", source="csv_import")

# Batch process 5 at a time
while True:
    batch = batch_next(collection_name="leads", batch_size=5)
    if batch.get("done"): break

    for item in batch["items"]:
        # Enrich each
        company_info = web_search(query=f"{item['company']} company info")
        update_collection_item(collection_name="leads", item_id=item["_id"],
                              updates={"company_size": ..., "industry": ...},
                              status="enriched")

export_collection(name="leads", output_path="~/Downloads/enriched_leads.csv")
```

### Pattern 4: Multi-Source Aggregation
```
# Create collection
data_collection(action="create", name="companies",
    schema={"name": "string", "website": "string", "source": "string"},
    key_field="website")  # Dedupe by website across sources

# Add from multiple sources
import_collection(name="companies", csv_path="~/list1.csv", key_field="website", source="csv1")
import_collection(name="companies", csv_path="~/list2.csv", key_field="website", source="csv2")

# Add from web search
results = web_search(query="top AI startups 2024")
data_collection(action="add", name="companies", source="web_search", items=[...])

# Query by source if needed
csv1_items = data_collection(action="query", name="companies", filter_source="csv1")
```

## Best Practices

1. **Always checkpoint** after meaningful progress (every 10-50 items, or after each phase)
2. **Use rate limiting** for external APIs to avoid throttling
3. **Use pace_work** to yield to higher priority tasks
4. **Track sources** to understand data provenance
5. **Reset stuck items** if processing was interrupted (from_status="processing")
6. **Export incrementally** if task might be interrupted

## Status Flow
pending → processing → enriched
                    → error (can be reset back to pending)

## Your Task
Complete the research task described below. Work autonomously, checkpoint progress,
handle errors gracefully, and provide a summary when done.
'''


@tool()
def research_task(
    goal: str,
    collection_name: str = None,
    output_path: str = None,
    priority: int = 8,
    budget_usd: float = None,
    agent=None
) -> dict:
    """
    Delegate a research task to the specialized Research Agent.

    Use this for long-running batch operations like:
    - Building lists (1000 VCs, companies, contacts)
    - CRM sync (paginate through HubSpot/Salesforce)
    - CSV enrichment (import → enrich → export)
    - Multi-source data aggregation

    The Research Agent handles:
    - Checkpointing (survives restarts)
    - Rate limiting (avoids API throttling)
    - Deduplication (by key field)
    - Progress tracking
    - CSV export

    Args:
        goal: Natural language description of the research task
        collection_name: Name for the data collection (auto-generated if not provided)
        output_path: Where to save results (default: ~/Downloads/{collection_name}.csv)
        priority: 1-10, lower = higher priority (default: 8 = low, won't block other work)
        budget_usd: Maximum cost allowed (optional)

    Examples:
        research_task(goal="Build a list of 500 AI-focused VCs with their investment focus and typical check size")

        research_task(goal="Sync all contacts from HubSpot and enrich with LinkedIn data",
                     collection_name="hubspot_contacts")

        research_task(goal="Import leads.csv, research each company, add employee count and industry",
                     output_path="~/Downloads/enriched_leads.csv")

    Returns:
        {"objective_goal": "...", "spawn_command": "..."} - Use with objective tool to start
    """
    import re
    from uuid import uuid4

    # Generate collection name if not provided
    if not collection_name:
        # Extract key noun from goal
        words = re.findall(r'\b(?:VCs?|companies|contacts|leads|startups|firms|people)\b', goal.lower())
        base_name = words[0] if words else "research"
        collection_name = f"{base_name}_{str(uuid4())[:4]}"

    output = output_path or f"~/Downloads/{collection_name}.csv"

    # Build the full goal with embedded research agent context
    full_goal = f"""RESEARCH TASK: {goal}

{RESEARCH_AGENT_PROMPT}

=== YOUR SPECIFIC TASK ===
Goal: {goal}
Collection name: {collection_name}
Output path: {output}

Work autonomously using the research tools. Checkpoint progress regularly.
When complete, export results and provide a summary."""

    return {
        "research_goal": full_goal,
        "collection_name": collection_name,
        "output_path": output,
        "priority": priority,
        "budget_usd": budget_usd,
        "spawn_command": f"objective(action='spawn', goal=<research_goal>, priority={priority})",
        "message": "Use the objective tool with the research_goal above to start this task. The goal includes all necessary context for the research agent."
    }


@tool()
def research_status(
    task_id: str = None,
    collection_name: str = None,
    agent=None
) -> dict:
    """
    Check status of research tasks and collections.

    Args:
        task_id: Specific task ID to check
        collection_name: Specific collection to check

    Returns overview of research progress including:
    - Task status (if task_id provided)
    - Collection stats (items, progress, errors)
    - Active checkpoints
    """
    # Import here to avoid circular dependency
    from tools.research import research_progress, _get_research_db, _get_collection
    import json

    result = {}

    # Check specific task if provided
    if task_id and agent:
        try:
            task_result = agent.tools["objective"]["fn"](
                {"action": "check", "id": task_id},
                agent
            )
            result["task"] = task_result
        except Exception as e:
            logger.debug("Could not check research task status for task '%s': %s", task_id, e)

    # Get research progress
    progress = research_progress()

    # Filter to specific collection if requested
    if collection_name:
        progress["collections"] = [
            c for c in progress["collections"]
            if c["name"] == collection_name
        ]
        progress["checkpoints"] = [
            c for c in progress["checkpoints"]
            if c["collection"] == collection_name
        ]

    result["research"] = progress

    return result
