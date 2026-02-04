"""
Research Tools - Long-running batch research with checkpointing and persistence.

Provides reliable tools for tasks like:
- Building a list of 1000 VCs through iterative search
- Researching each item to fill specific columns
- Paginating through CRM contacts
- Any batch data collection with deduplication

Key features:
- SQLite-backed collections with schema
- Automatic deduplication by key field
- Progress checkpointing (survives restarts)
- Rate limiting to avoid API throttling
- CSV export
- Low-priority mode (yields to other work)

Recommended usage for long-running research:

1. Start as low-priority objective:
   objective(action="spawn", goal="Research 1000 VCs...", priority=8)

2. Create collection with schema:
   data_collection(action="create", name="vc_list",
                   schema={"name": "string", "website": "string", ...},
                   key_field="name")

3. Build list (with deduplication):
   - web_search for VCs
   - data_collection(action="add", ...) to add results
   - Repeat until target count reached
   - Use checkpoint() to save progress

4. Enrich each item:
   - Use batch_next() to get items one by one
   - Research each, update_collection_item() with findings
   - Rate limiting prevents API throttling
   - Checkpoint after each batch

5. Export to CSV:
   export_collection(name="vc_list", output_path="~/Downloads/vcs.csv")
"""

import asyncio
import csv
import io
import json
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from tools import tool, tool_error


# ═══════════════════════════════════════════════════════════
# MEMORY INTEGRATION - Log research findings for extraction
# ═══════════════════════════════════════════════════════════


def _log_to_memory(agent, content: str, event_type: str = "research_finding", metadata: dict = None):
    """
    Log research findings to memory for extraction into searchable knowledge.

    This ensures research data (enrichments, findings, etc.) gets extracted
    by the memory system and becomes part of the agent's searchable knowledge.

    Args:
        agent: The agent instance (to access memory)
        content: The content to log (will be extracted for entities/facts)
        event_type: Type of event (research_finding, research_summary, etc.)
        metadata: Optional metadata dict
    """
    if agent is None:
        return

    try:
        # Access memory through agent
        memory = getattr(agent, 'memory', None)
        if memory is None:
            return

        # Log the event for extraction
        memory.log_event(
            content=content,
            event_type=event_type,
            direction="internal",
            metadata=metadata or {},
        )
    except Exception:
        pass  # Never fail research operations due to memory logging


def _format_enrichment_for_memory(item_name: str, collection_name: str, updates: dict) -> str:
    """Format enrichment data into natural language for extraction."""
    lines = [f"Research finding about {item_name} (from {collection_name} collection):"]

    for key, value in updates.items():
        if key.startswith("_"):
            continue  # Skip internal fields
        if value is not None and value != "":
            # Format as natural language for better extraction
            lines.append(f"- {key.replace('_', ' ').title()}: {value}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# DATA COLLECTION - Persistent batch data storage
# ═══════════════════════════════════════════════════════════


def _get_research_db() -> sqlite3.Connection:
    """Get connection to research database."""
    db_path = Path("~/.babyagi/research/research.db").expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_research_tables(conn)
    return conn


def _init_research_tables(conn: sqlite3.Connection):
    """Initialize research tables."""
    cur = conn.cursor()

    # Collections metadata
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            schema TEXT NOT NULL,
            key_field TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            item_count INTEGER DEFAULT 0,
            status TEXT DEFAULT 'active'
        )
    """)

    # Collection items (flexible JSON storage)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS collection_items (
            id TEXT PRIMARY KEY,
            collection_id TEXT NOT NULL,
            key_value TEXT NOT NULL,
            data TEXT NOT NULL,
            source TEXT,
            status TEXT DEFAULT 'pending',
            enriched_at TEXT,
            error TEXT,
            retry_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (collection_id) REFERENCES collections(id),
            UNIQUE (collection_id, key_value)
        )
    """)

    # Add source column if it doesn't exist (migration for existing DBs)
    try:
        cur.execute("ALTER TABLE collection_items ADD COLUMN source TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    # Checkpoints for resumable tasks
    cur.execute("""
        CREATE TABLE IF NOT EXISTS checkpoints (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL UNIQUE,
            collection_id TEXT,
            phase TEXT NOT NULL,
            progress TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (collection_id) REFERENCES collections(id)
        )
    """)

    # Rate limiter state
    cur.execute("""
        CREATE TABLE IF NOT EXISTS rate_limits (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            calls_per_minute INTEGER NOT NULL,
            last_call_at TEXT,
            call_count_this_minute INTEGER DEFAULT 0,
            minute_started_at TEXT
        )
    """)

    # Cursor states for API pagination (CRM, etc.)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cursor_states (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            cursor_value TEXT,
            page_number INTEGER DEFAULT 0,
            total_fetched INTEGER DEFAULT 0,
            has_more INTEGER DEFAULT 1,
            metadata TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    conn.commit()


@tool()
def data_collection(
    action: str,
    name: str = None,
    description: str = None,
    schema: dict = None,
    key_field: str = None,
    item: dict = None,
    items: list = None,
    source: str = None,
    filter_status: str = None,
    filter_source: str = None,
    limit: int = None,
    offset: int = 0,
    collection_id: str = None,
    item_id: str = None,
    new_status: str = None,
    error: str = None,
    agent=None
) -> dict:
    """
    Manage data collections for batch research tasks.

    Collections store items with automatic deduplication based on a key field.
    Perfect for building lists like "1000 VCs" or "all CRM contacts".

    Actions:
    - create: Create a new collection with schema
    - add: Add item(s) to collection (auto-dedupes by key_field)
    - list: List collections
    - get: Get collection details and stats
    - query: Query items from collection with optional status filter
    - update_status: Update item status (pending → processing → enriched → error)
    - delete: Delete a collection

    Args:
        action: create, add, list, get, query, update_status, delete
        name: Collection name (for create, add, get, query)
        description: Collection description (for create)
        schema: Column definitions {"name": "string", "website": "string", ...} (for create)
        key_field: Field to dedupe on, e.g. "name" or "website" (for create)
        item: Single item to add (for add)
        items: Multiple items to add (for add)
        source: Track where items came from, e.g. "crm_hubspot", "web_search", "csv_import" (for add)
        filter_status: Filter by status: pending, processing, enriched, error (for query)
        filter_source: Filter by source (for query)
        limit: Max items to return (for query)
        offset: Skip first N items (for query)
        collection_id: Collection ID (alternative to name)
        item_id: Item ID (for update_status)
        new_status: New status value (for update_status)
        error: Error message (for update_status with status=error)

    Example - Create VC collection:
        data_collection(action="create", name="vc_list",
                       schema={"name": "string", "website": "string", "focus": "string", "stage": "string"},
                       key_field="name")

    Example - Add VCs from web search (with source tracking):
        data_collection(action="add", name="vc_list", source="web_search", items=[
            {"name": "Sequoia", "website": "sequoiacap.com"},
            {"name": "a16z", "website": "a16z.com"}
        ])

    Example - Add from CRM:
        data_collection(action="add", name="contacts", source="hubspot_page_1", items=[...])
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    if action == "create":
        if not name or not schema or not key_field:
            return tool_error(
                "Missing required fields",
                fix="Provide name, schema, and key_field"
            )

        if key_field not in schema:
            return tool_error(
                f"key_field '{key_field}' must be in schema",
                fix=f"Add '{key_field}' to schema or use a different key_field"
            )

        coll_id = str(uuid4())[:8]
        try:
            cur.execute("""
                INSERT INTO collections (id, name, description, schema, key_field, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (coll_id, name, description, json.dumps(schema), key_field, now, now))
            conn.commit()

            return {
                "created": coll_id,
                "name": name,
                "schema": schema,
                "key_field": key_field,
                "message": f"Collection '{name}' created. Add items with data_collection(action='add', name='{name}', items=[...])"
            }
        except sqlite3.IntegrityError:
            return tool_error(
                f"Collection '{name}' already exists",
                fix=f"Use a different name or query existing with data_collection(action='get', name='{name}')"
            )

    elif action == "add":
        # Get collection
        coll = _get_collection(cur, name, collection_id)
        if not coll:
            return tool_error(
                f"Collection not found",
                fix="Create it first with data_collection(action='create', ...)"
            )

        # Normalize to list
        items_to_add = []
        if item:
            items_to_add.append(item)
        if items:
            items_to_add.extend(items)

        if not items_to_add:
            return tool_error("No items to add", fix="Provide item or items")

        key_field = coll["key_field"]
        added = 0
        skipped = 0
        errors = []

        for itm in items_to_add:
            if key_field not in itm:
                errors.append(f"Missing key_field '{key_field}' in item: {itm}")
                continue

            key_value = str(itm[key_field]).lower().strip()
            item_id = str(uuid4())[:8]

            try:
                cur.execute("""
                    INSERT INTO collection_items (id, collection_id, key_value, data, source, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (item_id, coll["id"], key_value, json.dumps(itm), source, now, now))
                added += 1
            except sqlite3.IntegrityError:
                # Already exists - this is expected deduplication
                skipped += 1

        # Update collection count
        cur.execute("""
            UPDATE collections
            SET item_count = (SELECT COUNT(*) FROM collection_items WHERE collection_id = ?),
                updated_at = ?
            WHERE id = ?
        """, (coll["id"], now, coll["id"]))
        conn.commit()

        # Get new count
        cur.execute("SELECT item_count FROM collections WHERE id = ?", (coll["id"],))
        new_count = cur.fetchone()["item_count"]

        result = {
            "collection": coll["name"],
            "added": added,
            "skipped_duplicates": skipped,
            "total_items": new_count
        }
        if errors:
            result["errors"] = errors

        return result

    elif action == "list":
        cur.execute("""
            SELECT id, name, description, key_field, item_count, status, created_at
            FROM collections
            ORDER BY created_at DESC
        """)
        collections = [dict(row) for row in cur.fetchall()]
        return {"collections": collections, "count": len(collections)}

    elif action == "get":
        coll = _get_collection(cur, name, collection_id)
        if not coll:
            return tool_error("Collection not found")

        # Get status breakdown
        cur.execute("""
            SELECT status, COUNT(*) as count
            FROM collection_items
            WHERE collection_id = ?
            GROUP BY status
        """, (coll["id"],))
        status_counts = {row["status"]: row["count"] for row in cur.fetchall()}

        return {
            "id": coll["id"],
            "name": coll["name"],
            "description": coll["description"],
            "schema": json.loads(coll["schema"]),
            "key_field": coll["key_field"],
            "item_count": coll["item_count"],
            "status_breakdown": status_counts,
            "created_at": coll["created_at"],
            "updated_at": coll["updated_at"]
        }

    elif action == "query":
        coll = _get_collection(cur, name, collection_id)
        if not coll:
            return tool_error("Collection not found")

        query = "SELECT * FROM collection_items WHERE collection_id = ?"
        params = [coll["id"]]

        if filter_status:
            query += " AND status = ?"
            params.append(filter_status)

        if filter_source:
            query += " AND source = ?"
            params.append(filter_source)

        query += " ORDER BY created_at"

        if limit:
            query += " LIMIT ?"
            params.append(limit)
        if offset:
            query += " OFFSET ?"
            params.append(offset)

        cur.execute(query, params)
        items = []
        for row in cur.fetchall():
            item_data = json.loads(row["data"])
            item_data["_id"] = row["id"]
            item_data["_status"] = row["status"]
            item_data["_source"] = row["source"]
            item_data["_enriched_at"] = row["enriched_at"]
            items.append(item_data)

        return {
            "collection": coll["name"],
            "items": items,
            "count": len(items),
            "filter_status": filter_status,
            "filter_source": filter_source
        }

    elif action == "update_status":
        if not item_id:
            return tool_error("item_id required")
        if not new_status:
            return tool_error("new_status required")

        update_fields = ["status = ?", "updated_at = ?"]
        params = [new_status, now]

        if new_status == "enriched":
            update_fields.append("enriched_at = ?")
            params.append(now)

        if error:
            update_fields.append("error = ?")
            update_fields.append("retry_count = retry_count + 1")
            params.append(error)

        params.append(item_id)

        cur.execute(f"""
            UPDATE collection_items
            SET {", ".join(update_fields)}
            WHERE id = ?
        """, params)
        conn.commit()

        return {"updated": item_id, "new_status": new_status}

    elif action == "delete":
        coll = _get_collection(cur, name, collection_id)
        if not coll:
            return tool_error("Collection not found")

        cur.execute("DELETE FROM collection_items WHERE collection_id = ?", (coll["id"],))
        cur.execute("DELETE FROM checkpoints WHERE collection_id = ?", (coll["id"],))
        cur.execute("DELETE FROM collections WHERE id = ?", (coll["id"],))
        conn.commit()

        return {"deleted": coll["name"], "items_removed": coll["item_count"]}

    return tool_error(f"Unknown action: {action}")


def _get_collection(cur, name: str = None, collection_id: str = None) -> dict | None:
    """Helper to get collection by name or ID."""
    if collection_id:
        cur.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
    elif name:
        cur.execute("SELECT * FROM collections WHERE name = ?", (name,))
    else:
        return None

    row = cur.fetchone()
    return dict(row) if row else None


# ═══════════════════════════════════════════════════════════
# CHECKPOINTING - Save and restore task progress
# ═══════════════════════════════════════════════════════════


@tool()
def checkpoint(
    action: str,
    task_id: str,
    collection_name: str = None,
    phase: str = None,
    progress: dict = None,
    agent=None
) -> dict:
    """
    Save and restore checkpoints for long-running tasks.

    Use this to make tasks resumable after interruptions.
    Store current progress (index, phase, partial results) and restore on restart.

    Actions:
    - save: Save current progress
    - load: Load saved progress (returns None if no checkpoint)
    - delete: Clear checkpoint after task completes

    Args:
        action: save, load, delete
        task_id: Unique identifier for this task
        collection_name: Associated collection (optional)
        phase: Current phase like "collecting", "enriching", "exporting"
        progress: Dict with progress state {"current_index": 50, "total": 1000, ...}

    Example - Save checkpoint:
        checkpoint(action="save", task_id="vc_research_001",
                  phase="enriching",
                  progress={"current_index": 50, "total": 1000, "last_processed": "sequoia"})

    Example - Load on resume:
        result = checkpoint(action="load", task_id="vc_research_001")
        if result["found"]:
            start_from = result["progress"]["current_index"]
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    if action == "save":
        if not phase or progress is None:
            return tool_error("phase and progress required for save")

        # Get collection ID if name provided
        coll_id = None
        if collection_name:
            coll = _get_collection(cur, collection_name)
            if coll:
                coll_id = coll["id"]

        # Upsert checkpoint
        cur.execute("""
            INSERT INTO checkpoints (id, task_id, collection_id, phase, progress, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task_id) DO UPDATE SET
                phase = excluded.phase,
                progress = excluded.progress,
                updated_at = excluded.updated_at
        """, (str(uuid4())[:8], task_id, coll_id, phase, json.dumps(progress), now, now))
        conn.commit()

        return {
            "saved": task_id,
            "phase": phase,
            "progress": progress,
            "message": f"Checkpoint saved. Task can resume from {phase}"
        }

    elif action == "load":
        cur.execute("""
            SELECT * FROM checkpoints WHERE task_id = ?
        """, (task_id,))
        row = cur.fetchone()

        if not row:
            return {"found": False, "task_id": task_id}

        return {
            "found": True,
            "task_id": task_id,
            "phase": row["phase"],
            "progress": json.loads(row["progress"]),
            "updated_at": row["updated_at"]
        }

    elif action == "delete":
        cur.execute("DELETE FROM checkpoints WHERE task_id = ?", (task_id,))
        conn.commit()
        return {"deleted": task_id}

    return tool_error(f"Unknown action: {action}")


# ═══════════════════════════════════════════════════════════
# RATE LIMITING - Avoid API throttling
# ═══════════════════════════════════════════════════════════


@tool()
def rate_limit(
    action: str,
    name: str = None,
    calls_per_minute: int = None,
    agent=None
) -> dict:
    """
    Manage rate limits for batch operations.

    Use this to avoid hitting API rate limits during batch research.
    Call 'check' before each API call to automatically throttle.

    Actions:
    - create: Create a rate limiter (e.g., 10 calls/minute for web search)
    - check: Check if we can make a call (waits if needed, returns when clear)
    - status: Get current rate limit status
    - delete: Remove rate limiter

    Args:
        action: create, check, status, delete
        name: Rate limiter name (e.g., "web_search", "crm_api")
        calls_per_minute: Max calls allowed per minute (for create)

    Example:
        # Set up rate limit
        rate_limit(action="create", name="web_search", calls_per_minute=10)

        # Before each search
        rate_limit(action="check", name="web_search")  # Waits if needed
        web_search(query="...")
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()
    now_ts = time.time()

    if action == "create":
        if not name or not calls_per_minute:
            return tool_error("name and calls_per_minute required")

        cur.execute("""
            INSERT INTO rate_limits (id, name, calls_per_minute, minute_started_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET calls_per_minute = excluded.calls_per_minute
        """, (str(uuid4())[:8], name, calls_per_minute, now))
        conn.commit()

        return {
            "created": name,
            "calls_per_minute": calls_per_minute,
            "message": f"Rate limiter '{name}' set to {calls_per_minute} calls/minute"
        }

    elif action == "check":
        if not name:
            return tool_error("name required")

        cur.execute("SELECT * FROM rate_limits WHERE name = ?", (name,))
        row = cur.fetchone()

        if not row:
            # No rate limit configured - allow immediately
            return {"allowed": True, "waited": 0}

        calls_per_minute = row["calls_per_minute"]
        minute_started = row["minute_started_at"]
        call_count = row["call_count_this_minute"] or 0

        # Check if we're in a new minute
        if minute_started:
            minute_start_ts = datetime.fromisoformat(minute_started).timestamp()
            elapsed = now_ts - minute_start_ts

            if elapsed >= 60:
                # New minute - reset counter
                call_count = 0
                minute_started = now
        else:
            minute_started = now
            call_count = 0

        waited = 0

        # If at limit, wait for next minute
        if call_count >= calls_per_minute:
            minute_start_ts = datetime.fromisoformat(minute_started).timestamp()
            wait_time = 60 - (now_ts - minute_start_ts) + 0.1  # Small buffer
            if wait_time > 0:
                time.sleep(wait_time)
                waited = wait_time
                # Reset for new minute
                call_count = 0
                minute_started = datetime.now().isoformat()

        # Record this call
        cur.execute("""
            UPDATE rate_limits
            SET last_call_at = ?, call_count_this_minute = ?, minute_started_at = ?
            WHERE name = ?
        """, (datetime.now().isoformat(), call_count + 1, minute_started, name))
        conn.commit()

        return {
            "allowed": True,
            "waited": round(waited, 2),
            "calls_this_minute": call_count + 1,
            "limit": calls_per_minute
        }

    elif action == "status":
        if not name:
            return tool_error("name required")

        cur.execute("SELECT * FROM rate_limits WHERE name = ?", (name,))
        row = cur.fetchone()

        if not row:
            return {"found": False, "name": name}

        return {
            "name": name,
            "calls_per_minute": row["calls_per_minute"],
            "calls_this_minute": row["call_count_this_minute"] or 0,
            "last_call_at": row["last_call_at"]
        }

    elif action == "delete":
        if not name:
            return tool_error("name required")

        cur.execute("DELETE FROM rate_limits WHERE name = ?", (name,))
        conn.commit()
        return {"deleted": name}

    return tool_error(f"Unknown action: {action}")


# ═══════════════════════════════════════════════════════════
# CSV EXPORT - Export collections to CSV
# ═══════════════════════════════════════════════════════════


@tool()
def export_collection(
    name: str = None,
    collection_id: str = None,
    output_path: str = None,
    columns: list = None,
    filter_status: str = None,
    include_metadata: bool = False,
    agent=None
) -> dict:
    """
    Export a collection to CSV file.

    Args:
        name: Collection name
        collection_id: Or collection ID
        output_path: Where to save CSV (default: ~/Downloads/{name}.csv)
        columns: Specific columns to include (default: all schema columns)
        filter_status: Only export items with this status
        include_metadata: Include _id, _status, _enriched_at columns

    Returns path to created CSV file.
    """
    conn = _get_research_db()
    cur = conn.cursor()

    coll = _get_collection(cur, name, collection_id)
    if not coll:
        return tool_error("Collection not found")

    schema = json.loads(coll["schema"])

    # Build column list
    if columns:
        export_cols = columns
    else:
        export_cols = list(schema.keys())

    if include_metadata:
        export_cols = ["_id", "_status", "_enriched_at"] + export_cols

    # Query items
    query = "SELECT * FROM collection_items WHERE collection_id = ?"
    params = [coll["id"]]

    if filter_status:
        query += " AND status = ?"
        params.append(filter_status)

    query += " ORDER BY created_at"
    cur.execute(query, params)

    # Build CSV
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=export_cols, extrasaction='ignore')
    writer.writeheader()

    row_count = 0
    for row in cur.fetchall():
        item_data = json.loads(row["data"])
        if include_metadata:
            item_data["_id"] = row["id"]
            item_data["_status"] = row["status"]
            item_data["_enriched_at"] = row["enriched_at"]
        writer.writerow(item_data)
        row_count += 1

    # Determine output path
    if not output_path:
        downloads_dir = Path("~/Downloads").expanduser()
        downloads_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(downloads_dir / f"{coll['name']}.csv")
    else:
        output_path = str(Path(output_path).expanduser())

    # Write file
    with open(output_path, 'w', newline='') as f:
        f.write(output.getvalue())

    # Log research completion summary to memory
    if agent and row_count > 0:
        summary = f"Research complete: Exported {row_count} items from '{coll['name']}' collection to {output_path}. Columns: {', '.join(export_cols[:5])}{'...' if len(export_cols) > 5 else ''}."
        _log_to_memory(
            agent,
            content=summary,
            event_type="research_summary",
            metadata={"collection": coll["name"], "row_count": row_count, "output_path": output_path}
        )

    return {
        "exported": output_path,
        "collection": coll["name"],
        "rows": row_count,
        "columns": export_cols,
        "filter_status": filter_status
    }


# ═══════════════════════════════════════════════════════════
# UPDATE COLLECTION ITEM - Update item data after enrichment
# ═══════════════════════════════════════════════════════════


@tool()
def update_collection_item(
    collection_name: str = None,
    collection_id: str = None,
    item_id: str = None,
    key_value: str = None,
    updates: dict = None,
    status: str = None,
    agent=None
) -> dict:
    """
    Update a collection item's data (e.g., after enrichment research).

    Use this to fill in columns for an item after researching it.

    Args:
        collection_name: Collection name
        collection_id: Or collection ID
        item_id: Item ID to update
        key_value: Or find by key value (e.g., company name)
        updates: Dict of field updates to merge into item data
        status: New status (pending, processing, enriched, error)

    Example - After researching a VC:
        update_collection_item(
            collection_name="vc_list",
            key_value="sequoia",
            updates={"focus": "Enterprise, AI, Consumer", "stage": "Series A-D", "partners": 15},
            status="enriched"
        )
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    coll = _get_collection(cur, collection_name, collection_id)
    if not coll:
        return tool_error("Collection not found")

    # Find the item
    if item_id:
        cur.execute("""
            SELECT * FROM collection_items WHERE id = ? AND collection_id = ?
        """, (item_id, coll["id"]))
    elif key_value:
        cur.execute("""
            SELECT * FROM collection_items WHERE key_value = ? AND collection_id = ?
        """, (key_value.lower().strip(), coll["id"]))
    else:
        return tool_error("item_id or key_value required")

    row = cur.fetchone()
    if not row:
        return tool_error("Item not found")

    # Merge updates into existing data
    item_data = json.loads(row["data"])
    if updates:
        item_data.update(updates)

    # Build update query
    update_parts = ["data = ?", "updated_at = ?"]
    params = [json.dumps(item_data), now]

    if status:
        update_parts.append("status = ?")
        params.append(status)
        if status == "enriched":
            update_parts.append("enriched_at = ?")
            params.append(now)

    params.append(row["id"])

    cur.execute(f"""
        UPDATE collection_items
        SET {", ".join(update_parts)}
        WHERE id = ?
    """, params)
    conn.commit()

    # Log enrichment to memory for extraction into searchable knowledge
    if status == "enriched" and updates and agent:
        item_name = item_data.get(coll["key_field"], row["key_value"])
        content = _format_enrichment_for_memory(item_name, coll["name"], updates)
        _log_to_memory(
            agent,
            content=content,
            event_type="research_finding",
            metadata={"collection": coll["name"], "item_id": row["id"]}
        )

    return {
        "updated": row["id"],
        "key_value": row["key_value"],
        "new_data": item_data,
        "status": status or row["status"]
    }


# ═══════════════════════════════════════════════════════════
# BATCH ITERATOR - Helper for processing items one by one
# ═══════════════════════════════════════════════════════════


@tool()
def batch_next(
    collection_name: str = None,
    collection_id: str = None,
    batch_size: int = 1,
    from_status: str = "pending",
    to_status: str = "processing",
    agent=None
) -> dict:
    """
    Get the next item(s) from a collection for processing.

    Atomically fetches item(s) and updates status to prevent double-processing.
    Use batch_size > 1 for parallel processing.

    Args:
        collection_name: Collection name
        collection_id: Or collection ID
        batch_size: Number of items to fetch (default: 1, max: 100)
        from_status: Only get items with this status (default: pending)
        to_status: Update items to this status (default: processing)

    Returns:
        - If batch_size=1: Single item dict with _id, or {"done": True}
        - If batch_size>1: {"items": [...], "count": N, "done": False/True}

    Example - Single item (sequential):
        while True:
            item = batch_next(collection_name="vc_list")
            if item.get("done"):
                break
            # Process item...

    Example - Batch of 5 (parallel):
        while True:
            batch = batch_next(collection_name="vc_list", batch_size=5)
            if batch.get("done"):
                break
            for item in batch["items"]:
                # Process item (can be parallelized)...
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    coll = _get_collection(cur, collection_name, collection_id)
    if not coll:
        return tool_error("Collection not found")

    # Clamp batch_size
    batch_size = max(1, min(batch_size, 100))

    # Get next pending item(s)
    cur.execute("""
        SELECT * FROM collection_items
        WHERE collection_id = ? AND status = ?
        ORDER BY created_at
        LIMIT ?
    """, (coll["id"], from_status, batch_size))

    rows = cur.fetchall()
    if not rows:
        # Get stats
        cur.execute("""
            SELECT status, COUNT(*) as count
            FROM collection_items
            WHERE collection_id = ?
            GROUP BY status
        """, (coll["id"],))
        status_counts = {r["status"]: r["count"] for r in cur.fetchall()}

        return {
            "done": True,
            "collection": coll["name"],
            "status_counts": status_counts,
            "message": f"No more items with status '{from_status}'"
        }

    # Update status for all fetched items atomically
    item_ids = [row["id"] for row in rows]
    placeholders = ",".join("?" * len(item_ids))
    cur.execute(f"""
        UPDATE collection_items
        SET status = ?, updated_at = ?
        WHERE id IN ({placeholders})
    """, [to_status, now] + item_ids)
    conn.commit()

    # Build items list
    items = []
    for row in rows:
        item_data = json.loads(row["data"])
        item_data["_id"] = row["id"]
        item_data["_status"] = to_status
        item_data["_source"] = row["source"]
        items.append(item_data)

    # Add progress info
    cur.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status = 'enriched' THEN 1 ELSE 0 END) as enriched,
               SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending
        FROM collection_items
        WHERE collection_id = ?
    """, (coll["id"],))
    stats = cur.fetchone()

    progress = {
        "total": stats["total"],
        "enriched": stats["enriched"],
        "pending": stats["pending"],
        "percent": round((stats["enriched"] / stats["total"]) * 100, 1) if stats["total"] > 0 else 0
    }

    # Return format depends on batch_size
    if batch_size == 1:
        # Single item - backward compatible flat format
        return {
            **items[0],
            "_progress": progress
        }
    else:
        # Multiple items - return list
        return {
            "items": items,
            "count": len(items),
            "done": False,
            "_progress": progress
        }


# ═══════════════════════════════════════════════════════════
# RESEARCH PROGRESS - Overview of all research tasks
# ═══════════════════════════════════════════════════════════


@tool()
def research_progress(agent=None) -> dict:
    """
    Get an overview of all research collections and their progress.

    Returns summary of all collections with:
    - Item counts and completion percentages
    - Active checkpoints
    - Rate limiter status

    Use this to monitor long-running research tasks.
    """
    conn = _get_research_db()
    cur = conn.cursor()

    # Get all collections with stats
    cur.execute("""
        SELECT c.*,
               (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id AND ci.status = 'enriched') as enriched_count,
               (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id AND ci.status = 'pending') as pending_count,
               (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id AND ci.status = 'processing') as processing_count,
               (SELECT COUNT(*) FROM collection_items ci WHERE ci.collection_id = c.id AND ci.status = 'error') as error_count
        FROM collections c
        ORDER BY c.updated_at DESC
    """)

    collections = []
    for row in cur.fetchall():
        total = row["item_count"] or 0
        enriched = row["enriched_count"] or 0
        pct = round((enriched / total) * 100, 1) if total > 0 else 0

        collections.append({
            "name": row["name"],
            "total": total,
            "enriched": enriched,
            "pending": row["pending_count"] or 0,
            "processing": row["processing_count"] or 0,
            "errors": row["error_count"] or 0,
            "percent_complete": pct,
            "updated_at": row["updated_at"]
        })

    # Get active checkpoints
    cur.execute("""
        SELECT cp.*, c.name as collection_name
        FROM checkpoints cp
        LEFT JOIN collections c ON cp.collection_id = c.id
        ORDER BY cp.updated_at DESC
    """)
    checkpoints = []
    for row in cur.fetchall():
        checkpoints.append({
            "task_id": row["task_id"],
            "collection": row["collection_name"],
            "phase": row["phase"],
            "progress": json.loads(row["progress"]),
            "updated_at": row["updated_at"]
        })

    # Get rate limiter status
    cur.execute("SELECT * FROM rate_limits")
    rate_limits = []
    for row in cur.fetchall():
        rate_limits.append({
            "name": row["name"],
            "calls_per_minute": row["calls_per_minute"],
            "calls_this_minute": row["call_count_this_minute"] or 0,
            "last_call_at": row["last_call_at"]
        })

    return {
        "collections": collections,
        "checkpoints": checkpoints,
        "rate_limits": rate_limits,
        "summary": {
            "total_collections": len(collections),
            "active_checkpoints": len(checkpoints),
            "total_items": sum(c["total"] for c in collections),
            "total_enriched": sum(c["enriched"] for c in collections)
        }
    }


# ═══════════════════════════════════════════════════════════
# PACE WORK - Yield to higher priority tasks
# ═══════════════════════════════════════════════════════════


@tool()
def pace_work(
    delay_seconds: float = 1.0,
    every_n_items: int = 1,
    current_item: int = 0,
    check_higher_priority: bool = True,
    agent=None
) -> dict:
    """
    Pace long-running work to be a good citizen.

    Call this between iterations of batch work to:
    1. Add delay to avoid hammering APIs
    2. Allow other work to run (yields to event loop)
    3. Check if higher-priority work is waiting

    For low-priority research tasks (priority 8-10), this ensures
    the agent remains responsive to user requests.

    Args:
        delay_seconds: Seconds to wait between items (default: 1.0)
        every_n_items: Only delay every N items (default: 1 = every item)
        current_item: Current item number in batch (for every_n_items)
        check_higher_priority: If True, checks for waiting higher-priority work

    Returns:
        {"continued": True} normally, or info about yielding

    Example:
        for i, item in enumerate(items):
            pace_work(delay_seconds=2, every_n_items=5, current_item=i)
            # ... process item ...
    """
    result = {"continued": True, "item": current_item}

    # Check if we should delay this iteration
    should_delay = (current_item % every_n_items == 0) if every_n_items > 1 else True

    if should_delay and delay_seconds > 0:
        time.sleep(delay_seconds)
        result["delayed"] = delay_seconds

    # Check for higher priority work (if agent available)
    if check_higher_priority and agent:
        try:
            # Check objective queue for higher priority waiting work
            if hasattr(agent, '_objective_queue') and hasattr(agent, 'objectives'):
                # Find lowest (best) priority in queue
                waiting_priorities = []
                for priority, _, obj_id in agent._objective_queue:
                    obj = agent.objectives.get(obj_id)
                    if obj and obj.status == "pending":
                        waiting_priorities.append(priority)

                if waiting_priorities:
                    best_waiting = min(waiting_priorities)
                    result["higher_priority_waiting"] = best_waiting
                    result["waiting_count"] = len(waiting_priorities)
        except Exception:
            pass  # Don't fail pacing due to check errors

    return result


# ═══════════════════════════════════════════════════════════
# RESET COLLECTION ITEMS - Reset items for reprocessing
# ═══════════════════════════════════════════════════════════


@tool()
def reset_collection_items(
    collection_name: str = None,
    collection_id: str = None,
    from_status: str = None,
    to_status: str = "pending",
    clear_errors: bool = True,
    agent=None
) -> dict:
    """
    Reset collection items to a previous status for reprocessing.

    Useful when:
    - Items failed and need to be retried
    - Processing was interrupted (items stuck in 'processing')
    - You want to re-enrich items with updated logic

    Args:
        collection_name: Collection name
        collection_id: Or collection ID
        from_status: Reset items with this status (None = all non-enriched)
        to_status: New status to set (default: pending)
        clear_errors: Clear error messages (default: True)

    Example - Reset failed items:
        reset_collection_items(collection_name="vc_list", from_status="error")

    Example - Reset stuck processing items:
        reset_collection_items(collection_name="vc_list", from_status="processing")
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    coll = _get_collection(cur, collection_name, collection_id)
    if not coll:
        return tool_error("Collection not found")

    # Build update query
    if from_status:
        where_clause = "collection_id = ? AND status = ?"
        params = [coll["id"], from_status]
    else:
        where_clause = "collection_id = ? AND status != 'enriched'"
        params = [coll["id"]]

    # Count before
    cur.execute(f"SELECT COUNT(*) as count FROM collection_items WHERE {where_clause}", params)
    count = cur.fetchone()["count"]

    if count == 0:
        return {
            "collection": coll["name"],
            "reset_count": 0,
            "message": f"No items found with status '{from_status}'" if from_status else "No non-enriched items found"
        }

    # Build update
    update_parts = ["status = ?", "updated_at = ?"]
    update_params = [to_status, now]

    if clear_errors:
        update_parts.append("error = NULL")
        update_parts.append("retry_count = 0")

    # Execute update
    cur.execute(f"""
        UPDATE collection_items
        SET {", ".join(update_parts)}
        WHERE {where_clause}
    """, update_params + params)
    conn.commit()

    return {
        "collection": coll["name"],
        "reset_count": count,
        "from_status": from_status or "all non-enriched",
        "to_status": to_status,
        "message": f"Reset {count} items to '{to_status}'"
    }


# ═══════════════════════════════════════════════════════════
# IMPORT COLLECTION - Bulk import from CSV or list
# ═══════════════════════════════════════════════════════════


@tool()
def import_collection(
    name: str,
    csv_path: str = None,
    items: list = None,
    key_field: str = None,
    source: str = None,
    create_if_missing: bool = True,
    agent=None
) -> dict:
    """
    Bulk import data into a collection from CSV file or list of dicts.

    For CRM pagination, importing existing data, or loading from files.
    Auto-detects schema from first item if creating new collection.

    Args:
        name: Collection name
        csv_path: Path to CSV file to import
        items: List of dicts to import (alternative to csv_path)
        key_field: Field for deduplication (required if creating new collection)
        source: Source label for tracking (e.g., "csv_import", "hubspot_api")
        create_if_missing: Create collection if it doesn't exist (default: True)

    Example - Import from CSV:
        import_collection(name="contacts", csv_path="~/Downloads/contacts.csv",
                         key_field="email", source="csv_import")

    Example - Import CRM page:
        import_collection(name="contacts", items=crm_response["contacts"],
                         key_field="email", source="hubspot_page_1")
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    # Load items from CSV if path provided
    if csv_path:
        csv_path = str(Path(csv_path).expanduser())
        if not os.path.exists(csv_path):
            return tool_error(f"CSV file not found: {csv_path}")

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            items = list(reader)

        if not source:
            source = f"csv:{Path(csv_path).name}"

    if not items:
        return tool_error("No items to import", fix="Provide csv_path or items")

    # Check if collection exists
    coll = _get_collection(cur, name)

    if not coll:
        if not create_if_missing:
            return tool_error(f"Collection '{name}' not found",
                            fix="Set create_if_missing=True or create collection first")

        if not key_field:
            return tool_error("key_field required when creating new collection")

        # Auto-detect schema from first item
        first_item = items[0]
        schema = {k: "string" for k in first_item.keys()}

        if key_field not in schema:
            return tool_error(f"key_field '{key_field}' not found in data")

        # Create collection
        coll_id = str(uuid4())[:8]
        cur.execute("""
            INSERT INTO collections (id, name, schema, key_field, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (coll_id, name, json.dumps(schema), key_field, now, now))

        coll = {"id": coll_id, "name": name, "key_field": key_field}
        created_collection = True
    else:
        created_collection = False
        key_field = coll["key_field"]

    # Import items
    added = 0
    skipped = 0
    errors = []

    for itm in items:
        if key_field not in itm:
            errors.append(f"Missing key_field '{key_field}'")
            continue

        key_value = str(itm[key_field]).lower().strip()
        item_id = str(uuid4())[:8]

        try:
            cur.execute("""
                INSERT INTO collection_items (id, collection_id, key_value, data, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (item_id, coll["id"], key_value, json.dumps(itm), source, now, now))
            added += 1
        except sqlite3.IntegrityError:
            skipped += 1

    # Update collection count
    cur.execute("""
        UPDATE collections
        SET item_count = (SELECT COUNT(*) FROM collection_items WHERE collection_id = ?),
            updated_at = ?
        WHERE id = ?
    """, (coll["id"], now, coll["id"]))
    conn.commit()

    cur.execute("SELECT item_count FROM collections WHERE id = ?", (coll["id"],))
    total = cur.fetchone()["item_count"]

    result = {
        "collection": name,
        "created_collection": created_collection,
        "imported": added,
        "skipped_duplicates": skipped,
        "total_items": total,
        "source": source
    }
    if errors:
        result["errors"] = errors[:10]  # Limit error messages
        result["error_count"] = len(errors)

    # Log import to memory for awareness
    if agent and added > 0:
        summary = f"Imported {added} items into '{name}' collection from {source or 'unknown source'}. Total items now: {total}."
        _log_to_memory(
            agent,
            content=summary,
            event_type="research_import",
            metadata={"collection": name, "imported": added, "source": source}
        )

    return result


# ═══════════════════════════════════════════════════════════
# CURSOR STATE - For API pagination (CRM, etc.)
# ═══════════════════════════════════════════════════════════


@tool()
def cursor_state(
    action: str,
    name: str,
    cursor_value: str = None,
    page_number: int = None,
    total_fetched: int = None,
    has_more: bool = None,
    metadata: dict = None,
    agent=None
) -> dict:
    """
    Manage pagination cursor state for APIs (CRM, etc.).

    Use this to track position when paginating through large datasets.
    Persists across restarts so you can resume where you left off.

    Actions:
    - save: Save current cursor/page state
    - load: Load saved cursor state
    - delete: Clear cursor state (when done)

    Args:
        action: save, load, delete
        name: Cursor identifier (e.g., "hubspot_contacts", "salesforce_leads")
        cursor_value: The cursor/token from API response
        page_number: Current page number (for offset-based pagination)
        total_fetched: Running total of items fetched
        has_more: Whether more pages exist
        metadata: Additional state to track (e.g., {"last_id": "abc123"})

    Example - Cursor-based pagination (HubSpot style):
        # Check for existing cursor
        state = cursor_state(action="load", name="hubspot_contacts")

        cursor = state.get("cursor_value") if state["found"] else None
        while True:
            response = hubspot.get_contacts(after=cursor)

            # Import page
            import_collection(name="contacts", items=response["results"],
                            source=f"hubspot_page_{state.get('page_number', 0)}")

            # Save cursor state
            cursor_state(action="save", name="hubspot_contacts",
                        cursor_value=response.get("paging", {}).get("next", {}).get("after"),
                        page_number=state.get("page_number", 0) + 1,
                        total_fetched=state.get("total_fetched", 0) + len(response["results"]),
                        has_more=response.get("paging") is not None)

            if not response.get("paging"):
                break
            cursor = response["paging"]["next"]["after"]

        # Clean up when done
        cursor_state(action="delete", name="hubspot_contacts")
    """
    conn = _get_research_db()
    cur = conn.cursor()
    now = datetime.now().isoformat()

    if action == "save":
        # Upsert cursor state
        cur.execute("""
            INSERT INTO cursor_states (id, name, cursor_value, page_number, total_fetched, has_more, metadata, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET
                cursor_value = excluded.cursor_value,
                page_number = COALESCE(excluded.page_number, cursor_states.page_number),
                total_fetched = COALESCE(excluded.total_fetched, cursor_states.total_fetched),
                has_more = COALESCE(excluded.has_more, cursor_states.has_more),
                metadata = COALESCE(excluded.metadata, cursor_states.metadata),
                updated_at = excluded.updated_at
        """, (
            str(uuid4())[:8],
            name,
            cursor_value,
            page_number,
            total_fetched,
            1 if has_more else 0 if has_more is not None else None,
            json.dumps(metadata) if metadata else None,
            now,
            now
        ))
        conn.commit()

        return {
            "saved": name,
            "cursor_value": cursor_value,
            "page_number": page_number,
            "total_fetched": total_fetched,
            "has_more": has_more
        }

    elif action == "load":
        cur.execute("SELECT * FROM cursor_states WHERE name = ?", (name,))
        row = cur.fetchone()

        if not row:
            return {"found": False, "name": name}

        return {
            "found": True,
            "name": name,
            "cursor_value": row["cursor_value"],
            "page_number": row["page_number"],
            "total_fetched": row["total_fetched"],
            "has_more": bool(row["has_more"]),
            "metadata": json.loads(row["metadata"]) if row["metadata"] else None,
            "updated_at": row["updated_at"]
        }

    elif action == "delete":
        cur.execute("DELETE FROM cursor_states WHERE name = ?", (name,))
        conn.commit()
        return {"deleted": name}

    return tool_error(f"Unknown action: {action}")
