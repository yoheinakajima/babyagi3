"""
Memory store - database operations for the memory system.

Uses SQLite with sqlite-vec for vector search.
"""

import json
import os
import sqlite3
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    AgentState,
    Credential,
    Edge,
    Entity,
    Event,
    EventTopic,
    Learning,
    SummaryNode,
    Task,
    ToolDefinition,
    ToolRecord,
    Topic,
)

# Vector dimensions
EMBEDDING_DIM = 1536


# ═══════════════════════════════════════════════════════════
# EVENT RETENTION POLICY
# ═══════════════════════════════════════════════════════════


@dataclass
class RetentionPolicy:
    """Configuration for event retention."""

    # Maximum age for events (None = keep forever)
    max_age_days: int | None = 365

    # Maximum number of events (None = no limit)
    max_events: int | None = 100000

    # Keep important events longer
    important_event_types: list[str] | None = None  # e.g., ["task_completed", "observation"]
    important_multiplier: float = 3.0  # Keep important events 3x longer

    # Never delete events with these properties
    preserve_with_entities: bool = True  # Keep events linked to entities
    preserve_owner_events: bool = True  # Keep owner's events

    # Batch size for cleanup
    cleanup_batch_size: int = 1000


def serialize_embedding(embedding: list[float] | None) -> bytes | None:
    """Serialize embedding to bytes for SQLite storage."""
    if embedding is None:
        return None
    return struct.pack(f"{len(embedding)}f", *embedding)


def deserialize_embedding(data: bytes | None) -> list[float] | None:
    """Deserialize embedding from bytes."""
    if data is None:
        return None
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


def serialize_json(obj: Any) -> str | None:
    """Serialize object to JSON string."""
    if obj is None:
        return None
    return json.dumps(obj)


def deserialize_json(data: str | None) -> Any:
    """Deserialize JSON string to object."""
    if data is None:
        return None
    return json.loads(data)


def generate_id() -> str:
    """Generate a unique ID."""
    return str(uuid4())


def now_iso() -> str:
    """Get current time as ISO string."""
    return datetime.now().isoformat()


def parse_datetime(s: str | None) -> datetime | None:
    """Parse ISO datetime string."""
    if s is None:
        return None
    return datetime.fromisoformat(s)


class MemoryStore:
    """
    SQLite-based storage for the memory system.
    """

    def __init__(self, store_path: str = "~/.babyagi/memory"):
        self.store_path = Path(store_path).expanduser()
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.db_path = self.store_path / "memory.db"
        self._conn: sqlite3.Connection | None = None
        self._vec_available = False

    @property
    def conn(self) -> sqlite3.Connection:
        """Get database connection (lazy initialization)."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # Try to load sqlite-vec extension
            try:
                self._conn.enable_load_extension(True)
                # Try common locations for sqlite-vec
                for ext_path in [
                    "vec0",
                    "/usr/local/lib/vec0",
                    "/usr/lib/vec0",
                    str(self.store_path / "vec0"),
                ]:
                    try:
                        self._conn.load_extension(ext_path)
                        self._vec_available = True
                        break
                    except sqlite3.OperationalError:
                        continue
            except Exception:
                pass  # Extension loading not available
        return self._conn

    def initialize(self):
        """Initialize the database schema."""
        self._create_tables()
        self._create_indices()
        self._migrate_tool_definitions()
        self._ensure_root_node()
        self._ensure_agent_state()

    def _migrate_tool_definitions(self):
        """Add new columns to tool_definitions for skills and composio support."""
        cur = self.conn.cursor()

        # Get existing columns
        cur.execute("PRAGMA table_info(tool_definitions)")
        existing_columns = {row["name"] for row in cur.fetchall()}

        # Add new columns if they don't exist
        migrations = [
            ("tool_type", "TEXT DEFAULT 'executable'"),
            ("skill_content", "TEXT"),
            ("composio_app", "TEXT"),
            ("composio_action", "TEXT"),
            ("depends_on", "TEXT"),
        ]

        for col_name, col_def in migrations:
            if col_name not in existing_columns:
                try:
                    cur.execute(f"ALTER TABLE tool_definitions ADD COLUMN {col_name} {col_def}")
                except sqlite3.OperationalError:
                    pass  # Column might already exist

        self.conn.commit()

    def _create_tables(self):
        """Create all tables."""
        cur = self.conn.cursor()

        # Events
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                channel TEXT,
                direction TEXT NOT NULL,
                event_type TEXT NOT NULL,
                task_id TEXT,
                tool_id TEXT,
                person_id TEXT,
                is_owner INTEGER NOT NULL DEFAULT 0,
                parent_event_id TEXT,
                conversation_id TEXT,
                content TEXT NOT NULL,
                content_embedding BLOB,
                metadata TEXT,
                extraction_status TEXT DEFAULT 'pending',
                extracted_at TEXT,
                created_at TEXT NOT NULL
            )
        """
        )

        # Entities
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                type_raw TEXT NOT NULL,
                aliases TEXT,
                description TEXT,
                name_embedding BLOB,
                is_owner INTEGER DEFAULT 0,
                is_self INTEGER DEFAULT 0,
                event_count INTEGER DEFAULT 0,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                source_event_ids TEXT,
                summary_node_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Edges
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS edges (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                relation_type TEXT,
                relation_embedding BLOB,
                is_current INTEGER DEFAULT 1,
                strength REAL DEFAULT 0.5,
                source_event_ids TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id)
            )
        """
        )

        # Topics
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                description TEXT,
                keywords TEXT,
                embedding BLOB,
                parent_topic_id TEXT,
                event_count INTEGER DEFAULT 0,
                entity_count INTEGER DEFAULT 0,
                summary_node_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (parent_topic_id) REFERENCES topics(id)
            )
        """
        )

        # Event-Topic junction
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS event_topics (
                event_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                relevance REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                PRIMARY KEY (event_id, topic_id),
                FOREIGN KEY (event_id) REFERENCES events(id),
                FOREIGN KEY (topic_id) REFERENCES topics(id)
            )
        """
        )

        # Tasks
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                type_raw TEXT,
                type_cluster TEXT,
                type_embedding BLOB,
                status TEXT DEFAULT 'pending',
                outcome TEXT,
                person_id TEXT,
                created_by_event_id TEXT,
                summary_node_id TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (person_id) REFERENCES entities(id),
                FOREIGN KEY (created_by_event_id) REFERENCES events(id)
            )
        """
        )

        # Summary Nodes
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS summary_nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                label TEXT NOT NULL,
                parent_id TEXT,
                summary TEXT,
                summary_embedding BLOB,
                summary_updated_at TEXT,
                events_since_update INTEGER DEFAULT 0,
                event_count INTEGER DEFAULT 0,
                first_event_at TEXT,
                last_event_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES summary_nodes(id)
            )
        """
        )

        # Tools (legacy - kept for backward compatibility)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                description_embedding BLOB,
                usage_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                summary_node_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """
        )

        # Tool Definitions - full tool persistence for self-improvement
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_definitions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT NOT NULL,

                -- Tool type: "executable", "skill", "composio"
                tool_type TEXT DEFAULT 'executable',

                -- Definition (what makes it executable)
                source_code TEXT,
                parameters TEXT,
                packages TEXT,
                env TEXT,
                tool_var_name TEXT,

                -- For skills
                skill_content TEXT,

                -- For composio tools
                composio_app TEXT,
                composio_action TEXT,

                -- Dependencies (JSON list of tool names this depends on)
                depends_on TEXT,

                -- Category
                category TEXT DEFAULT 'custom',

                -- State
                is_enabled INTEGER DEFAULT 1,
                is_dynamic INTEGER DEFAULT 1,

                -- Execution statistics
                usage_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                last_error TEXT,
                last_error_at TEXT,
                avg_duration_ms REAL DEFAULT 0,
                total_duration_ms REAL DEFAULT 0,

                -- Graph integration
                entity_id TEXT,
                summary_node_id TEXT,

                -- Versioning
                version INTEGER DEFAULT 1,

                -- Provenance
                created_by_event_id TEXT,

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY (entity_id) REFERENCES entities(id),
                FOREIGN KEY (summary_node_id) REFERENCES summary_nodes(id)
            )
        """
        )

        # Agent State
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_state (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                owner_entity_id TEXT,
                self_entity_id TEXT,
                current_topics TEXT,
                mood TEXT,
                focus TEXT,
                active_tasks TEXT,
                settings TEXT,
                state_updated_at TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (owner_entity_id) REFERENCES entities(id),
                FOREIGN KEY (self_entity_id) REFERENCES entities(id)
            )
        """
        )

        # Secure Credentials - for user accounts, credit cards, etc.
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS credentials (
                id TEXT PRIMARY KEY,
                credential_type TEXT NOT NULL,
                service TEXT NOT NULL,

                -- For user accounts
                username TEXT,
                email TEXT,
                password_ref TEXT,

                -- For credit cards
                card_last_four TEXT,
                card_type TEXT,
                card_expiry TEXT,
                card_ref TEXT,
                billing_name TEXT,
                billing_address TEXT,

                -- Common fields
                notes TEXT,
                metadata TEXT,

                -- Timestamps
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT
            )
        """
        )

        # Metrics tables
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_calls (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                model TEXT NOT NULL,
                thread_id TEXT,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                stop_reason TEXT
            )
        """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_calls (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                text_count INTEGER NOT NULL,
                token_estimate INTEGER NOT NULL,
                cost_usd REAL NOT NULL,
                duration_ms INTEGER NOT NULL,
                cached INTEGER NOT NULL DEFAULT 0
            )
        """
        )

        # Learnings table - for self-improvement system
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS learnings (
                id TEXT PRIMARY KEY,

                -- Source
                source_type TEXT NOT NULL,
                source_event_id TEXT,

                -- Content
                content TEXT NOT NULL,
                content_embedding BLOB,

                -- Classification
                sentiment TEXT NOT NULL DEFAULT 'neutral',
                confidence REAL DEFAULT 0.5,

                -- Associations
                tool_id TEXT,
                topic_ids TEXT,
                objective_type TEXT,
                entity_ids TEXT,

                -- Actionable insight
                applies_when TEXT,
                recommendation TEXT,

                -- Stats
                times_applied INTEGER DEFAULT 0,
                last_applied_at TEXT,

                -- Timestamps
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,

                FOREIGN KEY (source_event_id) REFERENCES events(id)
            )
        """
        )

        self.conn.commit()

    def _create_indices(self):
        """Create database indices."""
        cur = self.conn.cursor()

        indices = [
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_channel ON events(channel, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_person ON events(person_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_tool ON events(tool_id, timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_events_extraction ON events(extraction_status)",
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)",
            "CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type)",
            "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_edges_relation_type ON edges(relation_type)",
            "CREATE INDEX IF NOT EXISTS idx_topics_label ON topics(label)",
            "CREATE INDEX IF NOT EXISTS idx_topics_parent ON topics(parent_topic_id)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_type_cluster ON tasks(type_cluster)",
            "CREATE INDEX IF NOT EXISTS idx_summary_nodes_key ON summary_nodes(key)",
            "CREATE INDEX IF NOT EXISTS idx_summary_nodes_type ON summary_nodes(node_type)",
            "CREATE INDEX IF NOT EXISTS idx_summary_nodes_parent ON summary_nodes(parent_id)",
            "CREATE INDEX IF NOT EXISTS idx_summary_nodes_stale ON summary_nodes(events_since_update DESC)",
            # Tool definitions indices
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_name ON tool_definitions(name)",
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_enabled ON tool_definitions(is_enabled)",
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_dynamic ON tool_definitions(is_dynamic)",
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_category ON tool_definitions(category)",
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_error_count ON tool_definitions(error_count DESC)",
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_tool_type ON tool_definitions(tool_type)",
            "CREATE INDEX IF NOT EXISTS idx_tool_definitions_composio_app ON tool_definitions(composio_app)",
            # Credentials indices
            "CREATE INDEX IF NOT EXISTS idx_credentials_service ON credentials(service)",
            "CREATE INDEX IF NOT EXISTS idx_credentials_type ON credentials(credential_type)",
            # Metrics indices
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_timestamp ON llm_calls(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_source ON llm_calls(source)",
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_model ON llm_calls(model)",
            "CREATE INDEX IF NOT EXISTS idx_llm_calls_thread ON llm_calls(thread_id)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_calls_timestamp ON embedding_calls(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_embedding_calls_model ON embedding_calls(model)",
            # Learnings indices
            "CREATE INDEX IF NOT EXISTS idx_learnings_tool ON learnings(tool_id)",
            "CREATE INDEX IF NOT EXISTS idx_learnings_objective_type ON learnings(objective_type)",
            "CREATE INDEX IF NOT EXISTS idx_learnings_sentiment ON learnings(sentiment)",
            "CREATE INDEX IF NOT EXISTS idx_learnings_source_type ON learnings(source_type)",
            "CREATE INDEX IF NOT EXISTS idx_learnings_created_at ON learnings(created_at DESC)",
        ]

        for idx in indices:
            cur.execute(idx)

        self.conn.commit()

    def _ensure_root_node(self):
        """Ensure the root summary node exists."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM summary_nodes WHERE key = 'root'")
        if cur.fetchone() is None:
            now = now_iso()
            cur.execute(
                """
                INSERT INTO summary_nodes (id, node_type, key, label, summary, created_at, updated_at)
                VALUES (?, 'root', 'root', 'Knowledge', 'No information yet.', ?, ?)
            """,
                (generate_id(), now, now),
            )
            self.conn.commit()

    def _ensure_agent_state(self):
        """Ensure the agent state record exists."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM agent_state LIMIT 1")
        if cur.fetchone() is None:
            now = now_iso()
            cur.execute(
                """
                INSERT INTO agent_state (id, name, settings, created_at)
                VALUES (?, 'Agent', '{}', ?)
            """,
                (generate_id(), now),
            )
            self.conn.commit()

    # ═══════════════════════════════════════════════════════════
    # EVENTS
    # ═══════════════════════════════════════════════════════════

    def create_event(
        self,
        content: str,
        event_type: str = "message",
        channel: str | None = None,
        direction: str = "internal",
        task_id: str | None = None,
        tool_id: str | None = None,
        person_id: str | None = None,
        is_owner: bool = False,
        parent_event_id: str | None = None,
        conversation_id: str | None = None,
        metadata: dict | None = None,
        content_embedding: list[float] | None = None,
    ) -> Event:
        """Create a new event."""
        event_id = generate_id()
        now = now_iso()

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO events (
                id, timestamp, channel, direction, event_type, task_id, tool_id,
                person_id, is_owner, parent_event_id, conversation_id, content,
                content_embedding, metadata, extraction_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
        """,
            (
                event_id,
                now,
                channel,
                direction,
                event_type,
                task_id,
                tool_id,
                person_id,
                1 if is_owner else 0,
                parent_event_id,
                conversation_id,
                content,
                serialize_embedding(content_embedding),
                serialize_json(metadata),
                now,
            ),
        )
        self.conn.commit()

        # Increment staleness on relevant summary nodes
        self._increment_staleness_for_event(channel, tool_id, person_id, task_id)

        return Event(
            id=event_id,
            timestamp=parse_datetime(now),
            channel=channel,
            direction=direction,
            event_type=event_type,
            task_id=task_id,
            tool_id=tool_id,
            person_id=person_id,
            is_owner=is_owner,
            parent_event_id=parent_event_id,
            conversation_id=conversation_id,
            content=content,
            content_embedding=content_embedding,
            metadata=metadata,
            extraction_status="pending",
            created_at=parse_datetime(now),
        )

    def get_event(self, event_id: str) -> Event | None:
        """Get an event by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_event(row)

    def get_recent_events(
        self,
        limit: int = 10,
        channel: str | None = None,
        person_id: str | None = None,
        task_id: str | None = None,
        tool_id: str | None = None,
    ) -> list[Event]:
        """Get recent events with optional filters."""
        cur = self.conn.cursor()

        conditions = []
        params = []

        if channel:
            conditions.append("channel = ?")
            params.append(channel)
        if person_id:
            conditions.append("person_id = ?")
            params.append(person_id)
        if task_id:
            conditions.append("task_id = ?")
            params.append(task_id)
        if tool_id:
            conditions.append("tool_id = ?")
            params.append(tool_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cur.execute(
            f"SELECT * FROM events WHERE {where_clause} ORDER BY timestamp DESC LIMIT ?",
            params,
        )

        return [self._row_to_event(row) for row in cur.fetchall()]

    def get_events_for_entity(self, entity_id: str, limit: int = 50) -> list[Event]:
        """Get events related to an entity."""
        return self.get_recent_events(limit=limit, person_id=entity_id)

    def get_pending_extraction_events(self, limit: int = 100) -> list[Event]:
        """Get events pending extraction."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM events WHERE extraction_status = 'pending' ORDER BY timestamp ASC LIMIT ?",
            (limit,),
        )
        return [self._row_to_event(row) for row in cur.fetchall()]

    def update_event_extraction_status(
        self, event_id: str, status: str, extracted_at: datetime | None = None
    ):
        """Update the extraction status of an event."""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE events SET extraction_status = ?, extracted_at = ? WHERE id = ?",
            (status, extracted_at.isoformat() if extracted_at else None, event_id),
        )
        self.conn.commit()

    def increment_extraction_retry(self, event_id: str) -> int:
        """
        Increment the retry count for an event extraction.

        Returns the new retry count.
        """
        cur = self.conn.cursor()
        # Get current metadata
        cur.execute("SELECT metadata FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        if row is None:
            return 0

        metadata = deserialize_json(row["metadata"]) or {}
        retry_count = metadata.get("extraction_retries", 0) + 1
        metadata["extraction_retries"] = retry_count
        metadata["last_extraction_attempt"] = now_iso()

        cur.execute(
            "UPDATE events SET metadata = ? WHERE id = ?",
            (serialize_json(metadata), event_id),
        )
        self.conn.commit()
        return retry_count

    def get_extraction_retry_count(self, event_id: str) -> int:
        """Get the current retry count for an event."""
        cur = self.conn.cursor()
        cur.execute("SELECT metadata FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        if row is None:
            return 0

        metadata = deserialize_json(row["metadata"]) or {}
        return metadata.get("extraction_retries", 0)

    def get_failed_extraction_events(
        self, max_retries: int = 3, limit: int = 100
    ) -> list[Event]:
        """Get failed events that haven't exceeded max retries."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM events
            WHERE extraction_status = 'failed'
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (limit * 2,),  # Fetch extra to filter
        )

        events = []
        for row in cur.fetchall():
            event = self._row_to_event(row)
            retry_count = (event.metadata or {}).get("extraction_retries", 0)
            if retry_count < max_retries:
                events.append(event)
                if len(events) >= limit:
                    break

        return events

    def update_event_embedding(self, event_id: str, embedding: list[float]):
        """Update the content embedding of an event."""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE events SET content_embedding = ? WHERE id = ?",
            (serialize_embedding(embedding), event_id),
        )
        self.conn.commit()

    def _row_to_event(self, row: sqlite3.Row) -> Event:
        """Convert a database row to an Event."""
        return Event(
            id=row["id"],
            timestamp=parse_datetime(row["timestamp"]),
            channel=row["channel"],
            direction=row["direction"],
            event_type=row["event_type"],
            task_id=row["task_id"],
            tool_id=row["tool_id"],
            person_id=row["person_id"],
            is_owner=bool(row["is_owner"]),
            parent_event_id=row["parent_event_id"],
            conversation_id=row["conversation_id"],
            content=row["content"],
            content_embedding=deserialize_embedding(row["content_embedding"]),
            metadata=deserialize_json(row["metadata"]),
            extraction_status=row["extraction_status"],
            extracted_at=parse_datetime(row["extracted_at"]),
            created_at=parse_datetime(row["created_at"]),
        )

    def _increment_node_staleness(self, key: str, timestamp: str = None):
        """Increment staleness counter for a summary node by key.

        This is the core helper for updating summary node staleness.
        Used by _increment_staleness_for_event to avoid SQL duplication.

        Args:
            key: The summary node key (e.g., "root", "channel:email", "entity:uuid")
            timestamp: ISO timestamp to use, defaults to now
        """
        ts = timestamp or now_iso()
        self.conn.execute(
            """
            UPDATE summary_nodes
            SET events_since_update = events_since_update + 1,
                event_count = event_count + 1,
                last_event_at = ?,
                updated_at = ?
            WHERE key = ?
        """,
            (ts, ts, key),
        )

    def _increment_staleness_for_event(
        self,
        channel: str | None,
        tool_id: str | None,
        person_id: str | None,
        task_id: str | None,
    ):
        """Increment staleness counters for relevant summary nodes."""
        ts = now_iso()

        # Root always gets incremented
        self._increment_node_staleness("root", ts)

        # Optional nodes
        if channel:
            self._increment_node_staleness(f"channel:{channel}", ts)
        if tool_id:
            self._increment_node_staleness(f"tool:{tool_id}", ts)
        if person_id:
            self._increment_node_staleness(f"entity:{person_id}", ts)
        if task_id:
            self._increment_node_staleness(f"task:{task_id}", ts)

        self.conn.commit()

    # ═══════════════════════════════════════════════════════════
    # ENTITIES
    # ═══════════════════════════════════════════════════════════

    def create_entity(
        self,
        name: str,
        type: str,
        type_raw: str,
        aliases: list[str] | None = None,
        description: str | None = None,
        name_embedding: list[float] | None = None,
        is_owner: bool = False,
        is_self: bool = False,
        source_event_ids: list[str] | None = None,
    ) -> Entity:
        """Create a new entity and its summary node."""
        entity_id = generate_id()
        now = now_iso()

        # Create summary node first
        summary_node = self.create_summary_node(
            node_type="entity",
            key=f"entity:{entity_id}",
            label=name,
            parent_key=f"entity_type:{type}",
        )

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO entities (
                id, name, type, type_raw, aliases, description, name_embedding,
                is_owner, is_self, event_count, first_seen, last_seen,
                source_event_ids, summary_node_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
        """,
            (
                entity_id,
                name,
                type,
                type_raw,
                serialize_json(aliases or []),
                description,
                serialize_embedding(name_embedding),
                1 if is_owner else 0,
                1 if is_self else 0,
                now,
                now,
                serialize_json(source_event_ids or []),
                summary_node.id,
                now,
                now,
            ),
        )
        self.conn.commit()

        return Entity(
            id=entity_id,
            name=name,
            type=type,
            type_raw=type_raw,
            aliases=aliases or [],
            description=description,
            name_embedding=name_embedding,
            is_owner=is_owner,
            is_self=is_self,
            event_count=1,
            first_seen=parse_datetime(now),
            last_seen=parse_datetime(now),
            source_event_ids=source_event_ids or [],
            summary_node_id=summary_node.id,
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_entity(row)

    def find_entities(
        self, query: str | None = None, type: str | None = None, limit: int = 10
    ) -> list[Entity]:
        """Find entities by name or type."""
        cur = self.conn.cursor()

        conditions = []
        params = []

        if query:
            conditions.append("(name LIKE ? OR aliases LIKE ?)")
            params.extend([f"%{query}%", f"%{query}%"])
        if type:
            conditions.append("type = ?")
            params.append(type)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cur.execute(
            f"SELECT * FROM entities WHERE {where_clause} ORDER BY event_count DESC LIMIT ?",
            params,
        )

        return [self._row_to_entity(row) for row in cur.fetchall()]

    def update_entity(self, entity: Entity):
        """Update an existing entity."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            UPDATE entities SET
                name = ?, type = ?, type_raw = ?, aliases = ?, description = ?,
                name_embedding = ?, is_owner = ?, is_self = ?, event_count = ?,
                last_seen = ?, source_event_ids = ?, updated_at = ?
            WHERE id = ?
        """,
            (
                entity.name,
                entity.type,
                entity.type_raw,
                serialize_json(entity.aliases),
                entity.description,
                serialize_embedding(entity.name_embedding),
                1 if entity.is_owner else 0,
                1 if entity.is_self else 0,
                entity.event_count,
                entity.last_seen.isoformat() if entity.last_seen else now,
                serialize_json(entity.source_event_ids),
                now,
                entity.id,
            ),
        )
        self.conn.commit()

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        """Convert a database row to an Entity."""
        return Entity(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            type_raw=row["type_raw"],
            aliases=deserialize_json(row["aliases"]) or [],
            description=row["description"],
            name_embedding=deserialize_embedding(row["name_embedding"]),
            is_owner=bool(row["is_owner"]),
            is_self=bool(row["is_self"]),
            event_count=row["event_count"],
            first_seen=parse_datetime(row["first_seen"]),
            last_seen=parse_datetime(row["last_seen"]),
            source_event_ids=deserialize_json(row["source_event_ids"]) or [],
            summary_node_id=row["summary_node_id"],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # EDGES
    # ═══════════════════════════════════════════════════════════

    def create_edge(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation: str,
        relation_type: str | None = None,
        relation_embedding: list[float] | None = None,
        is_current: bool = True,
        strength: float = 0.5,
        source_event_ids: list[str] | None = None,
    ) -> Edge:
        """Create a new edge."""
        edge_id = generate_id()
        now = now_iso()

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO edges (
                id, source_entity_id, target_entity_id, relation, relation_type,
                relation_embedding, is_current, strength, source_event_ids,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                edge_id,
                source_entity_id,
                target_entity_id,
                relation,
                relation_type,
                serialize_embedding(relation_embedding),
                1 if is_current else 0,
                strength,
                serialize_json(source_event_ids or []),
                now,
                now,
            ),
        )
        self.conn.commit()

        return Edge(
            id=edge_id,
            source_entity_id=source_entity_id,
            target_entity_id=target_entity_id,
            relation=relation,
            relation_type=relation_type,
            relation_embedding=relation_embedding,
            is_current=is_current,
            strength=strength,
            source_event_ids=source_event_ids or [],
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def get_edges(
        self, entity_id: str, direction: str = "both", relation_type: str | None = None
    ) -> list[Edge]:
        """Get edges for an entity."""
        cur = self.conn.cursor()

        conditions = ["is_current = 1"]
        params = []

        if direction == "outgoing":
            conditions.append("source_entity_id = ?")
            params.append(entity_id)
        elif direction == "incoming":
            conditions.append("target_entity_id = ?")
            params.append(entity_id)
        else:  # both
            conditions.append("(source_entity_id = ? OR target_entity_id = ?)")
            params.extend([entity_id, entity_id])

        if relation_type:
            conditions.append("relation_type = ?")
            params.append(relation_type)

        where_clause = " AND ".join(conditions)
        cur.execute(f"SELECT * FROM edges WHERE {where_clause}", params)

        return [self._row_to_edge(row) for row in cur.fetchall()]

    def find_edge(
        self, source_id: str, target_id: str, relation: str
    ) -> Edge | None:
        """Find an existing edge between two entities."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM edges
            WHERE source_entity_id = ? AND target_entity_id = ? AND relation = ? AND is_current = 1
        """,
            (source_id, target_id, relation),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_edge(row)

    def update_edge(self, edge: Edge):
        """Update an existing edge."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            UPDATE edges SET
                relation = ?, relation_type = ?, relation_embedding = ?,
                is_current = ?, strength = ?, source_event_ids = ?, updated_at = ?
            WHERE id = ?
        """,
            (
                edge.relation,
                edge.relation_type,
                serialize_embedding(edge.relation_embedding),
                1 if edge.is_current else 0,
                edge.strength,
                serialize_json(edge.source_event_ids),
                now,
                edge.id,
            ),
        )
        self.conn.commit()

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        """Convert a database row to an Edge."""
        return Edge(
            id=row["id"],
            source_entity_id=row["source_entity_id"],
            target_entity_id=row["target_entity_id"],
            relation=row["relation"],
            relation_type=row["relation_type"],
            relation_embedding=deserialize_embedding(row["relation_embedding"]),
            is_current=bool(row["is_current"]),
            strength=row["strength"],
            source_event_ids=deserialize_json(row["source_event_ids"]) or [],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # TOPICS
    # ═══════════════════════════════════════════════════════════

    def create_topic(
        self,
        label: str,
        description: str | None = None,
        keywords: list[str] | None = None,
        embedding: list[float] | None = None,
        parent_topic_id: str | None = None,
    ) -> Topic:
        """Create a new topic and its summary node."""
        topic_id = generate_id()
        now = now_iso()

        # Create summary node
        parent_key = f"topic:{parent_topic_id}" if parent_topic_id else "root"
        summary_node = self.create_summary_node(
            node_type="topic",
            key=f"topic:{topic_id}",
            label=label,
            parent_key=parent_key,
        )

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO topics (
                id, label, description, keywords, embedding, parent_topic_id,
                event_count, entity_count, summary_node_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?)
        """,
            (
                topic_id,
                label,
                description,
                serialize_json(keywords or []),
                serialize_embedding(embedding),
                parent_topic_id,
                summary_node.id,
                now,
                now,
            ),
        )
        self.conn.commit()

        return Topic(
            id=topic_id,
            label=label,
            description=description,
            keywords=keywords or [],
            embedding=embedding,
            parent_topic_id=parent_topic_id,
            event_count=0,
            entity_count=0,
            summary_node_id=summary_node.id,
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def get_topic(self, topic_id: str) -> Topic | None:
        """Get a topic by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM topics WHERE id = ?", (topic_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_topic(row)

    def find_topic_by_label(self, label: str) -> Topic | None:
        """Find a topic by label."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM topics WHERE label = ?", (label,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_topic(row)

    def find_topics(self, query: str | None = None, limit: int = 10) -> list[Topic]:
        """Find topics by label search."""
        cur = self.conn.cursor()
        if query:
            cur.execute(
                "SELECT * FROM topics WHERE label LIKE ? ORDER BY event_count DESC LIMIT ?",
                (f"%{query}%", limit),
            )
        else:
            cur.execute(
                "SELECT * FROM topics ORDER BY event_count DESC LIMIT ?", (limit,)
            )
        return [self._row_to_topic(row) for row in cur.fetchall()]

    def link_event_topic(self, event_id: str, topic_id: str, relevance: float = 1.0):
        """Link an event to a topic."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            INSERT OR REPLACE INTO event_topics (event_id, topic_id, relevance, created_at)
            VALUES (?, ?, ?, ?)
        """,
            (event_id, topic_id, relevance, now),
        )
        # Update topic event count
        cur.execute(
            "UPDATE topics SET event_count = event_count + 1, updated_at = ? WHERE id = ?",
            (now, topic_id),
        )
        # Update topic summary node staleness
        cur.execute(
            """
            UPDATE summary_nodes
            SET events_since_update = events_since_update + 1, updated_at = ?
            WHERE key = ?
        """,
            (now, f"topic:{topic_id}"),
        )
        self.conn.commit()

    def get_events_for_topic(self, topic_id: str, limit: int = 50) -> list[Event]:
        """Get events associated with a topic."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT e.* FROM events e
            JOIN event_topics et ON e.id = et.event_id
            WHERE et.topic_id = ?
            ORDER BY e.timestamp DESC
            LIMIT ?
        """,
            (topic_id, limit),
        )
        return [self._row_to_event(row) for row in cur.fetchall()]

    def _row_to_topic(self, row: sqlite3.Row) -> Topic:
        """Convert a database row to a Topic."""
        return Topic(
            id=row["id"],
            label=row["label"],
            description=row["description"],
            keywords=deserialize_json(row["keywords"]) or [],
            embedding=deserialize_embedding(row["embedding"]),
            parent_topic_id=row["parent_topic_id"],
            event_count=row["event_count"],
            entity_count=row["entity_count"],
            summary_node_id=row["summary_node_id"],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # TASKS
    # ═══════════════════════════════════════════════════════════

    def create_task(
        self,
        title: str,
        description: str | None = None,
        type_raw: str | None = None,
        type_cluster: str | None = None,
        type_embedding: list[float] | None = None,
        person_id: str | None = None,
        created_by_event_id: str | None = None,
    ) -> Task:
        """Create a new task and its summary node."""
        task_id = generate_id()
        now = now_iso()

        # Create summary node
        parent_key = f"task_type:{type_cluster}" if type_cluster else "root"
        summary_node = self.create_summary_node(
            node_type="task",
            key=f"task:{task_id}",
            label=title,
            parent_key=parent_key,
        )

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO tasks (
                id, title, description, type_raw, type_cluster, type_embedding,
                status, person_id, created_by_event_id, summary_node_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
        """,
            (
                task_id,
                title,
                description,
                type_raw,
                type_cluster,
                serialize_embedding(type_embedding),
                person_id,
                created_by_event_id,
                summary_node.id,
                now,
            ),
        )
        self.conn.commit()

        return Task(
            id=task_id,
            title=title,
            description=description,
            type_raw=type_raw,
            type_cluster=type_cluster,
            type_embedding=type_embedding,
            status="pending",
            person_id=person_id,
            created_by_event_id=created_by_event_id,
            summary_node_id=summary_node.id,
            created_at=parse_datetime(now),
        )

    def get_task(self, task_id: str) -> Task | None:
        """Get a task by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_task(row)

    def find_tasks(
        self,
        status: str | None = None,
        type_cluster: str | None = None,
        person_id: str | None = None,
        limit: int = 10,
    ) -> list[Task]:
        """Find tasks by filters."""
        cur = self.conn.cursor()

        conditions = []
        params = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if type_cluster:
            conditions.append("type_cluster = ?")
            params.append(type_cluster)
        if person_id:
            conditions.append("person_id = ?")
            params.append(person_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        cur.execute(
            f"SELECT * FROM tasks WHERE {where_clause} ORDER BY created_at DESC LIMIT ?",
            params,
        )

        return [self._row_to_task(row) for row in cur.fetchall()]

    def update_task(self, task: Task):
        """Update an existing task."""
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE tasks SET
                title = ?, description = ?, type_raw = ?, type_cluster = ?,
                type_embedding = ?, status = ?, outcome = ?, person_id = ?,
                started_at = ?, completed_at = ?
            WHERE id = ?
        """,
            (
                task.title,
                task.description,
                task.type_raw,
                task.type_cluster,
                serialize_embedding(task.type_embedding),
                task.status,
                task.outcome,
                task.person_id,
                task.started_at.isoformat() if task.started_at else None,
                task.completed_at.isoformat() if task.completed_at else None,
                task.id,
            ),
        )
        self.conn.commit()

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert a database row to a Task."""
        return Task(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            type_raw=row["type_raw"],
            type_cluster=row["type_cluster"],
            type_embedding=deserialize_embedding(row["type_embedding"]),
            status=row["status"],
            outcome=row["outcome"],
            person_id=row["person_id"],
            created_by_event_id=row["created_by_event_id"],
            summary_node_id=row["summary_node_id"],
            created_at=parse_datetime(row["created_at"]),
            started_at=parse_datetime(row["started_at"]),
            completed_at=parse_datetime(row["completed_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # SUMMARY NODES
    # ═══════════════════════════════════════════════════════════

    def create_summary_node(
        self,
        node_type: str,
        key: str,
        label: str,
        parent_key: str | None = None,
        summary: str = "",
    ) -> SummaryNode:
        """Create a new summary node."""
        # Ensure parent exists
        parent_id = None
        if parent_key:
            parent = self.get_summary_node(parent_key)
            if parent:
                parent_id = parent.id
            else:
                # Create parent if it doesn't exist (for entity_type, task_type, etc.)
                parent_type = parent_key.split(":")[0]
                parent_label = parent_key.split(":")[1] if ":" in parent_key else parent_key
                parent_node = self.create_summary_node(
                    node_type=parent_type,
                    key=parent_key,
                    label=parent_label.replace("_", " ").title(),
                    parent_key="root",
                )
                parent_id = parent_node.id

        node_id = generate_id()
        now = now_iso()

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR IGNORE INTO summary_nodes (
                id, node_type, key, label, parent_id, summary, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (node_id, node_type, key, label, parent_id, summary, now, now),
        )
        self.conn.commit()

        # If insert was ignored (key already exists), fetch existing
        existing = self.get_summary_node(key)
        if existing:
            return existing

        return SummaryNode(
            id=node_id,
            node_type=node_type,
            key=key,
            label=label,
            parent_id=parent_id,
            summary=summary,
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def get_summary_node(self, key: str) -> SummaryNode | None:
        """Get a summary node by key."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM summary_nodes WHERE key = ?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_summary_node(row)

    def get_summary_node_by_id(self, node_id: str) -> SummaryNode | None:
        """Get a summary node by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM summary_nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_summary_node(row)

    def get_children(self, node_id: str) -> list[SummaryNode]:
        """Get child summary nodes."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM summary_nodes WHERE parent_id = ? ORDER BY label",
            (node_id,),
        )
        return [self._row_to_summary_node(row) for row in cur.fetchall()]

    def get_parent(self, node_id: str) -> SummaryNode | None:
        """Get parent summary node."""
        cur = self.conn.cursor()
        cur.execute("SELECT parent_id FROM summary_nodes WHERE id = ?", (node_id,))
        row = cur.fetchone()
        if row is None or row["parent_id"] is None:
            return None
        return self.get_summary_node_by_id(row["parent_id"])

    def get_stale_nodes(self, threshold: int = 10) -> list[SummaryNode]:
        """Get summary nodes that need refreshing."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM summary_nodes
            WHERE events_since_update >= ?
            ORDER BY
                CASE node_type
                    WHEN 'entity' THEN 1
                    WHEN 'topic' THEN 1
                    WHEN 'task' THEN 1
                    ELSE 2
                END,
                events_since_update DESC
        """,
            (threshold,),
        )
        return [self._row_to_summary_node(row) for row in cur.fetchall()]

    def update_summary_node(self, node: SummaryNode):
        """Update a summary node."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            UPDATE summary_nodes SET
                summary = ?, summary_embedding = ?, summary_updated_at = ?,
                events_since_update = ?, event_count = ?, updated_at = ?
            WHERE id = ?
        """,
            (
                node.summary,
                serialize_embedding(node.summary_embedding),
                node.summary_updated_at.isoformat() if node.summary_updated_at else now,
                node.events_since_update,
                node.event_count,
                now,
                node.id,
            ),
        )
        self.conn.commit()

    def increment_staleness(self, node_id_or_key: str):
        """Increment staleness counter for a node by ID or key."""
        cur = self.conn.cursor()

        # Check if it's a key (contains ':' or is a known key like 'root', 'user_preferences')
        if ":" in node_id_or_key or node_id_or_key in ("root", "user_preferences"):
            # It's a key, look up by key
            cur.execute(
                """
                UPDATE summary_nodes
                SET events_since_update = events_since_update + 1, updated_at = ?
                WHERE key = ?
            """,
                (now_iso(), node_id_or_key),
            )
        else:
            # It's an ID
            cur.execute(
                """
                UPDATE summary_nodes
                SET events_since_update = events_since_update + 1, updated_at = ?
                WHERE id = ?
            """,
                (now_iso(), node_id_or_key),
            )
        self.conn.commit()

    def _row_to_summary_node(self, row: sqlite3.Row) -> SummaryNode:
        """Convert a database row to a SummaryNode."""
        return SummaryNode(
            id=row["id"],
            node_type=row["node_type"],
            key=row["key"],
            label=row["label"],
            parent_id=row["parent_id"],
            summary=row["summary"] or "",
            summary_embedding=deserialize_embedding(row["summary_embedding"]),
            summary_updated_at=parse_datetime(row["summary_updated_at"]),
            events_since_update=row["events_since_update"],
            event_count=row["event_count"],
            first_event_at=parse_datetime(row["first_event_at"]),
            last_event_at=parse_datetime(row["last_event_at"]),
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # TOOLS
    # ═══════════════════════════════════════════════════════════

    def ensure_tool(self, tool_id: str, name: str, description: str = "") -> ToolRecord:
        """Ensure a tool record exists."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM tools WHERE id = ?", (tool_id,))
        row = cur.fetchone()

        if row:
            return self._row_to_tool(row)

        now = now_iso()

        # Create summary node
        summary_node = self.create_summary_node(
            node_type="tool",
            key=f"tool:{tool_id}",
            label=name,
            parent_key="root",
        )

        cur.execute(
            """
            INSERT INTO tools (
                id, name, description, usage_count, summary_node_id, created_at, updated_at
            ) VALUES (?, ?, ?, 0, ?, ?, ?)
        """,
            (tool_id, name, description, summary_node.id, now, now),
        )
        self.conn.commit()

        return ToolRecord(
            id=tool_id,
            name=name,
            description=description,
            usage_count=0,
            summary_node_id=summary_node.id,
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def increment_tool_usage(self, tool_id: str):
        """Increment usage count for a tool."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            UPDATE tools
            SET usage_count = usage_count + 1, last_used_at = ?, updated_at = ?
            WHERE id = ?
        """,
            (now, now, tool_id),
        )
        self.conn.commit()

    def _row_to_tool(self, row: sqlite3.Row) -> ToolRecord:
        """Convert a database row to a ToolRecord."""
        return ToolRecord(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            description_embedding=deserialize_embedding(row["description_embedding"]),
            usage_count=row["usage_count"],
            last_used_at=parse_datetime(row["last_used_at"]),
            summary_node_id=row["summary_node_id"],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # TOOL DEFINITIONS (Full Persistence for Self-Improvement)
    # ═══════════════════════════════════════════════════════════

    def save_tool_definition(
        self,
        name: str,
        description: str,
        parameters: dict,
        source_code: str | None = None,
        packages: list[str] | None = None,
        env: list[str] | None = None,
        tool_var_name: str | None = None,
        category: str = "custom",
        is_dynamic: bool = True,
        created_by_event_id: str | None = None,
        # New fields for skills and composio
        tool_type: str = "executable",
        skill_content: str | None = None,
        composio_app: str | None = None,
        composio_action: str | None = None,
        depends_on: list[str] | None = None,
    ) -> ToolDefinition:
        """
        Save or update a tool definition.

        If a tool with this name exists, updates it (increments version).
        Otherwise creates a new tool definition.

        Args:
            tool_type: "executable" (default), "skill", or "composio"
            skill_content: For skills - the SKILL.md markdown instructions
            composio_app: For composio - app name like "SLACK", "GITHUB"
            composio_action: For composio - action name like "SLACK_SEND_MESSAGE"
            depends_on: List of tool names this tool depends on
        """
        cur = self.conn.cursor()
        now = now_iso()

        # Check if tool already exists
        cur.execute("SELECT * FROM tool_definitions WHERE name = ?", (name,))
        existing = cur.fetchone()

        if existing:
            # Update existing - increment version
            new_version = existing["version"] + 1
            cur.execute(
                """
                UPDATE tool_definitions SET
                    description = ?,
                    tool_type = ?,
                    source_code = ?,
                    parameters = ?,
                    packages = ?,
                    env = ?,
                    tool_var_name = ?,
                    skill_content = ?,
                    composio_app = ?,
                    composio_action = ?,
                    depends_on = ?,
                    category = ?,
                    is_dynamic = ?,
                    version = ?,
                    updated_at = ?
                WHERE name = ?
            """,
                (
                    description,
                    tool_type,
                    source_code,
                    serialize_json(parameters),
                    serialize_json(packages or []),
                    serialize_json(env or []),
                    tool_var_name,
                    skill_content,
                    composio_app,
                    composio_action,
                    serialize_json(depends_on or []),
                    category,
                    1 if is_dynamic else 0,
                    new_version,
                    now,
                    name,
                ),
            )
            self.conn.commit()
            return self.get_tool_definition(name)

        # Create new
        tool_id = generate_id()

        # Create summary node for tool
        summary_node = self.create_summary_node(
            node_type="tool",
            key=f"tool:{name}",
            label=name,
            parent_key=f"tool_category:{category}",
        )

        # Create entity for tool (tools are first-class entities in the graph)
        entity = self.create_entity(
            name=name,
            type="tool",
            type_raw=category,
            description=description,
        )

        cur.execute(
            """
            INSERT INTO tool_definitions (
                id, name, description, tool_type, source_code, parameters, packages, env,
                tool_var_name, skill_content, composio_app, composio_action, depends_on,
                category, is_enabled, is_dynamic,
                entity_id, summary_node_id, created_by_event_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?)
        """,
            (
                tool_id,
                name,
                description,
                tool_type,
                source_code,
                serialize_json(parameters),
                serialize_json(packages or []),
                serialize_json(env or []),
                tool_var_name,
                skill_content,
                composio_app,
                composio_action,
                serialize_json(depends_on or []),
                category,
                1 if is_dynamic else 0,
                entity.id,
                summary_node.id,
                created_by_event_id,
                now,
                now,
            ),
        )
        self.conn.commit()

        return ToolDefinition(
            id=tool_id,
            name=name,
            description=description,
            tool_type=tool_type,
            source_code=source_code,
            parameters=parameters,
            packages=packages or [],
            env=env or [],
            tool_var_name=tool_var_name,
            skill_content=skill_content,
            composio_app=composio_app,
            composio_action=composio_action,
            depends_on=depends_on or [],
            category=category,
            is_enabled=True,
            is_dynamic=is_dynamic,
            entity_id=entity.id,
            summary_node_id=summary_node.id,
            created_by_event_id=created_by_event_id,
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def get_tool_definition(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM tool_definitions WHERE name = ?", (name,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_tool_definition(row)

    def get_tool_definition_by_id(self, tool_id: str) -> ToolDefinition | None:
        """Get a tool definition by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM tool_definitions WHERE id = ?", (tool_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_tool_definition(row)

    def get_all_tool_definitions(self, include_disabled: bool = False) -> list[ToolDefinition]:
        """Get all tool definitions."""
        cur = self.conn.cursor()
        if include_disabled:
            cur.execute("SELECT * FROM tool_definitions ORDER BY name")
        else:
            cur.execute(
                "SELECT * FROM tool_definitions WHERE is_enabled = 1 ORDER BY name"
            )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_dynamic_tool_definitions(self, enabled_only: bool = True) -> list[ToolDefinition]:
        """Get all dynamic (user-created) tool definitions for loading on startup."""
        cur = self.conn.cursor()
        if enabled_only:
            cur.execute(
                """
                SELECT * FROM tool_definitions
                WHERE is_dynamic = 1 AND is_enabled = 1
                ORDER BY name
            """
            )
        else:
            cur.execute(
                "SELECT * FROM tool_definitions WHERE is_dynamic = 1 ORDER BY name"
            )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_tools_by_category(self, category: str) -> list[ToolDefinition]:
        """Get all tools in a category."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM tool_definitions
            WHERE category = ? AND is_enabled = 1
            ORDER BY name
        """,
            (category,),
        )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_tools_by_type(self, tool_type: str, enabled_only: bool = True) -> list[ToolDefinition]:
        """Get all tools of a specific type (executable, skill, composio)."""
        cur = self.conn.cursor()
        if enabled_only:
            cur.execute(
                """
                SELECT * FROM tool_definitions
                WHERE tool_type = ? AND is_enabled = 1
                ORDER BY name
            """,
                (tool_type,),
            )
        else:
            cur.execute(
                "SELECT * FROM tool_definitions WHERE tool_type = ? ORDER BY name",
                (tool_type,),
            )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_skills(self, enabled_only: bool = True) -> list[ToolDefinition]:
        """Get all skill-type tools."""
        return self.get_tools_by_type("skill", enabled_only)

    def get_composio_tools(self, app: str | None = None, enabled_only: bool = True) -> list[ToolDefinition]:
        """Get all composio-type tools, optionally filtered by app."""
        cur = self.conn.cursor()
        if app:
            if enabled_only:
                cur.execute(
                    """
                    SELECT * FROM tool_definitions
                    WHERE tool_type = 'composio' AND composio_app = ? AND is_enabled = 1
                    ORDER BY name
                """,
                    (app,),
                )
            else:
                cur.execute(
                    """
                    SELECT * FROM tool_definitions
                    WHERE tool_type = 'composio' AND composio_app = ?
                    ORDER BY name
                """,
                    (app,),
                )
        else:
            return self.get_tools_by_type("composio", enabled_only)
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_tools_with_dependencies(self) -> list[ToolDefinition]:
        """Get all tools that have dependencies on other tools."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM tool_definitions
            WHERE depends_on IS NOT NULL AND depends_on != '[]' AND is_enabled = 1
            ORDER BY name
        """
        )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def record_tool_success(self, tool_name: str, duration_ms: int):
        """
        Record a successful tool execution.

        Updates usage_count, success_count, avg_duration_ms, and last_used_at.
        """
        cur = self.conn.cursor()
        now = now_iso()

        # Get current stats for running average calculation
        cur.execute(
            "SELECT usage_count, total_duration_ms FROM tool_definitions WHERE name = ?",
            (tool_name,),
        )
        row = cur.fetchone()
        if row is None:
            return  # Tool not tracked yet

        new_total_duration = row["total_duration_ms"] + duration_ms
        new_usage_count = row["usage_count"] + 1
        new_avg_duration = new_total_duration / new_usage_count

        cur.execute(
            """
            UPDATE tool_definitions SET
                usage_count = usage_count + 1,
                success_count = success_count + 1,
                total_duration_ms = ?,
                avg_duration_ms = ?,
                last_used_at = ?,
                updated_at = ?
            WHERE name = ?
        """,
            (new_total_duration, new_avg_duration, now, now, tool_name),
        )
        self.conn.commit()

        # Also update the legacy tools table for backward compatibility
        cur.execute(
            """
            UPDATE tools SET
                usage_count = usage_count + 1,
                last_used_at = ?,
                updated_at = ?
            WHERE name = ?
        """,
            (now, now, tool_name),
        )
        self.conn.commit()

    def record_tool_error(self, tool_name: str, error: str, duration_ms: int = 0):
        """
        Record a tool execution error.

        Updates usage_count, error_count, last_error, and last_error_at.
        Does not crash - errors in error recording are logged but suppressed.
        """
        try:
            cur = self.conn.cursor()
            now = now_iso()

            # Truncate error message if too long
            error_msg = error[:2000] if len(error) > 2000 else error

            # Get current stats
            cur.execute(
                "SELECT usage_count, total_duration_ms FROM tool_definitions WHERE name = ?",
                (tool_name,),
            )
            row = cur.fetchone()
            if row is None:
                return  # Tool not tracked

            new_total_duration = row["total_duration_ms"] + duration_ms
            new_usage_count = row["usage_count"] + 1
            new_avg_duration = new_total_duration / new_usage_count if new_usage_count > 0 else 0

            cur.execute(
                """
                UPDATE tool_definitions SET
                    usage_count = usage_count + 1,
                    error_count = error_count + 1,
                    total_duration_ms = ?,
                    avg_duration_ms = ?,
                    last_error = ?,
                    last_error_at = ?,
                    last_used_at = ?,
                    updated_at = ?
                WHERE name = ?
            """,
                (new_total_duration, new_avg_duration, error_msg, now, now, now, tool_name),
            )
            self.conn.commit()

            # Also update legacy table
            cur.execute(
                """
                UPDATE tools SET
                    usage_count = usage_count + 1,
                    last_used_at = ?,
                    updated_at = ?
                WHERE name = ?
            """,
                (now, now, tool_name),
            )
            self.conn.commit()
        except Exception:
            pass  # Never crash on error recording

    def disable_tool(self, name: str, reason: str | None = None) -> bool:
        """
        Disable a tool (soft delete).

        The tool remains in the database for history but won't be loaded.
        Returns True if tool was found and disabled.
        """
        cur = self.conn.cursor()
        now = now_iso()

        # Store reason in last_error if provided
        if reason:
            cur.execute(
                """
                UPDATE tool_definitions SET
                    is_enabled = 0,
                    last_error = ?,
                    last_error_at = ?,
                    updated_at = ?
                WHERE name = ?
            """,
                (f"Disabled: {reason}", now, now, name),
            )
        else:
            cur.execute(
                """
                UPDATE tool_definitions SET
                    is_enabled = 0,
                    updated_at = ?
                WHERE name = ?
            """,
                (now, name),
            )

        affected = cur.rowcount
        self.conn.commit()
        return affected > 0

    def enable_tool(self, name: str) -> bool:
        """
        Re-enable a disabled tool.

        Returns True if tool was found and enabled.
        """
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            UPDATE tool_definitions SET
                is_enabled = 1,
                updated_at = ?
            WHERE name = ?
        """,
            (now, name),
        )
        affected = cur.rowcount
        self.conn.commit()
        return affected > 0

    def delete_tool_definition(self, name: str) -> bool:
        """
        Permanently delete a tool definition.

        Use disable_tool() for soft delete (preferred).
        Returns True if tool was found and deleted.
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM tool_definitions WHERE name = ?", (name,))
        affected = cur.rowcount
        self.conn.commit()
        return affected > 0

    def get_problematic_tools(self, error_threshold: int = 5) -> list[ToolDefinition]:
        """
        Get tools with high error counts for review.

        Returns tools with error_count >= threshold, ordered by error count.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM tool_definitions
            WHERE error_count >= ?
            ORDER BY error_count DESC
        """,
            (error_threshold,),
        )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_unhealthy_tools(self) -> list[ToolDefinition]:
        """
        Get tools with poor success rates (< 50% success).

        Only considers tools with at least 3 executions.
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM tool_definitions
            WHERE usage_count >= 3
            AND (success_count * 1.0 / usage_count) < 0.5
            ORDER BY (success_count * 1.0 / usage_count) ASC
        """
        )
        return [self._row_to_tool_definition(row) for row in cur.fetchall()]

    def get_tool_stats(self) -> dict:
        """
        Get aggregate statistics about tools.

        Returns dict with counts and health metrics.
        """
        cur = self.conn.cursor()

        # Total counts
        cur.execute("SELECT COUNT(*) FROM tool_definitions")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM tool_definitions WHERE is_enabled = 1")
        enabled = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM tool_definitions WHERE is_dynamic = 1")
        dynamic = cur.fetchone()[0]

        # Error stats
        cur.execute("SELECT SUM(error_count), SUM(success_count) FROM tool_definitions")
        row = cur.fetchone()
        total_errors = row[0] or 0
        total_successes = row[1] or 0

        # Most used
        cur.execute(
            """
            SELECT name, usage_count FROM tool_definitions
            ORDER BY usage_count DESC LIMIT 5
        """
        )
        most_used = [{"name": r["name"], "count": r["usage_count"]} for r in cur.fetchall()]

        # Most errors
        cur.execute(
            """
            SELECT name, error_count FROM tool_definitions
            WHERE error_count > 0
            ORDER BY error_count DESC LIMIT 5
        """
        )
        most_errors = [{"name": r["name"], "count": r["error_count"]} for r in cur.fetchall()]

        return {
            "total_tools": total,
            "enabled_tools": enabled,
            "dynamic_tools": dynamic,
            "total_executions": total_errors + total_successes,
            "total_errors": total_errors,
            "total_successes": total_successes,
            "success_rate": (
                (total_successes / (total_errors + total_successes) * 100)
                if (total_errors + total_successes) > 0
                else 100.0
            ),
            "most_used": most_used,
            "most_errors": most_errors,
        }

    def _row_to_tool_definition(self, row: sqlite3.Row) -> ToolDefinition:
        """Convert a database row to a ToolDefinition."""
        # Handle potentially missing columns for backward compatibility
        row_dict = dict(row)

        return ToolDefinition(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            tool_type=row_dict.get("tool_type") or "executable",
            source_code=row["source_code"],
            parameters=deserialize_json(row["parameters"]) or {},
            packages=deserialize_json(row["packages"]) or [],
            env=deserialize_json(row["env"]) or [],
            tool_var_name=row["tool_var_name"],
            skill_content=row_dict.get("skill_content"),
            composio_app=row_dict.get("composio_app"),
            composio_action=row_dict.get("composio_action"),
            depends_on=deserialize_json(row_dict.get("depends_on")) or [],
            category=row["category"],
            is_enabled=bool(row["is_enabled"]),
            is_dynamic=bool(row["is_dynamic"]),
            usage_count=row["usage_count"],
            success_count=row["success_count"],
            error_count=row["error_count"],
            last_used_at=parse_datetime(row["last_used_at"]),
            last_error=row["last_error"],
            last_error_at=parse_datetime(row["last_error_at"]),
            avg_duration_ms=row["avg_duration_ms"] or 0.0,
            total_duration_ms=row["total_duration_ms"] or 0.0,
            entity_id=row["entity_id"],
            summary_node_id=row["summary_node_id"],
            version=row["version"],
            created_by_event_id=row["created_by_event_id"],
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # AGENT STATE
    # ═══════════════════════════════════════════════════════════

    def get_agent_state(self) -> AgentState:
        """Get the agent state."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM agent_state LIMIT 1")
        row = cur.fetchone()
        if row is None:
            # Should not happen after initialization
            self._ensure_agent_state()
            return self.get_agent_state()
        return self._row_to_agent_state(row)

    def update_agent_state(self, **updates) -> AgentState:
        """Update the agent state."""
        state = self.get_agent_state()
        cur = self.conn.cursor()
        now = now_iso()

        # Apply updates
        for key, value in updates.items():
            if hasattr(state, key):
                setattr(state, key, value)

        cur.execute(
            """
            UPDATE agent_state SET
                name = ?, description = ?, owner_entity_id = ?, self_entity_id = ?,
                current_topics = ?, mood = ?, focus = ?, active_tasks = ?,
                settings = ?, state_updated_at = ?
            WHERE id = ?
        """,
            (
                state.name,
                state.description,
                state.owner_entity_id,
                state.self_entity_id,
                serialize_json(state.current_topics),
                state.mood,
                state.focus,
                serialize_json(state.active_tasks),
                serialize_json(state.settings),
                now,
                state.id,
            ),
        )
        self.conn.commit()

        return self.get_agent_state()

    def _row_to_agent_state(self, row: sqlite3.Row) -> AgentState:
        """Convert a database row to AgentState."""
        return AgentState(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            owner_entity_id=row["owner_entity_id"],
            self_entity_id=row["self_entity_id"],
            current_topics=deserialize_json(row["current_topics"]) or [],
            mood=row["mood"],
            focus=row["focus"],
            active_tasks=deserialize_json(row["active_tasks"]) or [],
            settings=deserialize_json(row["settings"]) or {},
            state_updated_at=parse_datetime(row["state_updated_at"]),
            created_at=parse_datetime(row["created_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # EVENT RETENTION / CLEANUP
    # ═══════════════════════════════════════════════════════════

    def apply_retention_policy(self, policy: RetentionPolicy) -> int:
        """
        Apply retention policy to clean up old events.

        Returns the number of events deleted.
        """
        deleted = 0

        # Delete by age
        if policy.max_age_days is not None:
            deleted += self._delete_old_events(policy)

        # Delete by count
        if policy.max_events is not None:
            deleted += self._trim_to_max_events(policy)

        return deleted

    def _delete_old_events(self, policy: RetentionPolicy) -> int:
        """Delete events older than max_age_days."""
        cur = self.conn.cursor()
        deleted = 0

        # Calculate cutoff dates
        cutoff = (datetime.now() - timedelta(days=policy.max_age_days)).isoformat()
        important_cutoff = None
        if policy.important_event_types and policy.important_multiplier > 1:
            important_days = int(policy.max_age_days * policy.important_multiplier)
            important_cutoff = (datetime.now() - timedelta(days=important_days)).isoformat()

        # Build exclusion conditions
        exclusions = []
        params = [cutoff]

        if policy.preserve_with_entities:
            exclusions.append("person_id IS NOT NULL")

        if policy.preserve_owner_events:
            exclusions.append("is_owner = 1")

        # Handle important event types
        if policy.important_event_types and important_cutoff:
            important_types = ",".join(f"'{t}'" for t in policy.important_event_types)
            exclusions.append(f"(event_type IN ({important_types}) AND timestamp > ?)")
            params.append(important_cutoff)

        exclusion_clause = " OR ".join(exclusions) if exclusions else "0"

        # Delete in batches
        while True:
            cur.execute(
                f"""
                DELETE FROM events
                WHERE id IN (
                    SELECT id FROM events
                    WHERE timestamp < ?
                    AND NOT ({exclusion_clause})
                    LIMIT ?
                )
            """,
                params + [policy.cleanup_batch_size],
            )
            batch_deleted = cur.rowcount
            self.conn.commit()

            if batch_deleted == 0:
                break
            deleted += batch_deleted

        return deleted

    def _trim_to_max_events(self, policy: RetentionPolicy) -> int:
        """Trim events to max_events count."""
        cur = self.conn.cursor()

        # Get current count
        cur.execute("SELECT COUNT(*) FROM events")
        current_count = cur.fetchone()[0]

        if current_count <= policy.max_events:
            return 0

        excess = current_count - policy.max_events
        deleted = 0

        # Build exclusion conditions
        exclusions = []
        if policy.preserve_with_entities:
            exclusions.append("person_id IS NOT NULL")
        if policy.preserve_owner_events:
            exclusions.append("is_owner = 1")

        exclusion_clause = " OR ".join(exclusions) if exclusions else "0"

        # Delete oldest events in batches
        while deleted < excess:
            batch_size = min(policy.cleanup_batch_size, excess - deleted)
            cur.execute(
                f"""
                DELETE FROM events
                WHERE id IN (
                    SELECT id FROM events
                    WHERE NOT ({exclusion_clause})
                    ORDER BY timestamp ASC
                    LIMIT ?
                )
            """,
                (batch_size,),
            )
            batch_deleted = cur.rowcount
            self.conn.commit()

            if batch_deleted == 0:
                break
            deleted += batch_deleted

        return deleted

    def get_event_count(self) -> int:
        """Get the total number of events."""
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM events")
        return cur.fetchone()[0]

    def get_oldest_event_date(self) -> datetime | None:
        """Get the timestamp of the oldest event."""
        cur = self.conn.cursor()
        cur.execute("SELECT MIN(timestamp) FROM events")
        result = cur.fetchone()[0]
        return parse_datetime(result) if result else None

    def vacuum(self):
        """Compact the database after deletions."""
        self.conn.execute("VACUUM")

    # ═══════════════════════════════════════════════════════════
    # CREDENTIALS (secure storage for accounts and payment methods)
    # ═══════════════════════════════════════════════════════════

    def store_credential(
        self,
        service: str,
        credential_type: str = "account",
        username: str | None = None,
        email: str | None = None,
        password_ref: str | None = None,
        card_last_four: str | None = None,
        card_type: str | None = None,
        card_expiry: str | None = None,
        card_ref: str | None = None,
        billing_name: str | None = None,
        billing_address: str | None = None,
        notes: str | None = None,
        metadata: dict | None = None,
    ) -> Credential:
        """Store a credential (account or credit card).

        For accounts, the password should be stored in keyring first,
        then the reference passed as password_ref.

        For credit cards, the full card number should be stored in keyring,
        then the reference passed as card_ref. Only last 4 digits stored here.

        Args:
            service: Service name (e.g., "yohei.ai", "stripe.com")
            credential_type: "account", "credit_card", or "api_key"
            username: Username for account
            email: Email for account
            password_ref: Reference to password in keyring
            card_last_four: Last 4 digits of card
            card_type: Card type (visa, mastercard, etc.)
            card_expiry: Card expiry in MM/YY format
            card_ref: Reference to full card number in keyring
            billing_name: Name on card
            billing_address: Billing address
            notes: Additional notes
            metadata: Additional metadata dict
        """
        cred_id = generate_id()
        now = now_iso()

        cur = self.conn.cursor()

        # Check if credential for this service already exists
        cur.execute(
            "SELECT id FROM credentials WHERE service = ? AND credential_type = ?",
            (service, credential_type),
        )
        existing = cur.fetchone()

        if existing:
            # Update existing credential
            cur.execute(
                """
                UPDATE credentials SET
                    username = COALESCE(?, username),
                    email = COALESCE(?, email),
                    password_ref = COALESCE(?, password_ref),
                    card_last_four = COALESCE(?, card_last_four),
                    card_type = COALESCE(?, card_type),
                    card_expiry = COALESCE(?, card_expiry),
                    card_ref = COALESCE(?, card_ref),
                    billing_name = COALESCE(?, billing_name),
                    billing_address = COALESCE(?, billing_address),
                    notes = COALESCE(?, notes),
                    metadata = COALESCE(?, metadata),
                    updated_at = ?
                WHERE id = ?
            """,
                (
                    username,
                    email,
                    password_ref,
                    card_last_four,
                    card_type,
                    card_expiry,
                    card_ref,
                    billing_name,
                    billing_address,
                    notes,
                    serialize_json(metadata),
                    now,
                    existing["id"],
                ),
            )
            self.conn.commit()
            return self.get_credential(service, credential_type)

        # Insert new credential
        cur.execute(
            """
            INSERT INTO credentials (
                id, credential_type, service, username, email, password_ref,
                card_last_four, card_type, card_expiry, card_ref,
                billing_name, billing_address, notes, metadata,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                cred_id,
                credential_type,
                service,
                username,
                email,
                password_ref,
                card_last_four,
                card_type,
                card_expiry,
                card_ref,
                billing_name,
                billing_address,
                notes,
                serialize_json(metadata),
                now,
                now,
            ),
        )
        self.conn.commit()

        return Credential(
            id=cred_id,
            credential_type=credential_type,
            service=service,
            username=username,
            email=email,
            password_ref=password_ref,
            card_last_four=card_last_four,
            card_type=card_type,
            card_expiry=card_expiry,
            card_ref=card_ref,
            billing_name=billing_name,
            billing_address=billing_address,
            notes=notes,
            metadata=metadata,
            created_at=parse_datetime(now),
            updated_at=parse_datetime(now),
        )

    def get_credential(
        self, service: str, credential_type: str | None = None
    ) -> Credential | None:
        """Get a credential by service name.

        Args:
            service: Service name to look up
            credential_type: Optional filter by type
        """
        cur = self.conn.cursor()

        if credential_type:
            cur.execute(
                "SELECT * FROM credentials WHERE service = ? AND credential_type = ?",
                (service, credential_type),
            )
        else:
            cur.execute("SELECT * FROM credentials WHERE service = ?", (service,))

        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_credential(row)

    def get_credential_by_id(self, cred_id: str) -> Credential | None:
        """Get a credential by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM credentials WHERE id = ?", (cred_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_credential(row)

    def list_credentials(
        self, credential_type: str | None = None, limit: int = 50
    ) -> list[Credential]:
        """List all credentials, optionally filtered by type.

        Args:
            credential_type: Filter by type ("account", "credit_card", "api_key")
            limit: Maximum number to return
        """
        cur = self.conn.cursor()

        if credential_type:
            cur.execute(
                """
                SELECT * FROM credentials
                WHERE credential_type = ?
                ORDER BY service ASC
                LIMIT ?
            """,
                (credential_type, limit),
            )
        else:
            cur.execute(
                "SELECT * FROM credentials ORDER BY service ASC LIMIT ?", (limit,)
            )

        return [self._row_to_credential(row) for row in cur.fetchall()]

    def search_credentials(self, query: str, limit: int = 20) -> list[Credential]:
        """Search credentials by service name or username/email.

        Args:
            query: Search query
            limit: Maximum results
        """
        cur = self.conn.cursor()
        pattern = f"%{query}%"

        cur.execute(
            """
            SELECT * FROM credentials
            WHERE service LIKE ?
               OR username LIKE ?
               OR email LIKE ?
               OR billing_name LIKE ?
            ORDER BY service ASC
            LIMIT ?
        """,
            (pattern, pattern, pattern, pattern, limit),
        )

        return [self._row_to_credential(row) for row in cur.fetchall()]

    def update_credential_last_used(self, cred_id: str):
        """Update the last_used_at timestamp for a credential."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            "UPDATE credentials SET last_used_at = ?, updated_at = ? WHERE id = ?",
            (now, now, cred_id),
        )
        self.conn.commit()

    def delete_credential(self, cred_id: str) -> bool:
        """Delete a credential by ID.

        Returns True if deleted, False if not found.
        """
        cur = self.conn.cursor()
        cur.execute("DELETE FROM credentials WHERE id = ?", (cred_id,))
        deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def _row_to_credential(self, row: sqlite3.Row) -> Credential:
        """Convert a database row to a Credential."""
        return Credential(
            id=row["id"],
            credential_type=row["credential_type"],
            service=row["service"],
            username=row["username"],
            email=row["email"],
            password_ref=row["password_ref"],
            card_last_four=row["card_last_four"],
            card_type=row["card_type"],
            card_expiry=row["card_expiry"],
            card_ref=row["card_ref"],
            billing_name=row["billing_name"],
            billing_address=row["billing_address"],
            notes=row["notes"],
            metadata=deserialize_json(row["metadata"]),
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
            last_used_at=parse_datetime(row["last_used_at"]),
        )

    # ═══════════════════════════════════════════════════════════
    # LEARNINGS (Self-Improvement System)
    # ═══════════════════════════════════════════════════════════

    def create_learning(self, learning: Learning) -> Learning:
        """Create a new learning from feedback or evaluation."""
        cur = self.conn.cursor()
        now = now_iso()

        # Ensure ID exists
        if not learning.id:
            learning.id = generate_id()

        cur.execute(
            """
            INSERT INTO learnings
            (id, source_type, source_event_id, content, content_embedding,
             sentiment, confidence, tool_id, topic_ids, objective_type,
             entity_ids, applies_when, recommendation, times_applied,
             last_applied_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                learning.id,
                learning.source_type,
                learning.source_event_id,
                learning.content,
                serialize_embedding(learning.content_embedding),
                learning.sentiment,
                learning.confidence,
                learning.tool_id,
                serialize_json(learning.topic_ids),
                learning.objective_type,
                serialize_json(learning.entity_ids),
                learning.applies_when,
                learning.recommendation,
                learning.times_applied,
                learning.last_applied_at.isoformat() if learning.last_applied_at else None,
                now,
                now,
            ),
        )
        self.conn.commit()
        return learning

    def get_learning(self, learning_id: str) -> Learning | None:
        """Get a learning by ID."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM learnings WHERE id = ?", (learning_id,))
        row = cur.fetchone()
        return self._row_to_learning(row) if row else None

    def find_learnings(
        self,
        tool_id: str | None = None,
        objective_type: str | None = None,
        sentiment: str | None = None,
        source_type: str | None = None,
        limit: int = 20,
    ) -> list[Learning]:
        """Find learnings by filters."""
        cur = self.conn.cursor()

        query = "SELECT * FROM learnings WHERE 1=1"
        params = []

        if tool_id:
            query += " AND tool_id = ?"
            params.append(tool_id)
        if objective_type:
            query += " AND objective_type = ?"
            params.append(objective_type)
        if sentiment:
            query += " AND sentiment = ?"
            params.append(sentiment)
        if source_type:
            query += " AND source_type = ?"
            params.append(source_type)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cur.execute(query, params)
        return [self._row_to_learning(row) for row in cur.fetchall()]

    def search_learnings(
        self,
        embedding: list[float],
        tool_id: str | None = None,
        objective_type: str | None = None,
        limit: int = 10,
    ) -> list[Learning]:
        """Search learnings by vector similarity.

        Uses cosine similarity on content_embedding.
        Falls back to recent learnings if no embeddings available.
        """
        cur = self.conn.cursor()

        # Build filter conditions
        conditions = ["content_embedding IS NOT NULL"]
        params = []

        if tool_id:
            conditions.append("tool_id = ?")
            params.append(tool_id)
        if objective_type:
            conditions.append("objective_type = ?")
            params.append(objective_type)

        where_clause = " AND ".join(conditions)

        cur.execute(f"SELECT * FROM learnings WHERE {where_clause}", params)
        rows = cur.fetchall()

        if not rows:
            # Fallback to recent learnings without embeddings
            return self.find_learnings(
                tool_id=tool_id,
                objective_type=objective_type,
                limit=limit,
            )

        # Calculate similarities and sort
        scored = []
        for row in rows:
            row_embedding = deserialize_embedding(row["content_embedding"])
            if row_embedding:
                similarity = self._cosine_similarity(embedding, row_embedding)
                scored.append((similarity, row))

        # Sort by similarity descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [self._row_to_learning(row) for _, row in scored[:limit]]

    def get_learnings_for_tool(self, tool_id: str, limit: int = 5) -> list[Learning]:
        """Get learnings specific to a tool, prioritizing corrections."""
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT * FROM learnings
            WHERE tool_id = ?
            ORDER BY
                CASE sentiment WHEN 'negative' THEN 0 ELSE 1 END,
                created_at DESC
            LIMIT ?
        """,
            (tool_id, limit),
        )
        return [self._row_to_learning(row) for row in cur.fetchall()]

    def get_all_learnings(self, limit: int = 100) -> list[Learning]:
        """Get all learnings, ordered by recency."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM learnings ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [self._row_to_learning(row) for row in cur.fetchall()]

    def record_learning_applied(self, learning_id: str):
        """Record that a learning was used in context."""
        cur = self.conn.cursor()
        now = now_iso()
        cur.execute(
            """
            UPDATE learnings
            SET times_applied = times_applied + 1,
                last_applied_at = ?,
                updated_at = ?
            WHERE id = ?
        """,
            (now, now, learning_id),
        )
        self.conn.commit()

    def delete_learning(self, learning_id: str) -> bool:
        """Delete a learning by ID."""
        cur = self.conn.cursor()
        cur.execute("DELETE FROM learnings WHERE id = ?", (learning_id,))
        deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def get_learning_stats(self) -> dict:
        """Get aggregate statistics about learnings."""
        cur = self.conn.cursor()

        cur.execute("SELECT COUNT(*) as total FROM learnings")
        total = cur.fetchone()["total"]

        cur.execute(
            """
            SELECT sentiment, COUNT(*) as count
            FROM learnings
            GROUP BY sentiment
        """
        )
        by_sentiment = {row["sentiment"]: row["count"] for row in cur.fetchall()}

        cur.execute(
            """
            SELECT source_type, COUNT(*) as count
            FROM learnings
            GROUP BY source_type
        """
        )
        by_source = {row["source_type"]: row["count"] for row in cur.fetchall()}

        cur.execute(
            """
            SELECT tool_id, COUNT(*) as count
            FROM learnings
            WHERE tool_id IS NOT NULL
            GROUP BY tool_id
            ORDER BY count DESC
            LIMIT 10
        """
        )
        by_tool = {row["tool_id"]: row["count"] for row in cur.fetchall()}

        return {
            "total": total,
            "by_sentiment": by_sentiment,
            "by_source": by_source,
            "by_tool": by_tool,
        }

    def _row_to_learning(self, row: sqlite3.Row) -> Learning:
        """Convert a database row to a Learning."""
        return Learning(
            id=row["id"],
            source_type=row["source_type"],
            source_event_id=row["source_event_id"],
            content=row["content"],
            content_embedding=deserialize_embedding(row["content_embedding"]),
            sentiment=row["sentiment"],
            confidence=row["confidence"],
            tool_id=row["tool_id"],
            topic_ids=deserialize_json(row["topic_ids"]) or [],
            objective_type=row["objective_type"],
            entity_ids=deserialize_json(row["entity_ids"]) or [],
            applies_when=row["applies_when"],
            recommendation=row["recommendation"],
            times_applied=row["times_applied"],
            last_applied_at=parse_datetime(row["last_applied_at"]),
            created_at=parse_datetime(row["created_at"]),
            updated_at=parse_datetime(row["updated_at"]),
        )

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ═══════════════════════════════════════════════════════════
    # METRICS
    # ═══════════════════════════════════════════════════════════

    def record_llm_call(self, metric) -> None:
        """
        Record an LLM API call metric.

        Args:
            metric: LLMCallMetric instance from metrics.models
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO llm_calls
            (id, timestamp, source, model, thread_id, input_tokens,
             output_tokens, cost_usd, duration_ms, stop_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metric.id,
                metric.timestamp.isoformat(),
                metric.source,
                metric.model,
                metric.thread_id,
                metric.input_tokens,
                metric.output_tokens,
                metric.cost_usd,
                metric.duration_ms,
                metric.stop_reason,
            ),
        )
        self.conn.commit()

    def record_embedding_call(self, metric) -> None:
        """
        Record an embedding API call metric.

        Args:
            metric: EmbeddingCallMetric instance from metrics.models
        """
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO embedding_calls
            (id, timestamp, provider, model, text_count,
             token_estimate, cost_usd, duration_ms, cached)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                metric.id,
                metric.timestamp.isoformat(),
                metric.provider,
                metric.model,
                metric.text_count,
                metric.token_estimate,
                metric.cost_usd,
                metric.duration_ms,
                1 if metric.cached else 0,
            ),
        )
        self.conn.commit()

    def get_llm_call_stats(
        self,
        source: str | None = None,
        since: str | None = None,
    ) -> dict:
        """
        Get aggregated LLM call statistics.

        Args:
            source: Filter by source (optional)
            since: ISO timestamp to filter from (optional)

        Returns:
            Dict with call_count, total_tokens, total_cost, avg_latency_ms
        """
        cur = self.conn.cursor()

        query = "SELECT * FROM llm_calls WHERE 1=1"
        params = []

        if source:
            query += " AND source = ?"
            params.append(source)

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        cur.execute(query, params)
        rows = cur.fetchall()

        if not rows:
            return {
                "call_count": 0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "avg_latency_ms": 0.0,
            }

        total_tokens = sum(r["input_tokens"] + r["output_tokens"] for r in rows)
        total_cost = sum(r["cost_usd"] for r in rows)
        total_latency = sum(r["duration_ms"] for r in rows)

        return {
            "call_count": len(rows),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_latency_ms": total_latency / len(rows),
        }

    def get_embedding_call_stats(self, since: str | None = None) -> dict:
        """
        Get aggregated embedding call statistics.

        Args:
            since: ISO timestamp to filter from (optional)

        Returns:
            Dict with call_count, text_count, total_cost, cache_hit_rate
        """
        cur = self.conn.cursor()

        query = "SELECT * FROM embedding_calls WHERE 1=1"
        params = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since)

        cur.execute(query, params)
        rows = cur.fetchall()

        if not rows:
            return {
                "call_count": 0,
                "text_count": 0,
                "total_cost": 0.0,
                "cache_hit_rate": 0.0,
            }

        cached_count = sum(1 for r in rows if r["cached"])
        text_count = sum(r["text_count"] for r in rows)
        total_cost = sum(r["cost_usd"] for r in rows)

        return {
            "call_count": len(rows),
            "text_count": text_count,
            "total_cost": total_cost,
            "cache_hit_rate": cached_count / len(rows) * 100,
        }

    # ═══════════════════════════════════════════════════════════
    # CLOSE
    # ═══════════════════════════════════════════════════════════

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
