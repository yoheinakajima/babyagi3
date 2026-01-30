"""
Memory store - database operations for the memory system.

Uses SQLite with sqlite-vec for vector search.
"""

import json
import os
import sqlite3
import struct
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    AgentState,
    Edge,
    Entity,
    Event,
    EventTopic,
    SummaryNode,
    Task,
    ToolRecord,
    Topic,
)

# Vector dimensions
EMBEDDING_DIM = 1536


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
        self._ensure_root_node()
        self._ensure_agent_state()

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

        # Tools
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

    def _increment_staleness_for_event(
        self,
        channel: str | None,
        tool_id: str | None,
        person_id: str | None,
        task_id: str | None,
    ):
        """Increment staleness counters for relevant summary nodes."""
        cur = self.conn.cursor()
        now = now_iso()

        # Root always gets incremented
        cur.execute(
            """
            UPDATE summary_nodes
            SET events_since_update = events_since_update + 1,
                event_count = event_count + 1,
                last_event_at = ?,
                updated_at = ?
            WHERE key = 'root'
        """,
            (now, now),
        )

        # Channel
        if channel:
            cur.execute(
                """
                UPDATE summary_nodes
                SET events_since_update = events_since_update + 1,
                    event_count = event_count + 1,
                    last_event_at = ?,
                    updated_at = ?
                WHERE key = ?
            """,
                (now, now, f"channel:{channel}"),
            )

        # Tool
        if tool_id:
            cur.execute(
                """
                UPDATE summary_nodes
                SET events_since_update = events_since_update + 1,
                    event_count = event_count + 1,
                    last_event_at = ?,
                    updated_at = ?
                WHERE key = ?
            """,
                (now, now, f"tool:{tool_id}"),
            )

        # Entity (person)
        if person_id:
            cur.execute(
                """
                UPDATE summary_nodes
                SET events_since_update = events_since_update + 1,
                    event_count = event_count + 1,
                    last_event_at = ?,
                    updated_at = ?
                WHERE key = ?
            """,
                (now, now, f"entity:{person_id}"),
            )

        # Task
        if task_id:
            cur.execute(
                """
                UPDATE summary_nodes
                SET events_since_update = events_since_update + 1,
                    event_count = event_count + 1,
                    last_event_at = ?,
                    updated_at = ?
                WHERE key = ?
            """,
                (now, now, f"task:{task_id}"),
            )

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

    def increment_staleness(self, node_id: str):
        """Increment staleness counter for a node."""
        cur = self.conn.cursor()
        cur.execute(
            """
            UPDATE summary_nodes
            SET events_since_update = events_since_update + 1, updated_at = ?
            WHERE id = ?
        """,
            (now_iso(), node_id),
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
    # CLEANUP
    # ═══════════════════════════════════════════════════════════

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
