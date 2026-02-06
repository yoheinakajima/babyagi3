"""
File Index - Semantic search over stored files.

Maintains a lightweight SQLite index of file metadata, summaries, and
embeddings. Enables semantic search across files using the same embedding
infrastructure as the memory system.

The index is stored at ~/.babyagi/files/.file_index.db alongside the
file storage directory.
"""

import json
import sqlite3
import struct
import time
from pathlib import Path

import logging

logger = logging.getLogger(__name__)


def _serialize_embedding(embedding: list[float] | None) -> bytes | None:
    """Pack float list into compact binary."""
    if not embedding:
        return None
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(data: bytes | None) -> list[float] | None:
    """Unpack binary into float list."""
    if not data:
        return None
    count = len(data) // 4
    return list(struct.unpack(f"{count}f", data))


class FileIndex:
    """
    SQLite-based index for semantic file search.

    Stores file metadata (path, project, filename, summary, tags) alongside
    an embedding of the summary text. Searches combine keyword matching with
    cosine similarity for ranked results.
    """

    def __init__(self, base_path: str = "~/.babyagi/files"):
        db_path = Path(base_path).expanduser() / ".file_index.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self):
        """Create the file index table."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS file_index (
                path TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                filename TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]',
                embedding BLOB,
                indexed_at REAL NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_project ON file_index(project)"
        )
        self._conn.commit()

    def index_file(
        self,
        path: str,
        project: str,
        filename: str,
        summary: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ):
        """
        Add or update a file in the index.

        Args:
            path: Absolute file path (primary key).
            project: Project folder name.
            filename: The file's name.
            summary: LLM-generated summary of the file content.
            tags: List of tags for categorization.
            embedding: Precomputed embedding of the summary text.
                       If None, will be generated on demand during search.
        """
        self._conn.execute(
            """
            INSERT OR REPLACE INTO file_index
                (path, project, filename, summary, tags, embedding, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path,
                project,
                filename,
                summary,
                json.dumps(tags or []),
                _serialize_embedding(embedding),
                time.time(),
            ),
        )
        self._conn.commit()

    def remove(self, path: str):
        """Remove a file from the index."""
        self._conn.execute("DELETE FROM file_index WHERE path = ?", (path,))
        self._conn.commit()

    def update_path(self, old_path: str, new_path: str, new_project: str | None = None, new_filename: str | None = None):
        """Update the path (and optionally project/filename) after a move."""
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM file_index WHERE path = ?", (old_path,))
        row = cur.fetchone()
        if not row:
            return

        project = new_project if new_project is not None else row["project"]
        filename = new_filename if new_filename is not None else row["filename"]

        cur.execute(
            """
            INSERT OR REPLACE INTO file_index
                (path, project, filename, summary, tags, embedding, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (new_path, project, filename, row["summary"], row["tags"], row["embedding"], time.time()),
        )
        if old_path != new_path:
            cur.execute("DELETE FROM file_index WHERE path = ?", (old_path,))
        self._conn.commit()

    def search(
        self,
        query: str,
        limit: int = 20,
        project: str | None = None,
    ) -> list[dict]:
        """
        Search files by combining keyword matching and semantic similarity.

        Scoring:
        - Semantic similarity on summary embedding (0.0-1.0, weight 0.7)
        - Keyword hits on filename, project, tags, summary (0.0-1.0, weight 0.3)

        Falls back to keyword-only search if embeddings are unavailable.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            project: Optional project filter.

        Returns:
            List of file info dicts sorted by relevance, each with a
            "relevance" score.
        """
        # Load all indexed files (filtered by project if given)
        cur = self._conn.cursor()
        if project:
            cur.execute("SELECT * FROM file_index WHERE project = ?", (project,))
        else:
            cur.execute("SELECT * FROM file_index")
        rows = cur.fetchall()

        if not rows:
            return []

        # Compute keyword scores
        query_lower = query.lower()
        query_terms = query_lower.split()

        scored = []
        for row in rows:
            keyword_score = self._keyword_score(row, query_lower, query_terms)
            scored.append((row, keyword_score, None))  # semantic score filled below

        # Attempt semantic scoring
        query_embedding = self._get_query_embedding(query)
        if query_embedding:
            new_scored = []
            for row, kw_score, _ in scored:
                row_embedding = _deserialize_embedding(row["embedding"])
                if row_embedding:
                    sem_score = _cosine_similarity(query_embedding, row_embedding)
                else:
                    sem_score = 0.0
                new_scored.append((row, kw_score, sem_score))
            scored = new_scored

        # Combine scores: semantic (0.7) + keyword (0.3), or keyword-only
        results = []
        for row, kw_score, sem_score in scored:
            if sem_score is not None:
                combined = 0.7 * sem_score + 0.3 * kw_score
            else:
                combined = kw_score

            # Skip zero-relevance results
            if combined < 0.01:
                continue

            tags = json.loads(row["tags"]) if row["tags"] else []
            results.append({
                "path": row["path"],
                "project": row["project"],
                "filename": row["filename"],
                "summary": row["summary"],
                "tags": tags,
                "relevance": round(combined, 4),
            })

        # Sort by relevance descending
        results.sort(key=lambda r: r["relevance"], reverse=True)
        return results[:limit]

    def get_all(self) -> list[dict]:
        """Get all indexed files."""
        cur = self._conn.cursor()
        cur.execute("SELECT path, project, filename, summary, tags, indexed_at FROM file_index")
        results = []
        for row in cur.fetchall():
            tags = json.loads(row["tags"]) if row["tags"] else []
            results.append({
                "path": row["path"],
                "project": row["project"],
                "filename": row["filename"],
                "summary": row["summary"],
                "tags": tags,
            })
        return results

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ───────────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────────

    @staticmethod
    def _keyword_score(row, query_lower: str, query_terms: list[str]) -> float:
        """
        Score a file against keyword terms.

        Checks filename, project, tags, and summary text.
        Returns a 0.0-1.0 score based on term hit rate.
        """
        filename_lower = row["filename"].lower()
        project_lower = row["project"].lower()
        summary_lower = row["summary"].lower() if row["summary"] else ""
        tags_lower = row["tags"].lower() if row["tags"] else ""

        searchable = f"{filename_lower} {project_lower} {summary_lower} {tags_lower}"

        if not query_terms:
            return 0.0

        # Full phrase match gets a bonus
        hits = sum(1 for term in query_terms if term in searchable)
        term_score = hits / len(query_terms)

        # Exact phrase bonus
        phrase_bonus = 0.2 if query_lower in searchable else 0.0

        # Filename match bonus (filenames are more important)
        filename_bonus = 0.2 if query_lower in filename_lower else 0.0

        return min(1.0, term_score + phrase_bonus + filename_bonus)

    @staticmethod
    def _get_query_embedding(query: str) -> list[float] | None:
        """Get embedding for query, returning None if unavailable."""
        try:
            from memory.embeddings import get_embedding
            return get_embedding(query)
        except Exception as e:
            logger.debug("Could not generate query embedding: %s", e)
            return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)
