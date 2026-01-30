"""
Event Log - Immutable, append-only record of everything.

The event log is the source of truth. Events are tagged with deterministic
metadata at write time. No LLM calls needed for logging.

Storage: Simple JSON Lines file for durability, with in-memory index for queries.
"""

import json
import threading
from pathlib import Path
from typing import Iterator, Callable
from .models import Event, SliceKey


class EventLog:
    """
    Append-only event storage with tag-based querying.

    Design:
    - Events are immutable once written
    - Tags enable fast filtering without full scans
    - Supports both file persistence and in-memory operation
    """

    def __init__(self, storage_path: Path | None = None):
        self._events: list[Event] = []
        self._by_id: dict[str, Event] = {}
        self._by_tag: dict[str, dict[str, list[str]]] = {}  # tag_key -> tag_value -> event_ids
        self._lock = threading.Lock()
        self._storage_path = storage_path
        self._subscribers: list[Callable[[Event], None]] = []

        if storage_path:
            self._load()

    def _load(self):
        """Load events from disk."""
        if not self._storage_path or not self._storage_path.exists():
            return

        with open(self._storage_path, "r") as f:
            for line in f:
                if line.strip():
                    event = Event.from_dict(json.loads(line))
                    self._index_event(event)

    def _persist(self, event: Event):
        """Append event to disk."""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._storage_path, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def _index_event(self, event: Event):
        """Add event to in-memory indices."""
        self._events.append(event)
        self._by_id[event.id] = event

        # Index by each tag
        for tag_key, tag_value in event.tags.items():
            tag_key = tag_key.lower()
            tag_value = tag_value.lower()
            if tag_key not in self._by_tag:
                self._by_tag[tag_key] = {}
            if tag_value not in self._by_tag[tag_key]:
                self._by_tag[tag_key][tag_value] = []
            self._by_tag[tag_key][tag_value].append(event.id)

    def append(self, event: Event) -> Event:
        """
        Append a new event to the log.

        This is the only way to add events. Events are immutable after this.
        """
        with self._lock:
            if event.id in self._by_id:
                raise ValueError(f"Event {event.id} already exists")

            self._index_event(event)
            self._persist(event)

            # Notify subscribers
            for sub in self._subscribers:
                try:
                    sub(event)
                except Exception:
                    pass  # Don't let subscriber errors break logging

        return event

    def log(self, type: str, content: any, tags: dict[str, str]) -> Event:
        """
        Convenience method to create and append an event.

        Args:
            type: Event type (message_in, message_out, tool_call, etc.)
            content: Event content (flexible)
            tags: Deterministic tags for indexing

        Returns:
            The created event
        """
        event = Event.create(type=type, content=content, tags=tags)
        return self.append(event)

    def get(self, event_id: str) -> Event | None:
        """Get a single event by ID."""
        return self._by_id.get(event_id)

    def query(
        self,
        slice_key: SliceKey | None = None,
        types: list[str] | None = None,
        since: str | None = None,
        until: str | None = None,
        limit: int | None = None,
        reverse: bool = True,
    ) -> list[Event]:
        """
        Query events matching criteria.

        Args:
            slice_key: Filter by tag combination
            types: Filter by event types
            since: Only events after this timestamp
            until: Only events before this timestamp
            limit: Maximum events to return
            reverse: If True, return newest first

        Returns:
            List of matching events
        """
        # Start with all events or filtered by slice
        if slice_key and slice_key.tags:
            # Find intersection of events matching all tags
            candidate_ids = None
            for tag_key, tag_value in slice_key.tags.items():
                tag_events = set(
                    self._by_tag.get(tag_key, {}).get(tag_value, [])
                )
                if candidate_ids is None:
                    candidate_ids = tag_events
                else:
                    candidate_ids &= tag_events

            if candidate_ids is None:
                candidates = []
            else:
                candidates = [self._by_id[eid] for eid in candidate_ids]
        else:
            candidates = self._events.copy()

        # Apply filters
        results = []
        for event in candidates:
            if types and event.type not in types:
                continue
            if since and event.timestamp < since:
                continue
            if until and event.timestamp > until:
                continue
            results.append(event)

        # Sort by timestamp
        results.sort(key=lambda e: e.timestamp, reverse=reverse)

        # Apply limit
        if limit:
            results = results[:limit]

        return results

    def recent(self, n: int = 20, slice_key: SliceKey | None = None) -> list[Event]:
        """Get the N most recent events, optionally filtered by slice."""
        return self.query(slice_key=slice_key, limit=n, reverse=True)

    def count(self, slice_key: SliceKey | None = None) -> int:
        """Count events matching a slice."""
        if not slice_key or not slice_key.tags:
            return len(self._events)
        return len(self.query(slice_key=slice_key))

    def iter_events(self, reverse: bool = False) -> Iterator[Event]:
        """Iterate over all events."""
        events = reversed(self._events) if reverse else self._events
        for event in events:
            yield event

    def subscribe(self, callback: Callable[[Event], None]):
        """
        Subscribe to new events.

        Callback is called synchronously when new events are appended.
        Use this for triggering extraction pipelines.
        """
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Event], None]):
        """Remove a subscriber."""
        self._subscribers.remove(callback)

    def get_all_slices(self) -> list[SliceKey]:
        """
        Get all unique single-tag slices that have events.

        Useful for knowing what summaries to maintain.
        """
        slices = [SliceKey.root()]
        for tag_key, tag_values in self._by_tag.items():
            for tag_value in tag_values.keys():
                slices.append(SliceKey({tag_key: tag_value}))
        return slices

    def get_tag_values(self, tag_key: str) -> list[str]:
        """Get all unique values for a tag key."""
        return list(self._by_tag.get(tag_key.lower(), {}).keys())

    def __len__(self) -> int:
        return len(self._events)

    def __contains__(self, event_id: str) -> bool:
        return event_id in self._by_id
