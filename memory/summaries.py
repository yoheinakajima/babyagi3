"""
Summary Tree - Hierarchical pre-computed summaries for instant context assembly.

Every "slice" of the data has a summary. When new events come in, leaf summaries
get marked stale. When leaves update, parents get marked stale. Summaries refresh
on a schedule or when staleness exceeds a threshold.

The key insight: Context assembly becomes O(1) lookups instead of O(n) LLM calls.
"""

import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Callable, Awaitable
from .models import Summary, SliceKey, Event


class SummaryTree:
    """
    Hierarchical summary storage with staleness propagation.

    The tree structure is implicit in slice keys:
    - "*" (root) is parent of all single-tag slices
    - "channel:email" is parent of "channel:email+person:john"

    When a leaf gets stale, staleness propagates up to ancestors.
    """

    def __init__(self, storage_path: Path | None = None):
        self._summaries: dict[str, Summary] = {}  # slice_key -> Summary
        self._lock = threading.Lock()
        self._storage_path = storage_path
        self._summarize_fn: Callable[[SliceKey, list[Event]], Awaitable[str]] | None = None

        if storage_path:
            self._load()

    def _load(self):
        """Load summaries from disk."""
        if not self._storage_path or not self._storage_path.exists():
            return

        with open(self._storage_path, "r") as f:
            data = json.load(f)

        for sd in data.get("summaries", []):
            summary = Summary.from_dict(sd)
            self._summaries[summary.slice_key] = summary

    def _save(self):
        """Save summaries to disk."""
        if not self._storage_path:
            return

        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"summaries": [s.to_dict() for s in self._summaries.values()]}
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def set_summarize_fn(self, fn: Callable[[SliceKey, list[Event]], Awaitable[str]]):
        """Set the function used to generate summaries."""
        self._summarize_fn = fn

    def get(self, slice_key: SliceKey | str) -> Summary | None:
        """Get summary for a slice."""
        key = slice_key.key if isinstance(slice_key, SliceKey) else slice_key
        return self._summaries.get(key)

    def get_text(self, slice_key: SliceKey | str, default: str = "") -> str:
        """Get summary text, returning default if not found."""
        summary = self.get(slice_key)
        return summary.text if summary else default

    def set(
        self,
        slice_key: SliceKey,
        text: str,
        event_count: int,
        last_event_id: str,
    ) -> Summary:
        """
        Create or update a summary.

        This also marks parent summaries as stale.
        """
        with self._lock:
            key = slice_key.key
            existing = self._summaries.get(key)

            if existing:
                existing.refresh(text, event_count, last_event_id)
                summary = existing
            else:
                summary = Summary(
                    slice_key=key,
                    text=text,
                    event_count=event_count,
                    last_event_id=last_event_id,
                )
                self._summaries[key] = summary

            self._save()
            return summary

    def mark_stale(self, slice_key: SliceKey, propagate: bool = True):
        """
        Mark a summary as stale.

        If propagate=True, also marks all parent summaries as stale.
        """
        with self._lock:
            key = slice_key.key
            summary = self._summaries.get(key)

            if summary:
                # Calculate staleness based on how many new events
                summary.mark_stale(score=0.5)

            if propagate:
                # Mark parents stale too (with lower score since they're further from the change)
                for parent in slice_key.parents():
                    parent_summary = self._summaries.get(parent.key)
                    if parent_summary:
                        parent_summary.mark_stale(score=0.3)

            self._save()

    def mark_stale_for_event(self, event: Event):
        """
        Mark all summaries affected by a new event as stale.

        Called automatically when events are logged.
        """
        # Find all slices this event matches
        tags = event.tags
        affected_slices = [SliceKey.root()]  # Root is always affected

        # Single-tag slices
        for key, value in tags.items():
            affected_slices.append(SliceKey({key: value}))

        # Two-tag combinations (common intersections)
        keys = list(tags.keys())
        for i, k1 in enumerate(keys):
            for k2 in keys[i + 1:]:
                affected_slices.append(SliceKey({k1: tags[k1], k2: tags[k2]}))

        # Mark all affected slices stale
        with self._lock:
            for slice_key in affected_slices:
                summary = self._summaries.get(slice_key.key)
                if summary:
                    summary.mark_stale(score=0.5)
            self._save()

    def get_stale_summaries(self, threshold: float = 0.0) -> list[Summary]:
        """
        Get summaries that need refreshing.

        Args:
            threshold: Minimum staleness score to include

        Returns:
            List of stale summaries, sorted by staleness (most stale first)
        """
        stale = [
            s for s in self._summaries.values()
            if s.stale and s.staleness_score > threshold
        ]
        stale.sort(key=lambda s: s.staleness_score, reverse=True)
        return stale

    def get_fresh_summaries(self) -> list[Summary]:
        """Get all summaries that are not stale."""
        return [s for s in self._summaries.values() if not s.stale]

    def ensure_summary(
        self,
        slice_key: SliceKey,
        events: list[Event],
        force: bool = False,
    ) -> Summary | None:
        """
        Ensure a summary exists and is fresh.

        If the summary is stale or doesn't exist, and we have a summarize_fn,
        this will generate a new summary. This is synchronous and blocking.

        For async refresh, use refresh_stale_async() instead.
        """
        key = slice_key.key
        existing = self._summaries.get(key)

        if existing and not existing.stale and not force:
            return existing

        # Can't generate without a summarize function
        if not self._summarize_fn:
            return existing

        # Synchronous generation not recommended - use refresh_stale_async
        return existing

    async def refresh_stale_async(
        self,
        event_log,  # EventLog - imported dynamically to avoid circular imports
        max_refreshes: int = 10,
    ) -> int:
        """
        Refresh stale summaries asynchronously.

        Args:
            event_log: The event log to query for events
            max_refreshes: Maximum summaries to refresh in one call

        Returns:
            Number of summaries refreshed
        """
        if not self._summarize_fn:
            return 0

        stale = self.get_stale_summaries()[:max_refreshes]
        refreshed = 0

        for summary in stale:
            slice_key = SliceKey.from_key(summary.slice_key)
            events = event_log.query(slice_key=slice_key, limit=100)

            if events:
                try:
                    text = await self._summarize_fn(slice_key, events)
                    self.set(
                        slice_key,
                        text=text,
                        event_count=len(events),
                        last_event_id=events[0].id,
                    )
                    refreshed += 1
                except Exception:
                    pass  # Log error but don't break the loop

        return refreshed

    def get_all_slice_keys(self) -> list[str]:
        """Get all slice keys that have summaries."""
        return list(self._summaries.keys())

    def summary_count(self) -> int:
        return len(self._summaries)

    def stale_count(self) -> int:
        return sum(1 for s in self._summaries.values() if s.stale)

    def stats(self) -> dict:
        """Get summary statistics."""
        return {
            "total": self.summary_count(),
            "stale": self.stale_count(),
            "fresh": self.summary_count() - self.stale_count(),
            "total_events_covered": sum(s.event_count for s in self._summaries.values()),
        }


class SummaryRefresher:
    """
    Background worker that refreshes stale summaries.

    Can be run on a schedule or triggered manually.
    """

    def __init__(
        self,
        summary_tree: SummaryTree,
        event_log,  # EventLog
        refresh_interval: float = 60.0,  # seconds
        staleness_threshold: float = 0.3,
    ):
        self.tree = summary_tree
        self.event_log = event_log
        self.refresh_interval = refresh_interval
        self.staleness_threshold = staleness_threshold
        self._running = False
        self._task = None

    async def start(self):
        """Start the background refresh loop."""
        import asyncio

        self._running = True
        while self._running:
            try:
                await self.tree.refresh_stale_async(
                    self.event_log,
                    max_refreshes=5,
                )
            except Exception:
                pass  # Log error but continue

            await asyncio.sleep(self.refresh_interval)

    def stop(self):
        """Stop the background refresh loop."""
        self._running = False
