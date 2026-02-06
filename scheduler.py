"""
Elegant Scheduler with Full Cron Support

Three ways to schedule:
- "at": One-time execution at a specific timestamp
- "every": Recurring at fixed intervals (with optional anchor)
- "cron": Full 5-field cron expressions with timezone support

Design principles:
- Minimal code, maximum capability
- Human-readable schedule definitions
- Persistent across restarts
- Execution history for debugging
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Literal

logger = logging.getLogger(__name__)

# Optional: croniter for full cron support, falls back to simple patterns
try:
    from croniter import croniter
    HAS_CRONITER = True
except ImportError:
    HAS_CRONITER = False

# Optional: pytz/zoneinfo for timezone support
try:
    from zoneinfo import ZoneInfo
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
    except ImportError:
        ZoneInfo = None


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Schedule:
    """
    Unified schedule definition supporting three modes:

    1. One-time: Schedule(kind="at", at="2024-01-15T09:00:00", tz="America/New_York")
    2. Interval: Schedule(kind="every", every="5m") or every="2h" or every="1d"
    3. Cron:     Schedule(kind="cron", cron="0 9 * * 1-5", tz="America/New_York")

    Human-friendly intervals: 5m, 2h, 1d, 30s
    """
    kind: Literal["at", "every", "cron"]
    at: str | None = None      # ISO timestamp for one-time
    every: str | None = None   # Interval: "5m", "2h", "1d", "30s"
    cron: str | None = None    # Cron expression: "0 9 * * 1-5"
    tz: str | None = None      # Timezone: "America/New_York", "UTC"
    anchor: str | None = None  # Anchor time for intervals (ISO timestamp)

    def __post_init__(self):
        # Validate based on kind
        if self.kind == "at" and not self.at:
            raise ValueError("Schedule kind='at' requires 'at' timestamp")
        if self.kind == "every" and not self.every:
            raise ValueError("Schedule kind='every' requires 'every' interval")
        if self.kind == "cron" and not self.cron:
            raise ValueError("Schedule kind='cron' requires 'cron' expression")

    def next_run(self, after: datetime = None) -> datetime | None:
        """Calculate next run time after given datetime (or now)."""
        after = after or datetime.now(timezone.utc)

        # Ensure after is timezone-aware
        if after.tzinfo is None:
            after = after.replace(tzinfo=timezone.utc)

        if self.kind == "at":
            target = datetime.fromisoformat(self.at.replace('Z', '+00:00'))
            if target.tzinfo is None:
                target = target.replace(tzinfo=self._get_tz())
            return target if target > after else None

        elif self.kind == "every":
            interval_seconds = self._parse_interval(self.every)
            anchor = self._get_anchor()

            # Calculate next occurrence
            elapsed = (after - anchor).total_seconds()
            intervals_passed = int(elapsed / interval_seconds)
            next_time = anchor + timedelta(seconds=(intervals_passed + 1) * interval_seconds)
            return next_time

        elif self.kind == "cron":
            if not HAS_CRONITER:
                # Fallback to simple patterns
                return self._simple_cron_next(after)

            tz = self._get_tz()
            local_after = after.astimezone(tz)
            cron = croniter(self.cron, local_after)
            next_local = cron.get_next(datetime)
            return next_local.astimezone(timezone.utc)

        return None

    def _get_tz(self):
        """Get timezone object."""
        if not self.tz:
            return timezone.utc
        if ZoneInfo:
            return ZoneInfo(self.tz)
        return timezone.utc

    def _get_anchor(self) -> datetime:
        """Get anchor time for interval scheduling."""
        if self.anchor:
            anchor = datetime.fromisoformat(self.anchor.replace('Z', '+00:00'))
            if anchor.tzinfo is None:
                anchor = anchor.replace(tzinfo=timezone.utc)
            return anchor
        # Default anchor: midnight UTC today
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)

    def _parse_interval(self, interval: str) -> int:
        """Parse human-friendly interval to seconds: 5m, 2h, 1d, 30s"""
        unit_map = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        unit = interval[-1].lower()
        value = int(interval[:-1])
        return value * unit_map.get(unit, 60)

    def _simple_cron_next(self, after: datetime) -> datetime:
        """Simple cron-like patterns without croniter."""
        # Support basic patterns: hourly, daily, weekly
        pattern = self.cron.lower().strip()
        now = after.astimezone(self._get_tz())

        if pattern in ("hourly", "0 * * * *"):
            # Next hour
            next_time = now.replace(minute=0, second=0, microsecond=0)
            next_time += timedelta(hours=1)
            return next_time.astimezone(timezone.utc)

        elif pattern in ("daily", "0 0 * * *"):
            # Next midnight
            next_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            next_time += timedelta(days=1)
            return next_time.astimezone(timezone.utc)

        # Try to parse simple "minute hour" patterns
        parts = pattern.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            minute, hour = int(parts[0]), int(parts[1])
            next_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_time <= now:
                next_time += timedelta(days=1)
            return next_time.astimezone(timezone.utc)

        # Default: next hour
        next_time = now.replace(minute=0, second=0, microsecond=0)
        next_time += timedelta(hours=1)
        return next_time.astimezone(timezone.utc)

    def human_readable(self) -> str:
        """Return human-readable description of the schedule."""
        if self.kind == "at":
            return f"once at {self.at}"
        elif self.kind == "every":
            return f"every {self.every}"
        elif self.kind == "cron":
            tz_str = f" ({self.tz})" if self.tz else ""
            return f"cron: {self.cron}{tz_str}"
        return "unknown"


@dataclass
class ScheduledTask:
    """
    A scheduled task with execution tracking.

    Combines the simplicity of our Objective model with Moltbot's scheduling power.
    """
    id: str
    name: str
    goal: str  # What to execute
    schedule: Schedule
    enabled: bool = True

    # Execution state
    next_run_at: str | None = None  # ISO timestamp
    last_run_at: str | None = None
    last_status: str | None = None  # "ok", "error", "skipped"
    last_error: str | None = None
    last_duration_ms: int | None = None
    run_count: int = 0

    # Metadata
    created_at: str = ""
    thread_id: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        if not self.thread_id:
            self.thread_id = f"scheduled_{self.id}"
        if not self.next_run_at:
            self._update_next_run()

    def _update_next_run(self):
        """Calculate and set next run time."""
        next_time = self.schedule.next_run()
        self.next_run_at = next_time.isoformat() if next_time else None

    def to_dict(self) -> dict:
        """Serialize to dictionary for storage."""
        d = asdict(self)
        d['schedule'] = asdict(self.schedule)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ScheduledTask":
        """Deserialize from dictionary."""
        schedule_data = data.pop('schedule')
        schedule = Schedule(**schedule_data)
        return cls(schedule=schedule, **data)


@dataclass
class RunRecord:
    """Record of a single task execution."""
    task_id: str
    started_at: str
    completed_at: str | None = None
    status: str = "running"  # running, ok, error, skipped
    result: str | None = None
    error: str | None = None
    duration_ms: int | None = None


# =============================================================================
# Persistence
# =============================================================================

class SchedulerStore:
    """
    Simple JSON file persistence with atomic writes.

    Storage location: ~/.babyagi/scheduler/
    - tasks.json: All scheduled tasks
    - runs/{task_id}.jsonl: Execution history per task
    """

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir or os.path.expanduser("~/.babyagi/scheduler"))
        self.tasks_file = self.base_dir / "tasks.json"
        self.runs_dir = self.base_dir / "runs"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create directories if they don't exist."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(exist_ok=True)

    def load_tasks(self) -> dict[str, ScheduledTask]:
        """Load all tasks from storage."""
        if not self.tasks_file.exists():
            return {}

        try:
            with open(self.tasks_file) as f:
                data = json.load(f)
            return {
                task_id: ScheduledTask.from_dict(task_data)
                for task_id, task_data in data.items()
            }
        except (json.JSONDecodeError, KeyError) as e:
            # Corrupted file - start fresh but keep backup
            if self.tasks_file.exists():
                backup = self.tasks_file.with_suffix('.json.bak')
                self.tasks_file.rename(backup)
            return {}

    def save_tasks(self, tasks: dict[str, ScheduledTask]):
        """Save all tasks with atomic write."""
        data = {task_id: task.to_dict() for task_id, task in tasks.items()}

        # Write to temp file then rename (atomic on POSIX)
        temp_file = self.tasks_file.with_suffix('.json.tmp')
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        temp_file.rename(self.tasks_file)

    def append_run(self, record: RunRecord):
        """Append run record to task's history file."""
        run_file = self.runs_dir / f"{record.task_id}.jsonl"
        with open(run_file, 'a') as f:
            f.write(json.dumps(asdict(record)) + '\n')

        # Prune if too large (keep last 1000 entries)
        self._prune_runs(run_file, max_entries=1000)

    def get_runs(self, task_id: str, limit: int = 20) -> list[RunRecord]:
        """Get recent run history for a task."""
        run_file = self.runs_dir / f"{task_id}.jsonl"
        if not run_file.exists():
            return []

        runs = []
        with open(run_file) as f:
            for line in f:
                if line.strip():
                    try:
                        runs.append(RunRecord(**json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue

        return runs[-limit:]  # Return most recent

    def _prune_runs(self, run_file: Path, max_entries: int):
        """Keep only the most recent entries."""
        if not run_file.exists():
            return

        with open(run_file) as f:
            lines = f.readlines()

        if len(lines) > max_entries:
            with open(run_file, 'w') as f:
                f.writelines(lines[-max_entries:])


# =============================================================================
# Scheduler Engine
# =============================================================================

class Scheduler:
    """
    The scheduler: runs tasks at their scheduled times.

    Features:
    - Single timer loop (efficient, no busy-waiting)
    - Persistent across restarts
    - Execution history
    - Three schedule types: at, every, cron

    Usage:
        scheduler = Scheduler()
        scheduler.add(ScheduledTask(...))
        await scheduler.run()  # Starts the loop
    """

    def __init__(self, executor: Callable = None, store: SchedulerStore = None):
        """
        Args:
            executor: Async function(task: ScheduledTask) -> str that runs the task
            store: Storage backend (defaults to file-based)
        """
        self.executor = executor
        self.store = store or SchedulerStore()
        self.tasks: dict[str, ScheduledTask] = {}
        self._running = False
        self._timer_task: asyncio.Task | None = None
        self._running_tasks: set[str] = set()
        self._lock = asyncio.Lock()

        # Load persisted tasks
        self.tasks = self.store.load_tasks()

    # -------------------------------------------------------------------------
    # Task Management API
    # -------------------------------------------------------------------------

    def add(self, task: ScheduledTask) -> ScheduledTask:
        """Add a new scheduled task."""
        self.tasks[task.id] = task
        self.store.save_tasks(self.tasks)
        self._reschedule_timer()
        return task

    def update(self, task_id: str, **updates) -> ScheduledTask | None:
        """Update an existing task."""
        task = self.tasks.get(task_id)
        if not task:
            return None

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        # Recalculate next run if schedule changed
        if 'schedule' in updates or 'enabled' in updates:
            task._update_next_run()

        self.store.save_tasks(self.tasks)
        self._reschedule_timer()
        return task

    def remove(self, task_id: str) -> bool:
        """Remove a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.store.save_tasks(self.tasks)
            self._reschedule_timer()
            return True
        return False

    def get(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def list(self, include_disabled: bool = False) -> list[ScheduledTask]:
        """List all tasks."""
        tasks = list(self.tasks.values())
        if not include_disabled:
            tasks = [t for t in tasks if t.enabled]
        return sorted(tasks, key=lambda t: t.next_run_at or "")

    def get_runs(self, task_id: str, limit: int = 20) -> list[RunRecord]:
        """Get execution history for a task."""
        return self.store.get_runs(task_id, limit)

    async def run_now(self, task_id: str, force: bool = False) -> dict:
        """
        Manually trigger a task.

        Args:
            task_id: Task to run
            force: If True, run even if not due. If False, only run if overdue.
        """
        task = self.tasks.get(task_id)
        if not task:
            return {"error": f"Task {task_id} not found"}

        # Check if already running
        if task_id in self._running_tasks:
            return {"status": "skipped", "reason": "already running"}

        if not force:
            # Check if due
            if task.next_run_at:
                next_run = datetime.fromisoformat(task.next_run_at.replace('Z', '+00:00'))
                if next_run > datetime.now(timezone.utc):
                    return {"error": "Task not yet due", "next_run": task.next_run_at}

        # Mark as running before execution
        self._running_tasks.add(task_id)
        result = await self._execute_task(task)
        return result

    # -------------------------------------------------------------------------
    # Scheduler Loop
    # -------------------------------------------------------------------------

    async def start(self):
        """Start the scheduler loop."""
        if self._running:
            return

        self._running = True
        self._timer_task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the scheduler loop."""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        """Main scheduler loop - sleeps until next task is due."""
        while self._running:
            try:
                # Find next due task
                next_wake = self._next_wake_time()

                if next_wake is None:
                    # No tasks scheduled - check again in 60s
                    await asyncio.sleep(60)
                    continue

                # Sleep until next task is due
                sleep_seconds = max(0, (next_wake - datetime.now(timezone.utc)).total_seconds())
                if sleep_seconds > 0:
                    # Cap at 24 hours to handle clock changes
                    await asyncio.sleep(min(sleep_seconds, 86400))
                else:
                    # Critical: Even when tasks are immediately due, yield to the event loop
                    # This prevents starvation of other coroutines (CLI input, API calls, etc.)
                    await asyncio.sleep(0)

                # Run all due tasks
                await self._run_due_tasks()

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but keep running
                logger.error("Scheduler error: %s", e)
                await asyncio.sleep(60)

    def _next_wake_time(self) -> datetime | None:
        """Find the earliest next run time across all tasks."""
        next_times = []
        for task in self.tasks.values():
            if task.enabled and task.next_run_at:
                try:
                    next_time = datetime.fromisoformat(task.next_run_at.replace('Z', '+00:00'))
                    next_times.append(next_time)
                except ValueError:
                    continue

        return min(next_times) if next_times else None

    async def _run_due_tasks(self):
        """Execute all tasks that are due."""
        now = datetime.now(timezone.utc)

        for task in list(self.tasks.values()):
            if not task.enabled or not task.next_run_at:
                continue

            # Check if due
            try:
                next_run = datetime.fromisoformat(task.next_run_at.replace('Z', '+00:00'))
            except ValueError:
                continue

            if next_run <= now:
                # Check if already running
                if task.id in self._running_tasks:
                    continue

                # Mark as running BEFORE spawning to prevent race condition
                # (otherwise create_task returns immediately but _execute_task
                # hasn't added to _running_tasks yet, causing duplicate spawns)
                self._running_tasks.add(task.id)

                # Execute in background
                asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: ScheduledTask) -> dict:
        """Execute a single task with tracking."""
        # Task was already added to _running_tasks by _run_due_tasks() before spawning.
        # This eliminates the race condition and removes the need for an async lock here.
        # We still verify for edge cases (e.g., direct run_now() calls).
        if task.id not in self._running_tasks:
            self._running_tasks.add(task.id)

        started_at = datetime.now(timezone.utc)
        record = RunRecord(
            task_id=task.id,
            started_at=started_at.isoformat(),
            status="running"
        )

        try:
            if self.executor:
                result = await self.executor(task)
            else:
                result = f"No executor configured. Task goal: {task.goal}"

            # Success
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            record.completed_at = completed_at.isoformat()
            record.status = "ok"
            record.result = str(result)[:500]  # Truncate
            record.duration_ms = duration_ms

            task.last_run_at = completed_at.isoformat()
            task.last_status = "ok"
            task.last_error = None
            task.last_duration_ms = duration_ms
            task.run_count += 1

            return {"status": "ok", "result": result}

        except Exception as e:
            # Error
            completed_at = datetime.now(timezone.utc)
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            record.completed_at = completed_at.isoformat()
            record.status = "error"
            record.error = str(e)
            record.duration_ms = duration_ms

            task.last_run_at = completed_at.isoformat()
            task.last_status = "error"
            task.last_error = str(e)
            task.last_duration_ms = duration_ms
            task.run_count += 1

            return {"status": "error", "error": str(e)}

        finally:
            # Update next run time
            task._update_next_run()

            # Remove one-time tasks that have completed
            if task.schedule.kind == "at" and task.next_run_at is None:
                task.enabled = False

            # Persist in thread pool to avoid blocking the event loop
            await asyncio.to_thread(self.store.save_tasks, self.tasks)
            await asyncio.to_thread(self.store.append_run, record)

            self._running_tasks.discard(task.id)

    def _reschedule_timer(self):
        """Hint to wake the timer loop early."""
        # The loop will naturally pick up changes on next iteration
        pass


# =============================================================================
# Helper Functions
# =============================================================================

def parse_schedule(spec: str | dict) -> Schedule:
    """
    Parse a schedule specification into a Schedule object.

    Accepts:
    - dict: {"kind": "cron", "cron": "0 9 * * *", "tz": "America/New_York"}
    - str shortcuts:
        - "in 5m" or "in 2h" → one-time, relative
        - "every 5m" or "every 2h" → recurring interval
        - "at 2024-01-15T09:00" → one-time absolute
        - "daily at 9:00" → cron shortcut
        - "weekdays at 9:00" → cron shortcut
        - "0 9 * * *" → raw cron
    """
    if isinstance(spec, dict):
        return Schedule(**spec)

    spec = spec.strip().lower()

    # "in 5m" - relative one-time
    if spec.startswith("in "):
        interval = spec[3:].strip()
        seconds = Schedule(kind="every", every=interval)._parse_interval(interval)
        run_at = datetime.now(timezone.utc) + timedelta(seconds=seconds)
        return Schedule(kind="at", at=run_at.isoformat())

    # "at <timestamp>" - absolute one-time
    if spec.startswith("at "):
        timestamp = spec[3:].strip()
        return Schedule(kind="at", at=timestamp)

    # "every 5m" - interval
    if spec.startswith("every "):
        interval = spec[6:].strip()
        return Schedule(kind="every", every=interval)

    # "daily at 9:00" or "daily at 9am"
    if spec.startswith("daily at "):
        time_part = spec[9:].strip()
        hour, minute = _parse_time(time_part)
        return Schedule(kind="cron", cron=f"{minute} {hour} * * *")

    # "weekdays at 9:00"
    if spec.startswith("weekdays at "):
        time_part = spec[12:].strip()
        hour, minute = _parse_time(time_part)
        return Schedule(kind="cron", cron=f"{minute} {hour} * * 1-5")

    # "hourly" / "daily"
    if spec == "hourly":
        return Schedule(kind="cron", cron="0 * * * *")
    if spec == "daily":
        return Schedule(kind="cron", cron="0 0 * * *")

    # Assume raw cron expression
    return Schedule(kind="cron", cron=spec)


def _parse_time(time_str: str) -> tuple[int, int]:
    """Parse time string like "9:00", "9am", "14:30" to (hour, minute)."""
    time_str = time_str.lower().strip()

    # Handle am/pm
    is_pm = 'pm' in time_str
    is_am = 'am' in time_str
    time_str = time_str.replace('am', '').replace('pm', '').strip()

    # Parse hour:minute
    if ':' in time_str:
        parts = time_str.split(':')
        hour, minute = int(parts[0]), int(parts[1])
    else:
        hour, minute = int(time_str), 0

    # Adjust for PM
    if is_pm and hour < 12:
        hour += 12
    elif is_am and hour == 12:
        hour = 0

    return hour, minute


def create_task(
    name: str,
    goal: str,
    schedule: str | dict | Schedule,
    task_id: str = None
) -> ScheduledTask:
    """
    Convenience function to create a scheduled task.

    Examples:
        create_task("Daily standup", "Check emails and summarize", "daily at 9:00")
        create_task("Quick check", "Check server status", "every 5m")
        create_task("Reminder", "Send report", "in 2h")
    """
    if not isinstance(schedule, Schedule):
        schedule = parse_schedule(schedule)

    if task_id is None:
        task_id = uuid.uuid4().hex[:8]

    return ScheduledTask(
        id=task_id,
        name=name,
        goal=goal,
        schedule=schedule
    )
