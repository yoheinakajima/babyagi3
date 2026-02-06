"""
Unit tests for scheduler.py

Tests cover:
- Schedule dataclass validation and next_run computation
- Interval parsing (seconds, minutes, hours, days)
- Cron expression scheduling (with and without croniter)
- One-time "at" scheduling
- ScheduledTask lifecycle (creation, serialization, deserialization)
- SchedulerStore persistence (save, load, run records)
- Scheduler task management API (add, update, remove, list)
- parse_schedule helper for human-friendly strings
- _parse_time helper
- create_task convenience function
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from scheduler import (
    Schedule,
    ScheduledTask,
    RunRecord,
    SchedulerStore,
    Scheduler,
    parse_schedule,
    create_task,
    _parse_time,
)


# =============================================================================
# Schedule Dataclass Tests
# =============================================================================


class TestScheduleValidation:
    """Test Schedule __post_init__ validation."""

    def test_at_requires_timestamp(self):
        with pytest.raises(ValueError, match="requires 'at' timestamp"):
            Schedule(kind="at")

    def test_every_requires_interval(self):
        with pytest.raises(ValueError, match="requires 'every' interval"):
            Schedule(kind="every")

    def test_cron_requires_expression(self):
        with pytest.raises(ValueError, match="requires 'cron' expression"):
            Schedule(kind="cron")

    def test_valid_at_schedule(self):
        s = Schedule(kind="at", at="2030-01-15T09:00:00+00:00")
        assert s.kind == "at"
        assert s.at == "2030-01-15T09:00:00+00:00"

    def test_valid_every_schedule(self):
        s = Schedule(kind="every", every="5m")
        assert s.every == "5m"

    def test_valid_cron_schedule(self):
        s = Schedule(kind="cron", cron="0 9 * * 1-5")
        assert s.cron == "0 9 * * 1-5"


class TestScheduleIntervalParsing:
    """Test _parse_interval for human-friendly intervals."""

    def test_seconds(self):
        s = Schedule(kind="every", every="30s")
        assert s._parse_interval("30s") == 30

    def test_minutes(self):
        s = Schedule(kind="every", every="5m")
        assert s._parse_interval("5m") == 300

    def test_hours(self):
        s = Schedule(kind="every", every="2h")
        assert s._parse_interval("2h") == 7200

    def test_days(self):
        s = Schedule(kind="every", every="1d")
        assert s._parse_interval("1d") == 86400

    def test_unknown_unit_defaults_to_minutes(self):
        s = Schedule(kind="every", every="10x")
        # Unknown unit defaults to 60 (minute multiplier)
        assert s._parse_interval("10x") == 600


class TestScheduleNextRun:
    """Test next_run calculations for different schedule kinds."""

    def test_at_future_returns_target(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        s = Schedule(kind="at", at=future)
        nxt = s.next_run()
        assert nxt is not None
        assert nxt > datetime.now(timezone.utc)

    def test_at_past_returns_none(self):
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        s = Schedule(kind="at", at=past)
        assert s.next_run() is None

    def test_every_returns_future(self):
        s = Schedule(kind="every", every="5m")
        nxt = s.next_run()
        assert nxt is not None
        assert nxt > datetime.now(timezone.utc)

    def test_every_with_anchor(self):
        anchor = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
        s = Schedule(kind="every", every="5m", anchor=anchor)
        nxt = s.next_run()
        assert nxt is not None

    def test_every_after_specific_time(self):
        base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        s = Schedule(
            kind="every",
            every="1h",
            anchor="2025-01-01T00:00:00+00:00",
        )
        nxt = s.next_run(after=base)
        assert nxt is not None
        assert nxt.hour == 13  # Next hour after 12:00

    def test_cron_returns_future(self):
        s = Schedule(kind="cron", cron="0 * * * *")  # Every hour
        nxt = s.next_run()
        assert nxt is not None
        assert nxt > datetime.now(timezone.utc)

    def test_at_with_z_suffix(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=1))
        at_str = future.strftime("%Y-%m-%dT%H:%M:%SZ")
        s = Schedule(kind="at", at=at_str)
        nxt = s.next_run()
        assert nxt is not None

    def test_naive_after_gets_utc(self):
        """Passing a naive datetime should still work (treated as UTC)."""
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        s = Schedule(kind="at", at=future)
        nxt = s.next_run(after=datetime.utcnow())
        assert nxt is not None


class TestScheduleHumanReadable:
    """Test human_readable() descriptions."""

    def test_at(self):
        s = Schedule(kind="at", at="2030-01-15T09:00:00")
        assert "once at" in s.human_readable()

    def test_every(self):
        s = Schedule(kind="every", every="5m")
        assert "every 5m" in s.human_readable()

    def test_cron(self):
        s = Schedule(kind="cron", cron="0 9 * * *")
        assert "cron:" in s.human_readable()

    def test_cron_with_tz(self):
        s = Schedule(kind="cron", cron="0 9 * * *", tz="America/New_York")
        hr = s.human_readable()
        assert "America/New_York" in hr


class TestSimpleCronFallback:
    """Test _simple_cron_next when croniter is not available."""

    def test_hourly_pattern(self):
        s = Schedule(kind="cron", cron="hourly")
        base = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        nxt = s._simple_cron_next(base)
        assert nxt.minute == 0  # Top of next hour

    def test_daily_pattern(self):
        s = Schedule(kind="cron", cron="daily")
        base = datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc)
        nxt = s._simple_cron_next(base)
        assert nxt.hour == 0
        assert nxt.minute == 0

    def test_minute_hour_pattern(self):
        s = Schedule(kind="cron", cron="30 14 * * *")
        base = datetime(2025, 6, 15, 10, 0, 0, tzinfo=timezone.utc)
        nxt = s._simple_cron_next(base)
        assert nxt.hour == 14
        assert nxt.minute == 30


# =============================================================================
# ScheduledTask Tests
# =============================================================================


class TestScheduledTask:
    """Test ScheduledTask creation and serialization."""

    def test_creation_defaults(self):
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="t1", name="Test", goal="Do something", schedule=s)
        assert task.id == "t1"
        assert task.enabled is True
        assert task.run_count == 0
        assert task.thread_id == "scheduled_t1"
        assert task.created_at != ""
        assert task.next_run_at is not None

    def test_to_dict_and_from_dict(self):
        s = Schedule(kind="every", every="10m")
        task = ScheduledTask(id="t2", name="Roundtrip", goal="Test serialization", schedule=s)
        d = task.to_dict()
        assert d["id"] == "t2"
        assert d["schedule"]["kind"] == "every"

        restored = ScheduledTask.from_dict(d)
        assert restored.id == "t2"
        assert restored.name == "Roundtrip"
        assert restored.schedule.every == "10m"
        assert restored.schedule.kind == "every"

    def test_custom_thread_id(self):
        s = Schedule(kind="every", every="1h")
        task = ScheduledTask(
            id="t3", name="Custom", goal="Test",
            schedule=s, thread_id="custom_thread"
        )
        assert task.thread_id == "custom_thread"

    def test_disabled_task(self):
        s = Schedule(kind="every", every="1m")
        task = ScheduledTask(
            id="t4", name="Disabled", goal="Nothing",
            schedule=s, enabled=False
        )
        assert task.enabled is False


# =============================================================================
# RunRecord Tests
# =============================================================================


class TestRunRecord:
    """Test RunRecord dataclass."""

    def test_creation(self):
        r = RunRecord(
            task_id="t1",
            started_at="2025-01-01T00:00:00",
            status="running",
        )
        assert r.task_id == "t1"
        assert r.status == "running"
        assert r.completed_at is None

    def test_completed_record(self):
        r = RunRecord(
            task_id="t1",
            started_at="2025-01-01T00:00:00",
            completed_at="2025-01-01T00:01:00",
            status="ok",
            result="Done",
            duration_ms=60000,
        )
        assert r.status == "ok"
        assert r.duration_ms == 60000


# =============================================================================
# SchedulerStore Tests
# =============================================================================


class TestSchedulerStore:
    """Test file-based persistence."""

    def test_save_and_load_tasks(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="persist1", name="Persistent", goal="Save me", schedule=s)
        store.save_tasks({"persist1": task})

        loaded = store.load_tasks()
        assert "persist1" in loaded
        assert loaded["persist1"].name == "Persistent"
        assert loaded["persist1"].schedule.every == "5m"

    def test_load_empty(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        loaded = store.load_tasks()
        assert loaded == {}

    def test_append_and_get_runs(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        for i in range(5):
            r = RunRecord(
                task_id="run_test",
                started_at=f"2025-01-01T00:0{i}:00",
                completed_at=f"2025-01-01T00:0{i}:30",
                status="ok",
                result=f"Result {i}",
                duration_ms=30000,
            )
            store.append_run(r)

        runs = store.get_runs("run_test", limit=3)
        assert len(runs) == 3
        # Should be the last 3
        assert runs[-1].result == "Result 4"

    def test_get_runs_nonexistent_task(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        runs = store.get_runs("nonexistent")
        assert runs == []

    def test_corrupted_tasks_file(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        # Write corrupted JSON
        with open(store.tasks_file, "w") as f:
            f.write("{invalid json")
        loaded = store.load_tasks()
        assert loaded == {}
        # Backup should exist
        backup = store.tasks_file.with_suffix(".json.bak")
        assert backup.exists()

    def test_prune_runs(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        # Write more than max_entries
        for i in range(15):
            r = RunRecord(
                task_id="prune_test",
                started_at=f"2025-01-01T00:{i:02d}:00",
                status="ok",
            )
            store.append_run(r)

        # Prune to 10
        run_file = store.runs_dir / "prune_test.jsonl"
        store._prune_runs(run_file, max_entries=10)

        runs = store.get_runs("prune_test", limit=100)
        assert len(runs) == 10


# =============================================================================
# Scheduler Engine Tests
# =============================================================================


class TestScheduler:
    """Test the Scheduler task management API."""

    def test_add_task(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="add1", name="Added", goal="Test add", schedule=s)
        result = scheduler.add(task)
        assert result.id == "add1"
        assert "add1" in scheduler.tasks

    def test_get_task(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="get1", name="Get Me", goal="Test get", schedule=s)
        scheduler.add(task)
        retrieved = scheduler.get("get1")
        assert retrieved is not None
        assert retrieved.name == "Get Me"

    def test_get_nonexistent(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        assert scheduler.get("nope") is None

    def test_update_task(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="upd1", name="Original", goal="Update me", schedule=s)
        scheduler.add(task)
        updated = scheduler.update("upd1", name="Updated")
        assert updated.name == "Updated"

    def test_update_nonexistent(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        assert scheduler.update("nope", name="Fail") is None

    def test_remove_task(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="rm1", name="Remove Me", goal="Bye", schedule=s)
        scheduler.add(task)
        assert scheduler.remove("rm1") is True
        assert "rm1" not in scheduler.tasks

    def test_remove_nonexistent(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        assert scheduler.remove("nope") is False

    def test_list_tasks(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        for i in range(3):
            s = Schedule(kind="every", every=f"{i+1}m")
            task = ScheduledTask(id=f"list{i}", name=f"Task {i}", goal="List test", schedule=s)
            scheduler.add(task)
        tasks = scheduler.list()
        assert len(tasks) == 3

    def test_list_excludes_disabled(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        s = Schedule(kind="every", every="5m")
        enabled = ScheduledTask(id="en1", name="Enabled", goal="Yes", schedule=s)
        disabled = ScheduledTask(id="dis1", name="Disabled", goal="No", schedule=s, enabled=False)
        scheduler.add(enabled)
        scheduler.add(disabled)

        tasks = scheduler.list(include_disabled=False)
        assert len(tasks) == 1
        assert tasks[0].id == "en1"

        tasks_all = scheduler.list(include_disabled=True)
        assert len(tasks_all) == 2

    @pytest.mark.asyncio
    async def test_execute_task_no_executor(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="noexec", name="No Executor", goal="Test fallback", schedule=s)
        scheduler.add(task)
        result = await scheduler.run_now("noexec", force=True)
        assert result["status"] == "ok"
        assert "No executor configured" in result["result"]

    @pytest.mark.asyncio
    async def test_execute_task_with_executor(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)

        async def mock_executor(task):
            return f"Executed: {task.goal}"

        scheduler = Scheduler(executor=mock_executor, store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="exec1", name="Exec Test", goal="Run me", schedule=s)
        scheduler.add(task)
        result = await scheduler.run_now("exec1", force=True)
        assert result["status"] == "ok"
        assert "Executed: Run me" in result["result"]

    @pytest.mark.asyncio
    async def test_run_now_nonexistent(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        result = await scheduler.run_now("nope")
        assert "error" in result

    @pytest.mark.asyncio
    async def test_execute_task_error(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)

        async def failing_executor(task):
            raise RuntimeError("Execution failed!")

        scheduler = Scheduler(executor=failing_executor, store=store)
        s = Schedule(kind="every", every="5m")
        task = ScheduledTask(id="fail1", name="Fail Test", goal="Crash", schedule=s)
        scheduler.add(task)
        result = await scheduler.run_now("fail1", force=True)
        assert result["status"] == "error"
        assert "Execution failed!" in result["error"]

    @pytest.mark.asyncio
    async def test_one_time_task_disables_after_run(self, scheduler_dir):
        store = SchedulerStore(base_dir=scheduler_dir)
        scheduler = Scheduler(store=store)
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        s = Schedule(kind="at", at=past_time)
        task = ScheduledTask(id="once1", name="One Time", goal="Run once", schedule=s)
        # Force the next_run_at to be set even though it's past
        task.next_run_at = past_time
        scheduler.tasks["once1"] = task
        result = await scheduler.run_now("once1", force=True)
        assert task.enabled is False


# =============================================================================
# parse_schedule Helper Tests
# =============================================================================


class TestParseSchedule:
    """Test the parse_schedule convenience function."""

    def test_dict_input(self):
        s = parse_schedule({"kind": "every", "every": "10m"})
        assert s.kind == "every"
        assert s.every == "10m"

    def test_in_relative(self):
        s = parse_schedule("in 5m")
        assert s.kind == "at"
        assert s.at is not None

    def test_at_absolute(self):
        s = parse_schedule("at 2030-06-15T09:00:00")
        assert s.kind == "at"
        assert "2030" in s.at

    def test_every_interval(self):
        s = parse_schedule("every 30s")
        assert s.kind == "every"
        assert s.every == "30s"

    def test_daily_at(self):
        s = parse_schedule("daily at 9:00")
        assert s.kind == "cron"
        assert "9" in s.cron

    def test_weekdays_at(self):
        s = parse_schedule("weekdays at 14:30")
        assert s.kind == "cron"
        assert "1-5" in s.cron

    def test_hourly_shortcut(self):
        s = parse_schedule("hourly")
        assert s.kind == "cron"
        assert s.cron == "0 * * * *"

    def test_daily_shortcut(self):
        s = parse_schedule("daily")
        assert s.kind == "cron"
        assert s.cron == "0 0 * * *"

    def test_raw_cron(self):
        s = parse_schedule("0 9 * * 1-5")
        assert s.kind == "cron"
        assert s.cron == "0 9 * * 1-5"


# =============================================================================
# _parse_time Helper Tests
# =============================================================================


class TestParseTime:
    """Test the _parse_time helper."""

    def test_24h_format(self):
        h, m = _parse_time("14:30")
        assert h == 14
        assert m == 30

    def test_12h_am(self):
        h, m = _parse_time("9am")
        assert h == 9
        assert m == 0

    def test_12h_pm(self):
        h, m = _parse_time("3pm")
        assert h == 15
        assert m == 0

    def test_12_am_is_midnight(self):
        h, m = _parse_time("12am")
        assert h == 0
        assert m == 0

    def test_12_pm_is_noon(self):
        h, m = _parse_time("12pm")
        assert h == 12
        assert m == 0

    def test_simple_hour(self):
        h, m = _parse_time("9")
        assert h == 9
        assert m == 0

    def test_with_colon_and_am(self):
        h, m = _parse_time("9:30am")
        assert h == 9
        assert m == 30


# =============================================================================
# create_task Helper Tests
# =============================================================================


class TestCreateTask:
    """Test the create_task convenience function."""

    def test_basic_creation(self):
        task = create_task("Daily Check", "Check emails", "daily at 9:00")
        assert task.name == "Daily Check"
        assert task.goal == "Check emails"
        assert task.schedule.kind == "cron"

    def test_with_custom_id(self):
        task = create_task("Test", "Do stuff", "every 5m", task_id="custom123")
        assert task.id == "custom123"

    def test_auto_generated_id(self):
        task = create_task("Test", "Do stuff", "every 5m")
        assert len(task.id) == 8

    def test_with_schedule_object(self):
        s = Schedule(kind="every", every="1h")
        task = create_task("Test", "Do stuff", s)
        assert task.schedule.every == "1h"
