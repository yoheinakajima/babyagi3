"""
Unit tests for utils/ modules

Tests cover:
- EventEmitter: on, off, emit, once, wildcard, decorator syntax
- ThreadSafeList: append, pop, getitem, setitem, len, iter, copy, bool
"""

import threading
import time

import pytest

from utils.events import EventEmitter
from utils.collections import ThreadSafeList


# =============================================================================
# EventEmitter Tests
# =============================================================================


class TestEventEmitter:
    """Test EventEmitter pub/sub system."""

    def test_on_and_emit(self):
        emitter = EventEmitter()
        received = []
        emitter.on("test", lambda data: received.append(data))
        emitter.emit("test", {"value": 42})
        assert len(received) == 1
        assert received[0]["value"] == 42

    def test_emit_includes_event_name(self):
        emitter = EventEmitter()
        received = []
        emitter.on("test", lambda data: received.append(data))
        emitter.emit("test", {"value": 1})
        assert received[0]["_event"] == "test"

    def test_multiple_handlers(self):
        emitter = EventEmitter()
        results = []
        emitter.on("test", lambda d: results.append("a"))
        emitter.on("test", lambda d: results.append("b"))
        emitter.emit("test")
        assert results == ["a", "b"]

    def test_off_specific_handler(self):
        emitter = EventEmitter()
        results = []
        handler = lambda d: results.append("called")
        emitter.on("test", handler)
        emitter.off("test", handler)
        emitter.emit("test")
        assert results == []

    def test_off_all_handlers(self):
        emitter = EventEmitter()
        results = []
        emitter.on("test", lambda d: results.append("a"))
        emitter.on("test", lambda d: results.append("b"))
        emitter.off("test")
        emitter.emit("test")
        assert results == []

    def test_off_nonexistent_event(self):
        emitter = EventEmitter()
        # Should not raise
        emitter.off("nonexistent")

    def test_once(self):
        emitter = EventEmitter()
        results = []
        emitter.once("test", lambda d: results.append("once"))
        emitter.emit("test")
        emitter.emit("test")
        assert results == ["once"]  # Only fired once

    def test_wildcard_handler(self):
        emitter = EventEmitter()
        received = []
        emitter.on("*", lambda data: received.append(data["_event"]))
        emitter.emit("event_a", {"x": 1})
        emitter.emit("event_b", {"x": 2})
        assert received == ["event_a", "event_b"]

    def test_handler_error_doesnt_crash(self):
        emitter = EventEmitter()
        results = []

        def bad_handler(d):
            raise ValueError("handler error")

        def good_handler(d):
            results.append("ok")

        emitter.on("test", bad_handler)
        emitter.on("test", good_handler)
        emitter.emit("test")
        # good_handler should still execute
        assert results == ["ok"]

    def test_emit_default_data(self):
        emitter = EventEmitter()
        received = []
        emitter.on("test", lambda d: received.append(d))
        emitter.emit("test")  # No data arg
        assert received[0]["_event"] == "test"

    def test_on_returns_handler(self):
        emitter = EventEmitter()
        handler = lambda d: None
        result = emitter.on("test", handler)
        assert result is handler

    def test_decorator_syntax(self):
        emitter = EventEmitter()
        results = []

        @emitter.on("test")
        def handler(data):
            results.append(data["value"])

        emitter.emit("test", {"value": "decorated"})
        assert results == ["decorated"]

    def test_mixin_usage(self):
        """Test EventEmitter as a mixin class."""
        class MyClass(EventEmitter):
            def __init__(self):
                self.__init_events__()
                self.data = []

            def do_work(self):
                self.emit("work_done", {"result": "success"})

        obj = MyClass()
        obj.on("work_done", lambda d: obj.data.append(d["result"]))
        obj.do_work()
        assert obj.data == ["success"]

    def test_multiple_events(self):
        emitter = EventEmitter()
        a_results = []
        b_results = []
        emitter.on("a", lambda d: a_results.append(1))
        emitter.on("b", lambda d: b_results.append(1))
        emitter.emit("a")
        emitter.emit("b")
        emitter.emit("a")
        assert len(a_results) == 2
        assert len(b_results) == 1


# =============================================================================
# ThreadSafeList Tests
# =============================================================================


class TestThreadSafeList:
    """Test ThreadSafeList thread-safe operations."""

    def test_append_and_len(self):
        tsl = ThreadSafeList()
        tsl.append("a")
        tsl.append("b")
        assert len(tsl) == 2

    def test_getitem(self):
        tsl = ThreadSafeList()
        tsl.append("x")
        tsl.append("y")
        assert tsl[0] == "x"
        assert tsl[1] == "y"

    def test_setitem(self):
        tsl = ThreadSafeList()
        tsl.append("old")
        tsl[0] = "new"
        assert tsl[0] == "new"

    def test_pop(self):
        tsl = ThreadSafeList()
        tsl.append("a")
        tsl.append("b")
        assert tsl.pop() == "b"
        assert len(tsl) == 1

    def test_pop_index(self):
        tsl = ThreadSafeList()
        tsl.append("a")
        tsl.append("b")
        tsl.append("c")
        assert tsl.pop(0) == "a"
        assert len(tsl) == 2

    def test_iteration(self):
        tsl = ThreadSafeList()
        for i in range(5):
            tsl.append(i)
        items = list(tsl)
        assert items == [0, 1, 2, 3, 4]

    def test_iteration_returns_copy(self):
        """Iteration should return a copy, safe for concurrent modification."""
        tsl = ThreadSafeList()
        tsl.append("a")
        for item in tsl:
            tsl.append("b")  # Should not cause infinite loop
        assert len(tsl) == 2

    def test_copy(self):
        tsl = ThreadSafeList()
        tsl.append(1)
        tsl.append(2)
        c = tsl.copy()
        assert c == [1, 2]
        assert isinstance(c, list)

    def test_bool_empty(self):
        tsl = ThreadSafeList()
        assert bool(tsl) is False

    def test_bool_nonempty(self):
        tsl = ThreadSafeList()
        tsl.append("x")
        assert bool(tsl) is True

    def test_concurrent_append(self):
        """Test that concurrent appends don't lose data."""
        tsl = ThreadSafeList()
        n_threads = 10
        n_items = 100

        def append_items():
            for i in range(n_items):
                tsl.append(i)

        threads = [threading.Thread(target=append_items) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(tsl) == n_threads * n_items

    def test_index_error(self):
        tsl = ThreadSafeList()
        with pytest.raises(IndexError):
            _ = tsl[0]

    def test_pop_empty(self):
        tsl = ThreadSafeList()
        with pytest.raises(IndexError):
            tsl.pop()
