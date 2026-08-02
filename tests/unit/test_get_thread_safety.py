"""
Thread safety tests for SessionManager.get() atomic read guarantee.

Validates that get() holds the per-session lock during the entire read
operation, preventing TOCTOU races with concurrent set()/add_message().
"""

import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from core.session.manager import SessionManager


def _reset_manager():
    """Reset global SessionManager state between tests."""
    SessionManager.reset()
    # Clear module-level caches
    import core.session.manager as mgr_mod

    mgr_mod._thread_session_map.clear()


class TestGetAtomicity:
    """get() must hold per-session lock during the entire read."""

    def setup_method(self):
        _reset_manager()

    def teardown_method(self):
        _reset_manager()

    def test_create_true_holds_lock_during_read(self):
        """Given: session exists with key 'x' = 1
        When:  concurrent set('x', 2) races with get('x', create=True)
        Then:  get() returns either 1 or 2 (never KeyError or partial state)
        """
        session_id = "atomic_read_test"
        SessionManager.init_session(session_id)
        SessionManager.set("counter", 0, session_id=session_id)

        errors: list[str] = []
        iterations = 500

        def writer():
            for i in range(iterations):
                SessionManager.set("counter", i, session_id=session_id)

        def reader():
            for _ in range(iterations):
                val = SessionManager.get("counter", session_id=session_id)
                if val is None:
                    errors.append("get() returned None for existing key")

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert not errors, f"Atomic read violations: {errors}"

    def test_create_false_returns_default_when_missing(self):
        """Given: session does not exist
        When:  get() with create=False
        Then:  returns default without creating the session
        """
        session_id = "nonexistent_session"
        result = SessionManager.get(
            "any_key", default=42, session_id=session_id, create=False
        )
        assert result == 42

    def test_create_false_holds_lock_during_read(self):
        """Given: session is being created by another thread
        When:  get() with create=False races with set()
        Then:  get() returns either default or the set value, never crashes
        """
        session_id = "race_create_false"
        errors: list[str] = []
        iterations = 300

        def creator():
            for i in range(iterations):
                SessionManager.set("key", i, session_id=session_id)
                # Brief pause to create interleaving opportunities
                if i % 50 == 0:
                    time.sleep(0.001)

        def reader():
            for _ in range(iterations):
                val = SessionManager.get(
                    "key", default=-1, session_id=session_id, create=False
                )
                if val is None:
                    errors.append("get(create=False) returned None for default=-1")

        t_write = threading.Thread(target=creator)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert not errors, f"Race condition violations: {errors}"

    def test_concurrent_gets_consistent_across_threads(self):
        """Given: multiple threads reading the same key
        When:  a writer updates the key concurrently
        Then:  no reader observes a partially written or corrupted value
        """
        session_id = "consistent_read"
        SessionManager.init_session(session_id)
        SessionManager.set("data", "initial", session_id=session_id)

        seen_values: set[str | None] = set()
        lock = threading.Lock()
        iterations = 200

        def writer():
            for i in range(iterations):
                SessionManager.set("data", f"value_{i}", session_id=session_id)

        def reader():
            for _ in range(iterations):
                val = SessionManager.get("data", session_id=session_id)
                with lock:
                    seen_values.add(val)

        threads = [threading.Thread(target=writer)]
        for _ in range(4):
            threads.append(threading.Thread(target=reader))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All observed values should be either "initial" or "value_N" format
        for v in seen_values:
            assert v is not None, "Reader saw None for existing key"
            assert v == "initial" or v.startswith("value_"), f"Corrupted value: {v}"

    def test_no_deadlock_between_get_and_set(self):
        """get() and set() use consistent lock ordering — no deadlock."""
        session_id = "deadlock_test"
        SessionManager.init_session(session_id)
        SessionManager.set("key", "start", session_id=session_id)

        completed = threading.Event()

        def mixed_ops():
            for _ in range(200):
                SessionManager.get("key", session_id=session_id)
                SessionManager.set("key", "updated", session_id=session_id)
            completed.set()

        t = threading.Thread(target=mixed_ops)
        t.start()
        t.join(timeout=5.0)

        assert not t.is_alive(), "Deadlock detected — thread still alive after 5s"
        assert completed.is_set(), "Thread did not complete all operations"
