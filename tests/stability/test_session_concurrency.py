# tests/stability/test_session_concurrency.py
import threading
from concurrent.futures import ThreadPoolExecutor
from src.core.session import SessionManager
import pytest


def test_session_state_concurrency():
    # Initialize a test session
    sid = "test_concurrent_sid"
    SessionManager.init_session(sid)

    # Reset counter
    SessionManager.set("counter", 0, session_id=sid)

    def update_task(i):
        # Read current value, increment, and set back
        # Note: This is a classic race condition if not protected by a lock
        # However, SessionManager.set itself is locked, but the read-modify-write
        # pattern needs a higher level lock or an atomic operation if we want exact results.
        # But here we just want to see if the system crashes or remains stable.
        try:
            val = SessionManager.get("counter", session_id=sid)
            SessionManager.set("counter", val + 1, session_id=sid)
            return True
        except Exception as e:
            return e

    # Run 10 threads doing 100 updates each
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(update_task, range(1000)))

    # Check for errors
    errors = [r for r in results if r is not True]
    assert len(errors) == 0, f"Errors occurred during concurrency test: {errors}"

    # Check final value.
    # Because our update_task is NOT atomic (get and set are separate locked calls),
    # the final value might not be 1000, but it should be stable.
    final_val = SessionManager.get("counter", session_id=sid)
    print(f"Final counter value: {final_val} (expected <= 1000 due to non-atomic RMW)")
    assert final_val > 0
