"""
Concurrency stress test for SessionManager shared-state safety (F3).

Proves that the global-dict single-source-of-truth session store in
src/core/session/manager.py is safe under concurrent interleaved access from
multiple threads that all target ONE session id.

Design goals (determinism, not luck):
- Every thread targets the SAME fixed session id, so all of them serialize on
  the SAME per-session ``threading.Lock`` (``_acquire_lock``). There is no
  cross-session map mutation contention in this test; the map is only touched
  under ``_map_lock`` during initial session creation, which is also serialized.
- Each mutating call is internally atomic (a single ``_acquire_lock`` scope that
  covers the full read-modify-write). We therefore never build a read-modify-
  write cycle ACROSS two separate lock acquisitions in the test itself, which
  would be a test-side race rather than a store bug.
- Totals stay under the in-store caps (``MAX_MESSAGE_HISTORY == 100``,
  status_logs capped at 30) so final counts are exactly predictable and the
  assertions are not perturbed by trimming or consecutive-dedup behavior.

The test asserts three things:
1. No lost updates: the final ``messages`` and ``status_logs`` lists contain
   exactly one entry per ``add_message`` / ``add_status_log`` call, with unique
   ids and no duplicates (a duplicated id would prove a non-atomic append).
2. No exceptions escape any worker or reader thread.
3. Reads always observe a consistent, internally-valid snapshot: every snapshot
   taken by a reader has distinct message ids and a length within the cap.
"""

from __future__ import annotations

import threading

import pytest

from common.constants import MAX_MESSAGE_HISTORY
from core.session.manager import SessionManager

# One fixed session forces all threads onto the same per-session lock.
SESSION_ID = "f3_concurrency_stress_session"

# Caps in the store (kept in sync with manager.py / common.constants).
STATUS_LOG_CAP = 30

# Stress sizing: keep totals under the caps so final counts are exact.
# - messages: 16 * 5 = 80 <= MAX_MESSAGE_HISTORY (100)
# - status_logs: one per thread = 16 <= STATUS_LOG_CAP (30)
N_THREADS = 16
MSG_PER_THREAD = 5

TOTAL_MSG = N_THREADS * MSG_PER_THREAD
TOTAL_STATUS = N_THREADS


def _worker(thread_idx: int, errors: list[Exception]) -> None:
    """Interleave set / add_message / add_status_log on the shared session."""
    try:
        for op in range(MSG_PER_THREAD):
            msg_id = f"t{thread_idx}_m{op}"
            SessionManager.add_message(
                role="user",
                content=f"thread {thread_idx} op {op}",
                msg_type="general",
                session_id=SESSION_ID,
                msg_id=msg_id,
            )
            # A per-thread scalar write to exercise the set() path.
            SessionManager.set(
                f"thread_flag_{thread_idx}",
                op,
                session_id=SESSION_ID,
            )
        # One status log per thread (kept under the 30-entry store cap so the
        # final count is exactly predictable and not perturbed by trimming).
        SessionManager.add_status_log(
            f"t{thread_idx}_status",
            session_id=SESSION_ID,
        )
    except Exception as exc:  # noqa: BLE001 - capture any escaping error
        errors.append(exc)


def _reader(errors: list[Exception], stop: threading.Event) -> None:
    """Continuously read snapshots and assert each is internally consistent."""
    try:
        while not stop.is_set():
            msgs = SessionManager.get("messages", [], session_id=SESSION_ID)
            assert isinstance(msgs, list)
            # Materialize under the GIL; a snapshot must never contain dup ids.
            snap = tuple(msgs)
            ids = [m.get("msg_id") for m in snap if isinstance(m, dict)]
            assert len(ids) == len(set(ids)), "duplicate msg_id in snapshot"
            assert len(snap) <= MAX_MESSAGE_HISTORY

            logs = SessionManager.get("status_logs", [], session_id=SESSION_ID)
            assert isinstance(logs, list)
            assert len(logs) <= STATUS_LOG_CAP
    except Exception as exc:  # noqa: BLE001 - capture any escaping error
        errors.append(exc)


@pytest.fixture(autouse=True)
def _clean_session():
    SessionManager.reset()
    yield
    SessionManager.reset()


def test_concurrent_session_access_no_lost_updates():
    errors: list[Exception] = []

    # Reader threads run concurrently with the writers.
    stop = threading.Event()
    readers = [
        threading.Thread(target=_reader, args=(errors, stop), name=f"reader-{i}")
        for i in range(2)
    ]

    writers = [
        threading.Thread(target=_worker, args=(t, errors), name=f"writer-{t}")
        for t in range(N_THREADS)
    ]

    for t in readers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in readers:
        t.join()

    # 1. No exceptions from any thread.
    assert not errors, f"exceptions raised during concurrency: {errors}"

    # 2. No lost updates: counts match exactly and ids are unique.
    final_msgs = SessionManager.get("messages", [], session_id=SESSION_ID)
    assert isinstance(final_msgs, list)
    assert len(final_msgs) == TOTAL_MSG, (
        f"message count drift: expected {TOTAL_MSG}, got {len(final_msgs)}"
    )
    msg_ids = [m["msg_id"] for m in final_msgs]
    assert len(msg_ids) == len(set(msg_ids)), "duplicate msg_id in final state"

    final_logs = SessionManager.get("status_logs", [], session_id=SESSION_ID)
    assert isinstance(final_logs, list)
    assert len(final_logs) == TOTAL_STATUS, (
        f"status_log count drift: expected {TOTAL_STATUS}, got {len(final_logs)}"
    )

    # The per-thread scalar was written by every writer (last write wins, but
    # the key must exist with a value one of the writers produced).
    for t in range(N_THREADS):
        assert SessionManager.get(f"thread_flag_{t}", None, session_id=SESSION_ID) in (
            list(range(MSG_PER_THREAD))
        )


def test_concurrent_dirty_key_snapshot_is_atomic():
    """The dirty-key snapshot in sync_to_streamlit must not raise under load.

    With ``_ui_sync is None`` the method is a no-op for mirroring, but it still
    acquires the per-session lock and copies ``_dirty_keys`` under it. That
    snapshot path is the one called out in the concurrency contract, so we
    exercise it concurrently with writers to prove it stays exception-free.
    """
    errors: list[Exception] = []
    stop = threading.Event()

    def _sync_spammer(stop: threading.Event) -> None:
        try:
            while not stop.is_set():
                SessionManager.sync_to_streamlit(session_id=SESSION_ID)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    spammers = [
        threading.Thread(target=_sync_spammer, args=(stop,), name=f"sync-{i}")
        for i in range(4)
    ]
    writers = [
        threading.Thread(target=_worker, args=(t, errors), name=f"writer-{t}")
        for t in range(N_THREADS)
    ]

    for t in spammers + writers:
        t.start()
    for t in writers:
        t.join()
    stop.set()
    for t in spammers:
        t.join()

    assert not errors, f"exceptions during dirty-key snapshot stress: {errors}"
    # Writers still landed their updates despite concurrent sync spammers.
    final_msgs = SessionManager.get("messages", [], session_id=SESSION_ID)
    assert len(final_msgs) == TOTAL_MSG
