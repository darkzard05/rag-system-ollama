"""Regression: sync_to_streamlit same-key lost update.

Root cause: src/core/session/manager.py sync_to_streamlit snapshotted V1,
released the per-session lock for UI writes, then consumed _dirty_keys from
the stale snapshot — a concurrent set(K, V2) was silently dropped (mirror=V1,
has_pending=False). Fix: _dirty_versions counter + version-checked consume.

Docs: Python threading.Lock only guarantees mutual exclusion inside the held
critical section; unlock-then-relock gaps are not atomic.

Determinism: threading.Event synchronizes the writer so the interleaving is
explicit, not timing-dependent. Session state is reset before/after to keep
test isolation.
"""

import threading
from types import SimpleNamespace
from unittest.mock import patch

from core.session import SessionManager


class _BlockingAdapter:
    def __init__(self) -> None:
        self.store: dict = {}
        self.writes: list = []
        self.entered = threading.Event()
        self.release = threading.Event()

    def write(self, key: str, val: object) -> None:
        self.writes.append((key, val))
        if key == "global_status":
            self.entered.set()
            assert self.release.wait(timeout=10), "sync write not released"
        self.store[key] = val

    def read(self, key: str, default: object = None) -> object:
        return self.store.get(key, default)

    def delete(self, key: str) -> None:
        self.store.pop(key, None)


def test_same_key_write_during_sync_not_lost() -> None:
    sid = "lost_update_repro"
    SessionManager.reset()
    try:
        SessionManager.init_session(sid)
        SessionManager.set("global_status", "V1", session_id=sid)

        adapter = _BlockingAdapter()
        with (
            patch.object(SessionManager, "_ui_sync", adapter),
            patch.object(SessionManager, "_is_streamlit_running", return_value=True),
            patch(
                "streamlit.runtime.scriptrunner.get_script_run_ctx",
                return_value=SimpleNamespace(session_id="t"),
            ),
        ):
            t = threading.Thread(
                target=lambda: SessionManager.sync_to_streamlit(session_id=sid),
                daemon=True,
            )
            t.start()
            assert adapter.entered.wait(timeout=10), "sync never reached UI write"
            # Concurrent same-key update while the first sync is blocked in write.
            SessionManager.set("global_status", "V2", session_id=sid)
            adapter.release.set()
            t.join(timeout=10)
            assert not t.is_alive(), "sync thread hung"

            assert SessionManager.get("global_status", session_id=sid) == "V2"
            # Either the mirror already has V2, or the second write is still
            # pending for the next sync — but it must never be silently dropped.
            if adapter.store.get("global_status") == "V2":
                pass
            else:
                assert adapter.store.get("global_status") == "V1"
                assert SessionManager.has_pending_ui_sync(session_id=sid) is True, (
                    "lost update: V2 never delivered yet marked clean "
                    f"(mirror={adapter.store.get('global_status')!r})"
                )
                SessionManager.sync_to_streamlit(session_id=sid)
                assert adapter.store.get("global_status") == "V2"
                assert SessionManager.has_pending_ui_sync(session_id=sid) is False
    finally:
        SessionManager.reset()
