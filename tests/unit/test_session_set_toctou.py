"""F12: SessionManager.set lock-order (TOCTOU) guard tests.

The structural test is the RED/regression guard: ``set`` must acquire the
per-session lock BEFORE reading state, mirroring ``add_message``. Today the
``_get_state`` call happens outside the lock, so a concurrent
``delete_session``/``reset_all_state`` can detach the session between the
state read and the lock acquisition, causing a silent write loss.

The functional test is a smoke guard: concurrent delete/reset during ``set``
must never crash or corrupt state.
"""

import threading
import time

import pytest

from core.session.manager import SessionManager


@pytest.fixture(autouse=True)
def reset_session_manager():
    SessionManager.reset()
    yield
    SessionManager.reset()


def test_set_acquires_lock_before_get_state():
    calls: list[str] = []
    orig_lock = SessionManager._acquire_lock.__func__
    orig_get = SessionManager._get_state.__func__

    def spy_lock(cls, sid):
        calls.append("lock")
        return orig_lock(cls, sid)

    def spy_get(cls, sid, **kw):
        calls.append("get")
        return orig_get(cls, sid, **kw)

    SessionManager._acquire_lock = classmethod(spy_lock)
    SessionManager._get_state = classmethod(spy_get)
    try:
        SessionManager.set("k", "v", session_id="toctou-test")
        assert calls.index("lock") < calls.index("get"), f"order was {calls}"
    finally:
        SessionManager._acquire_lock = classmethod(orig_lock)
        SessionManager._get_state = classmethod(orig_get)
        SessionManager.delete_session("toctou-test")


def test_set_concurrent_delete_no_crash():
    errors: list[Exception] = []

    def writer():
        try:
            for i in range(200):
                SessionManager.set("heartbeat", i, session_id="racy")
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=writer)
    t.start()
    for _ in range(10):
        SessionManager.delete_session("racy")
        time.sleep(0.001)
    t.join(timeout=10)

    assert not t.is_alive(), "writer thread hung"
    assert not errors, f"unexpected exception in writer: {errors}"

    value = SessionManager.get("heartbeat", session_id="racy", create=False)
    assert value is None or isinstance(value, int)
