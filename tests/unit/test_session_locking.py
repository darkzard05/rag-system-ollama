import threading

import pytest

from core.session import SessionManager


@pytest.fixture(autouse=True)
def reset():
    SessionManager.reset()
    yield
    SessionManager.reset()


def test_per_session_isolation():
    """Two sessions should not deadlock each other."""
    completed = {"user_1": False, "user_2": False}

    def user_task(session_id: str):
        SessionManager.set_session_id(session_id)
        SessionManager.init_session()
        SessionManager.set("done", True)
        completed[session_id] = True

    t1 = threading.Thread(target=user_task, args=("user_1",))
    t2 = threading.Thread(target=user_task, args=("user_2",))

    t1.start()
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive(), "t1 deadlocked"
    assert not t2.is_alive(), "t2 deadlocked"
    assert completed["user_1"]
    assert completed["user_2"]
