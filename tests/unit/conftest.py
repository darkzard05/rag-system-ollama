"""
Unit-test scoped conftest: per-test isolation for the global SessionManager.

The root ``tests/conftest.py`` sets up sys.path and auth isolation, but because
``tests`` is a package, its autouse fixtures are not reliably inherited by
``tests/unit`` tests. This conftest lives at the ``tests/unit`` level so the
reset runs for every unit test.

SessionManager uses a process-global dict as its primary store and a pluggable
UI-sync adapter (set via ``set_ui_sync`` when main.py is imported). Leftover
state — e.g. the grade-decision memo, or an attached StreamlitSessionSync
adapter — leaks across tests and causes order-dependent failures. We reset both
the fallback store and the adapter before/after each test.

NOTE: ``src.core.session.manager`` and ``core.session.manager`` can resolve to
two distinct module objects at runtime, so we reset both aliases.
"""

import sys

import pytest

from core.session import SessionManager  # noqa: E402

try:  # noqa: E402
    import src.core.session.manager as _src_mgr  # type: ignore
except Exception:  # pragma: no cover
    _src_mgr = None


@pytest.fixture(autouse=True)
def _reset_session_manager_per_test():
    """격리: 테스트 간 전역 세션 상태(_GRADE_MEMO_KEY, _ui_sync) 누수 차단."""
    SessionManager.reset()
    SessionManager.set_ui_sync(None)
    if _src_mgr is not None:
        _src_mgr.SessionManager.reset()
        _src_mgr.SessionManager.set_ui_sync(None)
    yield
    SessionManager.reset()
    SessionManager.set_ui_sync(None)
    if _src_mgr is not None:
        _src_mgr.SessionManager.reset()
        _src_mgr.SessionManager.set_ui_sync(None)
