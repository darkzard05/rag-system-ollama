"""Routing audit (FIXED): ``session_id=`` kwarg routes SessionManager.set().

6-site fix (all ``SessionManager.set(..., current_sid=...)`` renamed to
``session_id=...`` — keyword form; positional ``session_id`` uses elsewhere
were already correct):

- ``src/ui/components/streaming.py:234`` — ``set("is_generating_answer",
  False, session_id=sid)`` (except path)
- ``src/ui/components/streaming.py:252`` — ``set("is_generating_answer",
  False, session_id=sid)`` (cancel/normal path)
- ``src/ui/components/streaming.py:268`` — ``set("generation_cancel",
  False, session_id=sid)``
- ``src/ui/components/chat.py:280`` — ``set("is_generating_answer", False,
  session_id=current_sid)`` (timeline finally-guard)
- ``src/ui/components/chat.py:674`` — ``set("is_generating_answer", False,
  session_id=current_sid)`` (stream teardown)
- ``src/ui/components/chat.py:675`` — ``set("generation_cancel", False,
  session_id=current_sid)`` (stream teardown)

Root cause (fixed at call sites): ``SessionManager.set`` signature is
``set(key, value, session_id=None, **kwargs)`` (manager.py:387-393) and
resolves ``sid = session_id or cls.get_session_id()`` (manager.py:395).
The old ``current_sid=`` kwarg fell into ``**kwargs`` (manager.py:397) and
was stored as a *data key* in the *active* session. ``get``/``delete`` use
``session_id`` correctly (manager.py:331-344, 411-419); ``rag_core.py``
294-300 uses ``session_id=self.session_id``. Single-session manual runs
masked this via the ContextVar fallback (``get_session_id``).

These tests assert the FIXED routing via ``session_id=`` and are GREEN.
"""

from __future__ import annotations

import pytest

from core.session.manager import SessionManager


@pytest.fixture(autouse=True)
def _reset_manager() -> object:
    SessionManager.reset()
    yield
    SessionManager.reset()


def _activate(session_id: str) -> None:
    SessionManager.init_session(session_id)
    SessionManager.set_session_id(session_id)


def test_set_with_session_id_kwarg_routes_to_intended_session() -> None:
    """set(k, v, session_id=A) under active B lands in A, not B."""
    _activate("session-A")
    _activate("session-B")  # active session is now B
    assert SessionManager.get_session_id() == "session-B"

    SessionManager.set("probe_key", "probe-value", session_id="session-A")

    routed = SessionManager.get("probe_key", None, session_id="session-A", create=False)
    leaked = SessionManager.get("probe_key", None, session_id="session-B", create=False)
    stray = SessionManager.get(
        "current_sid", None, session_id="session-B", create=False
    )
    assert routed == "probe-value", f"intended session-A missing value (got {routed!r})"
    assert leaked is None, f"cross-session miswrite: active B got {leaked!r}"
    assert stray is None, f"stray data key 'current_sid' stored in B: {stray!r}"


def test_flag_reset_with_session_id_kwarg_hits_intended_session() -> None:
    """Flag-reset: clearing A while B active clears A, leaves B intact."""
    _activate("session-A")
    SessionManager.set("is_generating_answer", True, session_id="session-A")
    _activate("session-B")
    SessionManager.set("is_generating_answer", True, session_id="session-B")

    # Fixed pattern (mirrors streaming.py:252 / chat.py:674 teardown).
    SessionManager.set("is_generating_answer", False, session_id="session-A")

    flag_a = SessionManager.get(
        "is_generating_answer", None, session_id="session-A", create=False
    )
    flag_b = SessionManager.get(
        "is_generating_answer", None, session_id="session-B", create=False
    )
    assert flag_a is False, f"session-A flag not reset (got {flag_a!r})"
    assert flag_b is True, f"session-B flag clobbered by A's reset (got {flag_b!r})"


def test_session_id_kwarg_round_trip_control() -> None:
    """Control: correct ``session_id=`` kwarg routes across sessions."""
    _activate("session-A")
    _activate("session-B")

    SessionManager.set("control_key", "control-value", session_id="session-A")

    assert (
        SessionManager.get("control_key", None, session_id="session-A", create=False)
        == "control-value"
    )
    assert (
        SessionManager.get("control_key", None, session_id="session-B", create=False)
        is None
    )
