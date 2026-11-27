"""Panel callbacks: BuildState transitions only (no st.rerun)."""

from __future__ import annotations

import time

from core.session import SessionManager


def on_cancel(sid: str) -> None:
    """Request rebuild cancellation (post-callback auto-rerun applies it)."""
    SessionManager.set("rebuild_cancelled", True, session_id=sid)


def on_retry(sid: str) -> None:
    """Reset pipeline state and flag a rebuild (3-line minimal reset)."""
    SessionManager.reset_build_state(session_id=sid)
    SessionManager.set("pdf_processing_error", "", session_id=sid)
    SessionManager.set("rebuild_error", None, session_id=sid)
    SessionManager.set("rebuild_cancelled", False, session_id=sid)
    SessionManager.set("is_building_rag", False, session_id=sid)
    SessionManager.set("needs_rag_rebuild", True, session_id=sid)
    SessionManager.set("rebuild_start_time", time.time(), session_id=sid)
    try:
        attempt = int(SessionManager.get("rebuild_attempt", 0, sid) or 0)
    except (TypeError, ValueError):
        attempt = 0
    SessionManager.set("rebuild_attempt", attempt + 1, session_id=sid)
