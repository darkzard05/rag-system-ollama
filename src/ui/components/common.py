"""Shared UI helpers for the RAG Streamlit frontend.

Centralizes cross-component UI idioms that were previously re-implemented
(verbatim) in multiple component files, so chat / viewer / streaming stop
drifting apart. Covers:

- ``AVATARS`` / ``role_avatar`` — canonical chat-message avatars (D3)
- ``get_doc_metadata`` — document metadata extraction (D2, 3x dup)
- ``status_line`` — middot-joined status fragments (D6/D7/C3)
- ``ui_error`` / ``show_pdf_error`` — unified, friendly-only error exposure (C1/D1)
- ``navigate_to_page`` — single page-jump helper (D4, manager-only)
"""

from __future__ import annotations

import contextlib
import time

import streamlit as st
from streamlit.errors import StreamlitAPIException

from core.session import SessionManager
from ui.strings import t
from ui.widget_keys import PDF_NAV_INPUT_KEY

# ---------------------------------------------------------------------------
# Avatars (D3): single source of truth across chat.py / viewer.py
# ---------------------------------------------------------------------------

AVATARS: dict[str, str] = {
    "assistant": "🤖",
    "user": "👤",
    "document": "📄",
    "building": "🔄",
    "error": "❌",
}


def role_avatar(role: str) -> str:
    """Return the canonical avatar emoji for a chat role."""
    return AVATARS.get(role, "👤")


# ---------------------------------------------------------------------------
# Document metadata extraction (D2): was duplicated 3x in chat/streaming
# ---------------------------------------------------------------------------


def get_doc_metadata(doc: object) -> dict:
    """Extract the ``metadata`` mapping from a document-like object.

    Handles both attribute-style (LangChain ``Document``) and
    mapping-style payloads. Returns ``{}`` when nothing usable is found.
    """
    if hasattr(doc, "metadata"):
        meta = doc.metadata  # type: ignore[attr-defined]
        return meta if isinstance(meta, dict) else {}
    if isinstance(doc, dict):
        meta = doc.get("metadata")
        return meta if isinstance(meta, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Status fragment joiner (D6/D7/C3)
# ---------------------------------------------------------------------------


def status_line(*parts: str) -> str:
    """Join non-empty status fragments with a middot.

    Used for "Answer complete · N references", "{file} · Ready",
    "Time · Retrieved · Model", etc. Empty fragments are skipped so callers
    don't need to guard every piece.
    """
    return " · ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Error exposure (C1/D1): unified, friendly-only
# ---------------------------------------------------------------------------


def ui_error(message: str) -> None:
    """Render a user-facing error through one consistent path.

    Never pass raw exception text here — surface a friendly constant instead.
    Centralizing the call keeps error phrasing/exposure policy in one place.
    """
    st.error(message)


def show_pdf_error(kind: str = "open") -> None:
    """Render the canonical PDF error message (D1).

    kind="open" -> file cannot be opened (corrupt / unsupported).
    kind="data" -> PDF bytes could not be loaded.
    """
    if kind == "data":
        st.error(t("pdf_error_data"))
        return
    st.error(t("pdf_error_open"))


# ---------------------------------------------------------------------------
# Navigation (D4): single page-jump helper shared by viewer callbacks
# ---------------------------------------------------------------------------


def navigate_to_page(page: int) -> None:
    """Single page-jump path: SessionManager keys only (shared by viewer nav).

    Writes ``current_page`` (clamped to >= 1; the upper bound is clamped by
    callers holding the cached total) and ``manual_nav_ts``. Also syncs the
    widget-bound nav input key (``PDF_NAV_INPUT_KEY``) so the viewer Page
    ``number_input`` displays the jumped page instead of staying stale: this
    write happens in callback phase (before the next script run instantiates
    the widget), which is the sanctioned Streamlit pattern — the bridge sync
    mirrors store keys only and never overwrites widget keys. Never calls
    ``st.rerun`` (Task2) — the nav input widget receives its default from the
    manager at render time (``value=current_page`` in ``render_pdf_controls``).
    """
    target = max(1, page)
    SessionManager.set("current_page", target)
    SessionManager.set("manual_nav_ts", time.time())
    with contextlib.suppress(AttributeError, RuntimeError, StreamlitAPIException):
        st.session_state[PDF_NAV_INPUT_KEY] = target
