"""Shared UI helpers for the RAG Streamlit frontend.

Centralizes cross-component UI idioms that were previously re-implemented
(verbatim) in multiple component files, so chat / viewer / streaming stop
drifting apart. Covers:

- ``AVATARS`` / ``role_avatar`` — canonical chat-message avatars (D3)
- ``get_doc_metadata`` — document metadata extraction (D2, 3x dup)
- ``status_line`` — middot-joined status fragments (D6/D7/C3)
- ``ui_error`` / ``show_pdf_error`` — unified, friendly-only error exposure (C1/D1)
- ``navigate_to_page`` — single page-jump helper (D4)
"""

from __future__ import annotations

import time

import streamlit as st

from core.session import SessionManager
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
        st.error("⚠️ PDF 데이터를 불러올 수 없습니다.")
        return
    st.error(
        "⚠️ PDF 파일을 열 수 없습니다. 파일이 손상되었거나 지원되지 않는 형식입니다."
    )


# ---------------------------------------------------------------------------
# Navigation (D4): single page-jump helper shared by viewer callbacks
# ---------------------------------------------------------------------------


def navigate_to_page(page: int) -> None:
    """Update current page + sync nav input widget (shared by viewer nav).

    Keeps ``current_page``, ``manual_nav_ts`` and the ``pdf_nav_input_v6``
    widget key in lockstep so manual navigation never desyncs the input box.
    """
    SessionManager.set("current_page", page)
    SessionManager.set("manual_nav_ts", time.time())
    st.session_state[PDF_NAV_INPUT_KEY] = page
