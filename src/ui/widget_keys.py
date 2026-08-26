"""Central registry for Streamlit widget keys and session-state keys.

Single source of truth for:
- Widget-key factories (e.g. ``cancel_rebuild_key``).
- The interactive-key protection set consumed by ``UIBridge.sync_session()``
  (see ``ui/bridge.py``).
- Named session-state keys that were previously scattered as bare string
  literals (C7), so every key has one definition to grep and change.

Transient widget keys (cancel/jump) are intentionally NOT part of
``INTERACTIVE_KEYS``: they are bound to buttons whose widget state is
re-created identically on every rerun, and the session store never writes to
them (``SessionManager.set``/``add_message`` never touch these keys), so
``sync_session()`` can never overwrite them. Adding them to the protection set
would only add needless snapshot/restore work and could resurrect a stale
clicked state after a sync.
"""

# ---------------------------------------------------------------------------
# Session-state keys (C7): every bare-string key lives here, once.
# ---------------------------------------------------------------------------

# Fired once per session to avoid re-injecting the global CSS/JS script.

# Main chat-input widget — also in INTERACTIVE_KEYS.
MAIN_CHAT_INPUT_KEY: str = "main_chat_input"
# PDF nav input widget — also in INTERACTIVE_KEYS (v6 is the live generation).
PDF_NAV_INPUT_KEY: str = "pdf_nav_input_v6"
# One-shot page-jump token set by chat reference buttons.
PDF_TARGET_PAGE_KEY: str = "pdf_target_page"
# Timestamp of the user's last manual navigation (used to invalidate auto jumps).
MANUAL_NAV_TS_KEY: str = "manual_nav_ts"

# Keys bound to interactive widgets (uploaders, selectors, chat box, PDF nav)
# that must not be overwritten by the session-store sync during a rerun to
# prevent flicker (cursor jumping, input reset).
INTERACTIVE_KEYS: frozenset[str] = frozenset(
    {
        "pdf_uploader",
        "model_selector",
        "embedding_model_selector",
        "main_chat_input",
        PDF_NAV_INPUT_KEY,
    }
)


def cancel_rebuild_key(sid: str) -> str:
    """Widget key for the RAG rebuild cancel button (components/chat.py)."""
    return f"cancel_rebuild_{sid}"


def jump_key(msg_id: str, page: int, idx: int) -> str:
    """Widget key for a reference page-jump button (components/chat.py)."""
    return f"jump_{msg_id}_{page}_{idx}"


def pdf_viewer_key(file_hash: str, page: int) -> str:
    """Widget key for the PDF viewer component (components/viewer.py).

    페이지를 키에 포함해 페이지 변경 시 컴포넌트가 재마운트되도록 한다
    (streamlit-pdf-viewer 0.0.30 프론트엔드는 pages_to_render 변경을 감시하지 않음).
    """
    return f"pdf_v8_{file_hash}_{page}"
