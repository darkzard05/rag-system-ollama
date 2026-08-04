"""Central registry for Streamlit widget keys used across the UI.

Single source of truth for widget-key factories and for the interactive-key
protection set consumed by ``UIBridge.sync_session()`` (see ``ui/bridge.py``).

Transient widget keys (cancel/jump) are intentionally NOT part of
``INTERACTIVE_KEYS``: they are bound to buttons whose widget state is
re-created identically on every rerun, and the session store never writes to
them (``SessionManager.set``/``add_message`` never touch these keys), so
``sync_session()`` can never overwrite them. Adding them to the protection set
would only add needless snapshot/restore work and could resurrect a stale
clicked state after a sync.
"""

# Keys bound to interactive widgets (uploaders, selectors, chat box, PDF nav)
# that must not be overwritten by the session-store sync during a rerun to
# prevent flicker (cursor jumping, input reset).
INTERACTIVE_KEYS: frozenset[str] = frozenset(
    {
        "pdf_uploader",
        "model_selector",
        "embedding_model_selector",
        "main_chat_input",
        "pdf_nav_input_v6",
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
