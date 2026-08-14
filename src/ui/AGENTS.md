# UI AGENTS

## OVERVIEW
Streamlit-based frontend orchestration for the RAG system, managing the two-column layout (PDF Viewer | Chat) and real-time state synchronization.

## STRUCTURE
- `ui.py`: Main layout orchestration and global CSS injection.
- `bridge.py`: Real-time synchronization between `SessionStore` and `st.session_state` (with interactive key preservation).
- `components/`: Modular UI elements.
    - `chat.py`: Unified conversational timeline (single-pass `@st.fragment(run_every=UI_TIMELINE_POLL_SECONDS)`), streaming message rendering, native status/expander components.
    - `streaming.py`: Async stream consumption bridge (`stream_chunks`) and message-driven streaming updates (`start_streaming_turn`, `_spawn_stream_consumer`).
    - `viewer.py`: PDF rendering fragment (`@st.fragment`), page navigation, annotation display.
    - `sidebar.py`: Global settings and session controls.
- `styles/`: CSS and theme assets.
    - `main.css`: Layout flex chain, responsive breakpoints, sticky chat input. Component styling delegated to native Streamlit elements.

*Removed:* `canvas.py`, `status_box.py`, `analysis.py` (dead code, replaced by message-driven timeline in `chat.py`).

## WHERE TO LOOK
| Issue | File | Notes |
|-------|------|-------|
| Layout/CSS | `src/ui/ui.py` + `src/ui/styles/main.css` | Flex chain selectors in CSS |
| PDF flickering | `src/ui/components/viewer.py` | `@st.fragment` wrapper |
| Chat timeline | `src/ui/components/chat.py` | Unified timeline fragment polls every 1.0s (`timeline_poll_seconds`) |
| Streaming updates | `src/ui/components/streaming.py` | Background thread updates message dicts |
| State sync | `src/ui/bridge.py` + `main.py` | `UIBridge.sync_session()` |
| Sidebar/settings | `src/ui/components/sidebar.py` | |

## CONVENTIONS
- **Fragment first**: Wrap localized UI sections in `@st.fragment` to prevent full-page reruns.
- **PDF + Chat**: Separate fragments — PDF navigation does not force chat rerun, and vice versa.
- **Message polling**: `_render_unified_timeline` runs every 1.0s (`timeline_poll_seconds`) to pick up new messages without explicit `st.rerun()`.
- **Everything is a message**: Document context, build progress/errors, logs, and streaming state are all stored as messages (`msg_type` discriminator) in `SessionManager.messages`.
- **Interactive key protection**: `UIBridge.sync_session()` preserves widget-bound keys during state sync.
- **Module-level callbacks**: Navigation `on_click` handlers are module-level (not per-render closures).
- **All UI in `src/ui/`**: Keep Streamlit-specific code out of `src/core/`.

## ANTI-PATTERNS
- **Full reruns for local state**: Use `@st.fragment` instead. Only `st.rerun()` when submitting a message or resetting.
- **`st.rerun()` after streaming**: Removed — the timeline fragment polls instead.
- **Direct `st.session_state` mutation**: Always go through `SessionManager` then `UIBridge`.
- **UI Logic in Core**: Keep all Streamlit-specific code within `src/ui/`.
- **Nested fragments**: `@st.fragment` cannot contain another `@st.fragment` — inline helper functions.
- **Bespoke HTML chrome**: Use native `st.status`, `st.expander`, `st.caption`, `st.popover` instead of custom HTML/CSS.