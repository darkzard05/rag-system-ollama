# Design Spec: SessionStore + UI Bridge Decoupling

- **Topic**: Decoupling `SessionManager` into a Pure Python Store and a Streamlit UI Bridge.
- **Date**: 2026-06-02
- **Status**: Draft

## 1. Problem Statement
The current `SessionManager` is a "God Object" that tightly couples business logic state with the Streamlit framework. This makes it difficult to:
- Test core logic without a Streamlit environment.
- Safely update state from background threads without triggering "Missing ScriptRunContext" errors.
- Scale the system beyond a single UI framework.

## 2. Architecture

### 2.1 SessionStore (Pure Python)
The `SessionStore` is the Single Source of Truth (SSoT). It is entirely unaware of Streamlit.

- **Storage**: `dict[session_id, dict[key, value]]`
- **Concurrency**: 
    - `global_lock`: `threading.RLock` for protecting the session registry.
    - `session_locks`: `dict[session_id, threading.RLock]` for granular session access.
- **Interface**:
    - `get(session_id, key, default)`
    - `set(session_id, key, value)`
    - `update(session_id, key, func)`: Atomic update (e.g., appending to messages).
    - `get_metadata(session_id)`: Returns versioning/timestamp info for the UIBridge.

### 2.2 UIBridge (Framework-Specific)
The `UIBridge` acts as an observer that bridges the gap between the `SessionStore` and the Streamlit UI.

- **Mechanism**: Uses `@st.fragment(run_every=1.0)` as a heartbeat.
- **Responsibility**: 
    - Detects changes in `SessionStore` via timestamps.
    - Syncs specific "UI-reactive" keys to `st.session_state`.
    - Triggers `st.rerun()` if a full page refresh is needed.

### 2.3 ContextManager (Reliable Propagation)
A utility to handle the transfer of `session_id` and Streamlit context across thread boundaries.

- **Propagation Logic**:
    ```python
    def safe_spawn(target_func, *args, **kwargs):
        st_ctx = get_script_run_ctx()
        cv_ctx = contextvars.copy_context()
        
        def wrapped():
            add_script_run_ctx(threading.current_thread(), st_ctx)
            cv_ctx.run(target_func, *args, **kwargs)
            
        return threading.Thread(target=wrapped)
    ```

## 3. Data Flow
1. **Action**: Background thread updates `SessionStore` using `store.update("messages", append_func)`.
2. **Notification**: `SessionStore` updates the session's `last_updated` timestamp.
3. **Bridge**: `UIBridge` (running in a fragment) sees the new timestamp.
4. **Update**: `UIBridge` updates `st.session_state` and the user sees the new message.

## 4. Risks & Mitigations
| Risk | Mitigation |
| :--- | :--- |
| **Race Conditions** | Per-session `RLock` ensures only one thread modifies a session at a time. |
| **State Desync** | `SessionStore` is the ONLY source of truth. `st.session_state` is treated as a temporary UI view. |
| **Memory Leaks** | `SessionStore` implements `cleanup_expired_sessions` based on `last_accessed` timestamps. |

## 5. Success Criteria
- [ ] `SessionStore` can be unit tested without `import streamlit`.
- [ ] Background threads can update status logs without crashing.
- [ ] UI reflects background updates within 1 second.
