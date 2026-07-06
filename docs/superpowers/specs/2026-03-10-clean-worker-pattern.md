# Design: Clean Worker Pattern and Monkey Patch Removal

**Goal**: Modernize the RAG system's async handling by removing risky `nest_asyncio` patches and implementing a robust background worker utility for Streamlit.

## Architecture

We are moving away from "nested loops" in the same thread towards a "dedicated worker thread with its own loop" model for heavy tasks.

### 1. Robust Worker Utility
A new utility `run_in_background_worker` will be added to `src/common/utils.py`. It will:
- Capture the Streamlit `ScriptRunContext`.
- Launch a new `threading.Thread`.
- Attach the context to the new thread so Streamlit functions (like `st.session_state` or `st.toast`) work correctly (though we mostly use `SessionManager` for state).
- Use `asyncio.run()` to execute a coroutine in isolation.
- Trigger a session rerun using `streamlit.runtime.get_instance().request_rerun(session_id)` upon completion.

### 2. Removal of async patches
The following will be removed from `src/main.py`:
- `nest_asyncio` import and application.
- `_patch_current_task_for_nest_asyncio` workaround.

### 3. Refactoring RAG Rebuild
The `_bg_rebuild_thread` function in `src/main.py` will be refactored into an async task and launched via the new worker utility.

## Implementation Details

### `src/common/utils.py`
```python
def run_in_background_worker(coro, session_id: str):
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
    from streamlit.runtime import get_instance
    import threading
    import asyncio
    from core.session import SessionManager

    ctx = get_script_run_ctx()

    def _wrapper():
        # 1. Attach Streamlit context
        add_script_run_ctx(threading.current_thread(), ctx)
        
        # 2. Set session ID for isolation
        SessionManager.set_session_id(session_id)
        
        # 3. Run the coroutine
        try:
            asyncio.run(coro)
        except Exception as e:
            logger.error(f"Background worker error: {e}", exc_info=True)
        finally:
            # 4. Trigger rerun
            instance = get_instance()
            if instance:
                instance.request_rerun(session_id)

    threading.Thread(target=_wrapper, daemon=True).start()
```

### `src/main.py`
- Remove all `nest_asyncio` related code.
- Refactor `_bg_rebuild_thread` to `async def _bg_rebuild_task(...)`.
- Update `_handle_pending_tasks` to use `run_in_background_worker`.

## Verification Plan
1. **Static Analysis**: Ensure no syntax errors and all imports are correct.
2. **Functional Test**: Upload a PDF and verify that indexing starts, progress is logged, and the UI reruns upon completion.
3. **Isolation Test**: (Optional) Verify that multiple sessions can run background workers independently.
