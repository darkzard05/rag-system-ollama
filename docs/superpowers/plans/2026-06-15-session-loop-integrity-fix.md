# Session & Loop Integrity Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve Session State synchronization issues between background threads and Streamlit, and fix Event Loop mismatch errors in LangGraph/LangChain execution.

**Architecture:** 
1. **Thread-Safe Session Sync:** Implement a "Pull-based" sync mechanism. `SessionManager` will update a global `_fallback_sessions` store. `st.session_state` will only be updated from the UI thread via `sync_to_streamlit()`.
2. **Loop-Aware Engine Factory:** Implement a mechanism to detect `asyncio` loop changes and re-compile the LangGraph graph on the current running loop to prevent "attached to a different loop" errors.

**Tech Stack:** Python, Streamlit, LangChain, LangGraph, asyncio

---

### Task 1: Foundation - Thread-Safe Session Manager Refactoring

**Files:**
- Modify: `src/core/session/manager.py`
- Test: `tests/unit/test_session_sync.py`

- [ ] **Step 1: Write a test to reproduce the thread-unsafe `set` behavior.**

```python
# Create tests/unit/test_session_sync.py
import threading
from src.core.session.manager import SessionManager
import pytest

def test_thread_safe_global_state():
    """Verify that background threads update global store without touching streamlit state."""
    SessionManager.init_session("test_session")
    
    def worker():
        SessionManager.set("bg_key", "bg_value", session_id="test_session")
        
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    
    # Value should be in global store
    assert SessionManager.get("bg_key", session_id="test_session") == "bg_value"
```

- [ ] **Step 2: Run the test to verify current behavior.**

Run: `pytest tests/unit/test_session_sync.py`
Expected: PASS (but we want to ensure `st.session_state` is NOT touched in threads in the next step).

- [ ] **Step 3: Refactor `SessionManager.set` to avoid direct `st.session_state` access in non-UI threads.**

```python
# src/core/session/manager.py:165 (approx)
    @classmethod
    def set(cls, key: str | None = None, value: Any = None, session_id: str | None = None, **kwargs):
        sid = session_id or cls.get_session_id()
        state = cls._get_state(sid)
        updates = kwargs.copy()
        if key is not None:
            updates[key] = value

        with cls._acquire_lock(sid):
            for k, v in updates.items():
                state[k] = v

        # ONLY update st.session_state if we are in the main Streamlit thread
        if cls._is_streamlit_running():
            try:
                from streamlit.runtime.scriptrunner import get_script_run_ctx
                if get_script_run_ctx(): # This check ensures we are in a UI thread
                    for k, v in updates.items():
                        st.session_state[k] = v
            except Exception:
                pass
```

- [ ] **Step 4: Update `sync_to_streamlit` to handle list/dict references safely.**

```python
# src/core/session/manager.py:126 (approx)
    @classmethod
    def sync_to_streamlit(cls, session_id: str | None = None):
        if not cls._is_streamlit_running():
            return
        # ... (check context)
        sid = session_id or cls.get_session_id()
        state = cls._get_state(sid)
        
        # Keys to sync from global store to UI
        sync_keys = ["pdf_processed", "is_generating_answer", "status_logs", "messages", "current_page"]
        
        for k in sync_keys:
            if k in state:
                # Use copy for mutable objects to avoid shared reference issues
                val = state[k]
                if isinstance(val, (list, dict)):
                    import copy
                    st.session_state[k] = copy.copy(val)
                else:
                    st.session_state[k] = val
```

- [ ] **Step 5: Commit.**

```bash
git add src/core/session/manager.py tests/unit/test_session_sync.py
git commit -m "refactor: implement thread-safe pull-based session sync"
```

---

### Task 2: Loop Integrity - RAG Core Engine Factory

**Files:**
- Modify: `src/core/rag_core.py`
- Test: `tests/unit/test_loop_integrity.py`

- [ ] **Step 1: Write a test to reproduce the loop mismatch error.**

```python
# Create tests/unit/test_loop_integrity.py
import asyncio
import pytest
from src.core.rag_core import RAGSystem

@pytest.mark.asyncio
async def test_engine_recompilation_on_loop_change():
    """Verify that RAGSystem re-compiles the graph when event loop changes."""
    rag = RAGSystem(session_id="test_loop")
    
    # Initialize engine in current loop
    engine1 = await rag._get_rag_engine()
    loop1 = asyncio.get_running_loop()
    
    # Simulate a new loop (e.g., a different thread's loop)
    new_loop = asyncio.new_event_loop()
    try:
        async def run_in_new_loop():
            # This call should trigger re-compilation
            engine2 = await rag._get_rag_engine()
            return engine2
            
        engine2 = new_loop.run_until_complete(run_in_new_loop())
        assert engine1 is not engine2 # Should be a fresh instance/compilation
    finally:
        new_loop.close()
```

- [ ] **Step 2: Run the test to verify it fails.**

Run: `pytest tests/unit/test_loop_integrity.py`
Expected: FAIL (if current logic doesn't correctly detect and re-compile).

- [ ] **Step 3: Refactor `_get_rag_engine` to be more robust.**

```python
# src/core/rag_core.py:145 (approx)
    async def _get_rag_engine(self) -> Any:
        try:
            current_loop = asyncio.get_running_loop()
            current_loop_id = id(current_loop)
        except RuntimeError:
            current_loop_id = 0

        rag_engine = SessionManager.get("rag_engine", session_id=self.session_id)
        cached_loop_id = SessionManager.get("rag_engine_loop_id", 0, session_id=self.session_id)

        # Re-compile if loop changed or engine is missing
        if not rag_engine or cached_loop_id != current_loop_id:
            logger.info(f"[RAG] Re-compiling graph for loop {current_loop_id}")
            from core.graph_builder import build_graph
            rag_engine = build_graph() # This recompiles the state graph
            
            SessionManager.set("rag_engine", rag_engine, session_id=self.session_id)
            SessionManager.set("rag_engine_loop_id", current_loop_id, session_id=self.session_id)
            
        return rag_engine
```

- [ ] **Step 4: Run tests to verify fix.**

Run: `pytest tests/unit/test_loop_integrity.py`
Expected: PASS

- [ ] **Step 5: Commit.**

```bash
git add src/core/rag_core.py tests/unit/test_loop_integrity.py
git commit -m "fix: implement loop-aware graph engine factory"
```

---

### Task 3: UI Integration and Verification

**Files:**
- Modify: `src/main.py`
- Test: Manual verification with Streamlit

- [ ] **Step 1: Ensure `sync_to_streamlit` is called at the start of `main()`.**

```python
# src/main.py:270 (approx)
def main() -> None:
    from core.session import SessionManager
    from ui.ui import inject_custom_css

    # 1. Initialize Session
    SessionManager.init_session()

    # 2. Sync global store to Streamlit (CRITICAL for background updates)
    SessionManager.sync_to_streamlit()
    
    # ... rest of main ...
```

- [ ] **Step 2: Verify `astream_events` uses the updated engine factory.**

Check `src/core/rag_core.py:astream_events` to ensure it calls `await self._get_rag_engine()` inside the `_consumer` generator if possible, or right before.

- [ ] **Step 3: Run full integration test suite.**

Run: `pytest tests/test_p0_2_session_sync.py` (Existing integration test)
Expected: PASS

- [ ] **Step 4: Commit.**

```bash
git add src/main.py
git commit -m "feat: complete UI-Background sync bridge"
```

---

### Final Verification

1.  Run all unit and integration tests.
2.  Start Streamlit app: `streamlit run src/main.py`
3.  Upload a PDF and observe if status logs update smoothly without UI freezing.
4.  Ask questions and verify streaming output works consistently.
