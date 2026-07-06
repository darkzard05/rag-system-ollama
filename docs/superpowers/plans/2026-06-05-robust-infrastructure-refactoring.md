# RAG System Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the core infrastructure to remove risky monkey patches, ensure global VRAM protection across threads, and decouple session management for better stability and scalability.

**Architecture:** 
1. **Global Concurrency Control:** Replace `asyncio.Semaphore` with `threading.Semaphore` in `ModelManager` to protect VRAM across multiple threads/loops.
2. **Clean Worker Pattern:** Use `threading.Thread` with `asyncio.run()` for heavy tasks, eliminating `nest_asyncio` dependency and monkey patches.
3. **Session Decoupling:** Extract state storage into a pure Python `SessionStore` (SSoT) to remove tight coupling with Streamlit's `st.session_state`.

**Tech Stack:** Streamlit 1.54+, LangGraph 0.2.74, threading, asyncio.

---

### Task 1: Global VRAM Protection in ModelManager

**Files:**
- Modify: `src/core/model_loader.py`
- Test: `tests/unit/test_model_manager_threads.py`

- [ ] **Step 1: Write a test verifying that asyncio.Semaphore fails across threads**
```python
import asyncio
import threading
import pytest
from core.model_loader import ModelManager

@pytest.mark.asyncio
async def test_semaphore_isolation_across_threads():
    # This test should demonstrate that asyncio.Semaphore doesn't protect across threads
    pass # Implementation details in the task
```

- [ ] **Step 2: Replace asyncio.Semaphore with threading.Semaphore in ModelManager**
Modify `_inference_semaphore` to be a `threading.Semaphore(MAX_CONCURRENT_INFERENCE)`. Update `inference_session` to be a regular (not async) context manager or handle the transition.

- [ ] **Step 3: Update inference_session and acquire/release methods**
```python
    _inference_semaphore = threading.Semaphore(MAX_CONCURRENT_INFERENCE)

    @classmethod
    @contextlib.contextmanager # Changed from asynccontextmanager
    def inference_session(cls):
        cls._inference_semaphore.acquire()
        try:
            yield
        finally:
            cls._inference_semaphore.release()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

- [ ] **Step 4: Verify protection across multiple worker threads**
Run a test that starts 5 threads, each calling a mock inference, and verify that only `MAX_CONCURRENT_INFERENCE` are active at once.

---

### Task 2: Remove Monkey Patches and Implement Clean Worker Pattern

**Files:**
- Modify: `src/main.py`
- Modify: `src/common/utils.py`

- [ ] **Step 1: Remove `nest_asyncio` and `_patch_current_task_for_nest_asyncio` from `src/main.py`**
Delete the patch function and the `nest_asyncio.apply()` call.

- [ ] **Step 2: Implement a robust background worker utility**
In `src/common/utils.py`, add `run_in_background_worker(coro, session_id)` which:
1. Gets `get_script_run_ctx()`.
2. Starts `threading.Thread`.
3. Calls `add_script_run_ctx`.
4. Executes `asyncio.run(coro)`.
5. Calls `get_instance().request_rerun(session_id)` on completion.

- [ ] **Step 3: Refactor `_bg_rebuild_thread` in `src/main.py` to use the new utility**
Ensure it no longer manually manages loops or uses `nest_asyncio` logic.

- [ ] **Step 4: Verify that PDF processing still works without the patch**
Test the upload and indexing flow.

---

### Task 3: Decouple SessionManager into SessionStore

**Files:**
- Create: `src/core/session/store.py` (Expand existing)
- Modify: `src/core/session/manager.py`

- [ ] **Step 1: Complete `SessionStore` implementation**
Ensure it supports `RLock` per session and an atomic `update` method.

- [ ] **Step 2: Refactor `SessionManager` to use `SessionStore` as SSoT**
Modify `_get_state` to delegate to `SessionStore`. Ensure `st.session_state` is only used for UI mirroring.

- [ ] **Step 3: Implement `SessionBridge` (UI Fragment)**
In `src/ui/ui.py` or a new component, add a fragment that periodically syncs `SessionStore` -> `st.session_state`.

- [ ] **Step 4: Run integration tests**
Verify that multi-user sessions are isolated and stable.

---

### Task 4: Parallel Grading in LangGraph

**Files:**
- Modify: `src/core/graph_builder.py`

- [ ] **Step 1: Refactor `grade_documents` to use internal parallelism**
Use `asyncio.gather` for LLM calls to grade documents.

- [ ] **Step 2: Implement Early Exit logic**
If any document is graded as relevant, cancel pending tasks and proceed.

- [ ] **Step 3: Integrate StreamWriter for real-time status**
Update the UI with "Evaluating doc X/Y..." using `writer`.

- [ ] **Step 4: Verify performance improvement**
Benchmark against the old sequential version.
