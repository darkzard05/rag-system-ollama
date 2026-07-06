# SessionManager Refactoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the monolithic `SessionManager` into modular services (`SessionStore`, `ContextManager`, `TaskService`, `UIBridge`) to decouple core logic from Streamlit.

**Architecture:** A service-oriented approach where the data layer (`SessionStore`) is framework-agnostic, and the UI layer (`UIBridge`) handles framework-specific synchronization. Context is managed via `contextvars`.

**Tech Stack:** Python 3.11+, `contextvars`, `threading.RLock`, `streamlit`.

---

### Task 1: Implement SessionStore (Pure Data Layer)

**Files:**
- Create: `src/core/session/store.py`
- Test: `tests/unit/test_session_store.py`

- [ ] **Step 1: Write unit tests for SessionStore**
```python
import pytest
from core.session.store import SessionStore

def test_store_set_get():
    store = SessionStore()
    store.set("key1", "value1", session_id="s1")
    assert store.get("key1", session_id="s1") == "value1"
    assert store.get("key1", session_id="s2") is None

def test_store_clear():
    store = SessionStore()
    store.set("key1", "value1", session_id="s1")
    store.clear("s1")
    assert store.get("key1", session_id="s1") is None
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/unit/test_session_store.py`
Expected: FAIL (Module not found)

- [ ] **Step 3: Implement SessionStore**
```python
import threading
from typing import Any, Dict, Optional

class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get(self, key: str, session_id: str, default: Any = None) -> Any:
        with self._lock:
            return self._sessions.get(session_id, {}).get(key, default)

    def set(self, key: str, value: Any, session_id: str):
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = {}
            self._sessions[session_id][key] = value

    def delete(self, key: str, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].pop(key, None)

    def clear(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/unit/test_session_store.py`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/core/session/store.py tests/unit/test_session_store.py
git commit -m "feat(session): add SessionStore for framework-agnostic data management"
```

### Task 2: Implement ContextManager (Infrastructure)

**Files:**
- Create: `src/core/session/context.py`
- Test: `tests/unit/test_session_context.py`

- [ ] **Step 1: Write unit tests for ContextManager**
```python
from core.session.context import ContextManager
import threading

def test_context_binding():
    ContextManager.set_current_session_id("test_id")
    assert ContextManager.get_current_session_id() == "test_id"

def test_thread_isolation():
    def worker():
        ContextManager.set_current_session_id("thread_id")
        assert ContextManager.get_current_session_id() == "thread_id"
    
    ContextManager.set_current_session_id("main_id")
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert ContextManager.get_current_session_id() == "main_id"
```

- [ ] **Step 2: Run test to verify it fails**
Expected: FAIL

- [ ] **Step 3: Implement ContextManager**
```python
from contextvars import ContextVar
from typing import Optional

_current_session_id: ContextVar[Optional[str]] = ContextVar("current_session_id", default=None)

class ContextManager:
    @staticmethod
    def get_current_session_id() -> Optional[str]:
        return _current_session_id.get()

    @staticmethod
    def set_current_session_id(session_id: str):
        _current_session_id.set(session_id)
```

- [ ] **Step 4: Run test to verify it passes**
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/core/session/context.py tests/unit/test_session_context.py
git commit -m "feat(session): add ContextManager using contextvars for Session ID tracking"
```

### Task 3: Implement TaskService (Async Utilities)

**Files:**
- Create: `src/infra/task_service.py`
- Test: `tests/unit/test_task_service.py`

- [ ] **Step 1: Write tests for TaskService**
```python
from infra.task_service import TaskService
from core.session.store import SessionStore

def test_status_logging():
    store = SessionStore()
    service = TaskService(store)
    service.add_status_log("Starting...", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 1
    assert "Starting..." in logs[0]
```

- [ ] **Step 2: Run test and FAIL**

- [ ] **Step 3: Implement TaskService**
```python
from datetime import datetime
from typing import List, Any
from core.session.store import SessionStore

class TaskService:
    def __init__(self, store: SessionStore):
        self.store = store

    def add_status_log(self, message: str, session_id: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs = self.store.get("status_logs", session_id, default=[])
        logs.append(f"[{timestamp}] {message}")
        self.store.set("status_logs", logs, session_id)

    def get_status_logs(self, session_id: str) -> List[str]:
        return self.store.get("status_logs", session_id, default=[])
```

- [ ] **Step 4: Run test and PASS**

- [ ] **Step 5: Commit**
```bash
git add src/infra/task_service.py tests/unit/test_task_service.py
git commit -m "feat(infra): add TaskService for async status logging"
```

### Task 4: Implement UIBridge (Streamlit Sync)

**Files:**
- Create: `src/ui/bridge.py`
- Modify: `src/main.py` (partial)

- [ ] **Step 1: Implement UIBridge class**
```python
import streamlit as st
from core.session.store import SessionStore
from core.session.context import ContextManager

class UIBridge:
    def __init__(self, store: SessionStore):
        self.store = store

    def initialize_session(self):
        if "session_id" not in st.session_state:
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        
        sid = st.session_state.session_id
        ContextManager.set_current_session_id(sid)

    def sync_store_to_ui(self):
        sid = ContextManager.get_current_session_id()
        # Mirror common keys to st.session_state for UI reactivity
        keys_to_sync = ["messages", "status_logs", "is_generating_answer"]
        for key in keys_to_sync:
            val = self.store.get(key, sid)
            if val is not None:
                st.session_state[key] = val
```

- [ ] **Step 2: Commit UIBridge**
```bash
git add src/ui/bridge.py
git commit -m "feat(ui): add UIBridge for Streamlit state synchronization"
```

### Task 5: Integration & Legacy Migration

**Files:**
- Modify: `src/main.py`, `src/core/rag_core.py`
- Delete: `src/core/session.py` (legacy)

- [ ] **Step 1: Replace SessionManager in RAGCore**
Update `src/core/rag_core.py` to use `ContextManager` and `TaskService`.

- [ ] **Step 2: Replace SessionManager in main.py**
Initialize `UIBridge` and replace `SessionManager` calls.

- [ ] **Step 3: Verify end-to-end functionality**
Run the Streamlit app and perform a RAG query.

- [ ] **Step 4: Delete legacy SessionManager and Commit**
```bash
git rm src/core/session.py
git commit -m "refactor: complete SessionManager migration and remove legacy code"
```
