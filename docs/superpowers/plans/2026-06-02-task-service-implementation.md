# TaskService Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement TaskService to manage status logs and pending tasks using SessionStore.

**Architecture:** TaskService will be an infrastructure service that abstracts session-specific task and log management, ensuring thread-safe access through the underlying SessionStore.

**Tech Stack:** Python, pytest

---

### Task 1: Setup and Red Phase (Status Logging)

**Files:**
- Create: `tests/unit/test_task_service.py`
- Test: `tests/unit/test_task_service.py`

- [ ] **Step 1: Write the failing test for status logging**

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

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_task_service.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'infra.task_service')

---

### Task 2: Green Phase (Status Logging Implementation)

**Files:**
- Create: `src/infra/task_service.py`
- Test: `tests/unit/test_task_service.py`

- [ ] **Step 1: Implement TaskService with add_status_log and get_status_logs**

```python
# 비동기 작업 및 상태 로그를 관리하는 서비스
from datetime import datetime
from typing import List, Any
from core.session.store import SessionStore

class TaskService:
    def __init__(self, store: SessionStore):
        self.store = store

    def add_status_log(self, message: str, session_id: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs = self.store.get("status_logs", session_id, default=[])
        # 중복 로그 방지
        if logs and logs[-1].endswith(message):
            return
        logs.append(f"[{timestamp}] {message}")
        # 최대 30개 로그 유지
        self.store.set("status_logs", logs[-30:], session_id)

    def get_status_logs(self, session_id: str) -> List[str]:
        return self.store.get("status_logs", session_id, default=[])
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/unit/test_task_service.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/infra/task_service.py tests/unit/test_task_service.py
git commit -m "feat(infra): implement TaskService for status logging"
```

---

### Task 3: Verification and Finalization

**Files:**
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [ ] **Step 1: Run all unit tests to ensure no regressions**

Run: `pytest tests/unit -v`

- [ ] **Step 2: Update checklist.md and context-notes.md**

- [ ] **Step 3: Run graphify update**

Run: `graphify update .`
