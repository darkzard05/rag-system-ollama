# Task 3: Implement TaskService (Async Utilities) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `TaskService` to provide utilities for status logging and managing background tasks within the session context.

**Architecture:** `TaskService` acts as a service layer over `SessionStore`, providing high-level methods for status logs and pending tasks. It ensures status logs are truncated and prevents duplicate sequential logs.

**Tech Stack:** Python, `SessionStore` (core component), `pytest` for testing.

---

### Task 1: Initialize TaskService Tests (TDD - Red)

**Files:**
- Create: `tests/unit/test_task_service.py`

- [ ] **Step 1: Write the failing tests**

```python
from src.infra.task_service import TaskService
from src.core.session.store import SessionStore
import pytest

def test_status_logging():
    store = SessionStore()
    service = TaskService(store)
    service.add_status_log("Starting...", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 1
    assert "Starting..." in logs[0]

def test_status_logging_prevents_duplicate_sequential():
    store = SessionStore()
    service = TaskService(store)
    service.add_status_log("Processing", session_id="s1")
    service.add_status_log("Processing", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 1

def test_status_logging_truncation():
    store = SessionStore()
    service = TaskService(store)
    for i in range(40):
        service.add_status_log(f"Log {i}", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 30
    assert "Log 39" in logs[-1]

def test_pending_tasks():
    store = SessionStore()
    service = TaskService(store)
    service.push_pending_task("rebuild", {"file": "test.pdf"}, session_id="s1")
    tasks = service.pop_pending_tasks(session_id="s1")
    assert len(tasks) == 1
    assert tasks[0]["type"] == "rebuild"
    assert service.pop_pending_tasks(session_id="s1") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_task_service.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.infra.task_service'`

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_task_service.py
git commit -m "test: add tests for TaskService"
```

### Task 2: Implement TaskService (Green)

**Files:**
- Create: `src/infra/task_service.py`

- [ ] **Step 1: Implement minimal TaskService**

```python
# 세션 기반 작업 및 상태 로그 관리 서비스
from datetime import datetime
from typing import List, Any, Dict
from src.core.session.store import SessionStore

class TaskService:
    def __init__(self, store: SessionStore):
        self.store = store

    def add_status_log(self, message: str, session_id: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs = self.store.get("status_logs", session_id, default=[])
        
        # Prevent duplicate sequential messages
        if logs and logs[-1].endswith(message):
            return
            
        logs.append(f"[{timestamp}] {message}")
        
        # Keep last 30 logs
        if len(logs) > 30:
            logs = logs[-30:]
            
        self.store.set("status_logs", logs, session_id)

    def get_status_logs(self, session_id: str) -> List[str]:
        return self.store.get("status_logs", session_id, default=[])

    def push_pending_task(self, task_type: str, payload: Any, session_id: str):
        tasks = self.store.get("pending_tasks", session_id, default=[])
        tasks.append({"type": task_type, "payload": payload})
        self.store.set("pending_tasks", tasks, session_id)

    def pop_pending_tasks(self, session_id: str) -> List[Dict[str, Any]]:
        tasks = self.store.get("pending_tasks", session_id, default=[])
        self.store.set("pending_tasks", [], session_id)
        return tasks
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/unit/test_task_service.py`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/infra/task_service.py
git commit -m "feat: implement TaskService"
```

### Task 3: Final Verification and Self-Review

- [ ] **Step 1: Run all unit tests**

Run: `pytest tests/unit`

- [ ] **Step 2: Update graphify**

Run: `graphify update .`

- [ ] **Step 3: Self-review**
Check if implementation matches requirements and matches coding standards (Korean header comment included).

- [ ] **Step 4: Commit final changes (if any)**
