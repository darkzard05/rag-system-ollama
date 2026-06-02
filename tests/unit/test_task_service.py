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
