from src.infra.task_service import TaskService
from src.core.session.store import SessionStore

def test_status_logging():
    store = SessionStore()
    service = TaskService(store)
    service.add_status_log("Starting...", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 1
    assert "Starting..." in logs[0]

def test_status_logging_duplicate_prevention():
    store = SessionStore()
    service = TaskService(store)
    service.add_status_log("Task A", session_id="s1")
    service.add_status_log("Task A", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 1

def test_status_logging_limit():
    store = SessionStore()
    service = TaskService(store)
    for i in range(40):
        service.add_status_log(f"Step {i}", session_id="s1")
    logs = service.get_status_logs(session_id="s1")
    assert len(logs) == 30
    assert "Step 39" in logs[-1]
    assert "Step 0" not in logs[0]
