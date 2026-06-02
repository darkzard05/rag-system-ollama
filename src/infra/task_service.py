# 세션 기반 작업 및 상태 로그 관리 서비스
from datetime import datetime
from typing import Any

from src.core.session.store import SessionStore


class TaskService:
    def __init__(self, store: SessionStore):
        self.store = store

    def add_status_log(self, message: str, session_id: str):
        """세션에 상태 로그를 추가합니다. (최대 30개 유지, 중복 방지)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs = self.store.get("status_logs", session_id, default=[])

        # 중복 로그 방지 (마지막 로그와 메시지가 같은 경우)
        if logs and logs[-1].endswith(message):
            return

        logs.append(f"[{timestamp}] {message}")
        # 최대 30개 로그 유지
        self.store.set("status_logs", logs[-30:], session_id)

    def get_status_logs(self, session_id: str) -> list[str]:
        """세션의 상태 로그 목록을 반환합니다."""
        return self.store.get("status_logs", session_id, default=[])

    def push_pending_task(self, task_type: str, payload: Any, session_id: str):
        """대기 중인 작업을 큐에 추가합니다."""
        tasks = self.store.get("pending_tasks", session_id, default=[])
        tasks.append({"type": task_type, "payload": payload})
        self.store.set("pending_tasks", tasks, session_id)

    def pop_pending_tasks(self, session_id: str) -> list[dict[str, Any]]:
        """대기 중인 작업을 모두 가져오고 큐를 비웁니다."""
        tasks = self.store.get("pending_tasks", session_id, default=[])
        self.store.set("pending_tasks", [], session_id)
        return tasks
