# 비동기 작업 및 상태 로그를 관리하는 서비스
from datetime import datetime

from src.core.session.store import SessionStore


class TaskService:
    def __init__(self, store: SessionStore):
        self.store = store

    def add_status_log(self, message: str, session_id: str):
        """세션에 상태 로그를 추가합니다. (최대 30개 유지, 중복 방지)"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        logs = self.store.get("status_logs", session_id, default=[])

        # 중복 로그 방지 (마지막 로그와 메시지가 같은 경우)
        if logs and logs[-1].split("] ", 1)[-1] == message:
            return

        logs.append(f"[{timestamp}] {message}")
        # 최대 30개 로그 유지
        self.store.set("status_logs", logs[-30:], session_id)

    def get_status_logs(self, session_id: str) -> list[str]:
        """세션의 상태 로그 목록을 반환합니다."""
        return self.store.get("status_logs", session_id, default=[])
