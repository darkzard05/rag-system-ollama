# 프레임워크에 독립적인 세션 데이터 저장소 클래스.
import threading
from typing import Any


class SessionStore:
    def __init__(self):
        self._sessions: dict[str, dict[str, Any]] = {}
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
