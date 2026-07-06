import threading
from typing import Any

# 전역 세션 상태 저장소
# SessionManager 클래스가 여러 번 로드되더라도 이 모듈은 단 한 번만 로드되어
# 모든 SessionManager 인스턴스가 동일한 데이터를 공유하도록 보장합니다.
fallback_sessions: dict[str, dict[str, Any]] = {}
session_locks: dict[str, threading.RLock] = {}
global_lock = threading.RLock()
