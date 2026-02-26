"""
Streamlit 세션 상태 관리를 위한 SessionManager 클래스.
ThreadSafeSessionManager를 상속받아 세션 관리를 통합합니다.
"""

from core.thread_safe_session import ThreadSafeSessionManager


class SessionManager(ThreadSafeSessionManager):
    """
    세션 상태를 관리하는 통합 클래스.
    모든 핵심 로직은 ThreadSafeSessionManager에서 상속받습니다.
    """

    pass
