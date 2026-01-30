"""
Streamlit 세션 상태 관리를 위한 SessionManager 클래스.
Thread-safe 구현: ThreadSafeSessionManager를 기반으로 동시성 안전성 제공.
"""

import logging

from common.typing_utils import SessionKey, SessionValue
from core.thread_safe_session import ThreadSafeSessionManager

logger = logging.getLogger(__name__)


class SessionManager(ThreadSafeSessionManager):
    """
    세션 상태를 관리하는 클래스 (ThreadSafeSessionManager 기반)

    기능:
    - Thread-safe read/write operations
    - Atomic updates
    - Lock-based concurrency protection
    - Streamlit session_state와의 통합
    """

    @classmethod
    def is_ready_for_chat(cls) -> bool:
        """
        채팅 준비 상태를 확인합니다 (캐시됨).

        Returns:
            bool: PDF 처리 완료, 에러 없음, QA 체인 준비 시 True.
        """
        return super().is_ready_for_chat()

    @classmethod
    def get(cls, key: str, default: SessionValue | None = None) -> SessionValue | None:
        """
        세션 상태에서 값을 가져옵니다.
        """
        return super().get(key, default)

    @classmethod
    def get_messages(cls) -> list[dict[str, str]]:
        """
        메시지 목록을 가져옵니다.
        """
        return super().get_messages()

    @classmethod
    def set(cls, key: SessionKey, value: SessionValue) -> None:
        """
        세션 상태에 값을 설정합니다.
        """
        super().set(key, value)

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        """
        특정 세션을 삭제합니다.
        """
        return super().delete_session(session_id)

    @classmethod
    def reset_all_state(cls):
        """
        세션의 모든 상태를 기본값으로 리셋합니다.
        """
        super().reset_all_state()
