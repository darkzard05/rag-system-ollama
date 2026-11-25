"""
시스템 알림 통합 관리 모듈
상태 로그(Status Box)와 로거(Logger)를 일원화하여 관리합니다.
"""

import logging

from core.session import SessionManager

logger = logging.getLogger(__name__)


class SystemNotifier:
    """
    시스템 상태 알림 및 로그를 중앙에서 관리하는 클래스
    상태 박스와 백엔드 로깅을 동시에 처리합니다.
    """

    # 기본 아이콘 매핑 (uiux-fix-p2: 사용자 표면 이모지 제거 — 빈 문자열로 접두사 미노출)
    ICONS = {
        "info": "",
        "success": "",
        "warning": "",
        "error": "",
    }

    @classmethod
    def _notify(
        cls,
        message: str,
        level: str = "info",
        icon: str | None = None,
    ) -> None:
        """내부 통합 알림 처리 로직."""
        if not icon:
            icon = cls.ICONS.get(level, "ℹ️")

        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

        full_message = f"{icon} {message}" if icon else message
        SessionManager.add_status_log(full_message)

    @classmethod
    def info(cls, message: str, icon: str | None = None):
        """일반 정보 알림"""
        cls._notify(message, "info", icon=icon)

    @classmethod
    def success(cls, message: str, icon: str | None = None):
        """성공 알림"""
        cls._notify(message, "success", icon=icon)

    @classmethod
    def warning(cls, message: str):
        """경고 알림"""
        cls._notify(message, "warning")

    @classmethod
    def error(cls, message: str, details: str | None = None):
        """에러 알림"""
        full_msg = f"{message}: {details}" if details else message
        cls._notify(full_msg, "error")
        SessionManager.set("last_error", full_msg)
