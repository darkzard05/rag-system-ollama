"""
시스템 알림 통합 관리 모듈
Streamlit Toast, 상태 로그(Status Box), 로거(Logger)를 일원화하여 관리합니다.
"""

import logging

from core.session import SessionManager

logger = logging.getLogger(__name__)


class SystemNotifier:
    """
    시스템 상태 알림 및 로그를 중앙에서 관리하는 클래스
    UI 컴포넌트(Toast, Status Box)와 백엔드 로깅을 동시에 처리합니다.
    """

    # 기본 아이콘 매핑 (uiux-fix-p2: 사용자 표면 이모지 제거 — 빈 문자열로 접두사 미노출)
    ICONS = {
        "info": "",
        "success": "",
        "warning": "",
        "error": "",
        "loading": "",
        "brain": "",
        "file": "",
        "setting": "",
    }

    @classmethod
    def _notify(
        cls,
        message: str,
        level: str = "info",
        show_toast: bool = False,  # 하위 호환성을 위해 유지하되 무시
        icon: str | None = None,
        duration: int = 4000,
        add_to_chat: bool = True,
    ) -> None:
        """내부 통합 알림 처리 로직 (이제 토스트를 사용하지 않음)"""

        # 1. 아이콘 결정
        if not icon:
            icon = cls.ICONS.get(level, "ℹ️")

        # 2. 백엔드 로깅 (콘솔/파일)
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

        # [수정] 이제 add_status_log 내부에서 add_message를 호출하므로 중복 방지를 위해 로직 통합
        # 아이콘을 메시지 앞에 붙여서 전달하면 상태 박스와 채팅창 모두에 아이콘이 표시됩니다.
        full_message = f"{icon} {message}" if icon else message
        SessionManager.add_status_log(full_message)

        # 4. [제거됨] Streamlit Toast 알림은 더 이상 사용하지 않음
        # 모든 알림은 채팅창이나 사이드바의 캡션 등을 통해 전달됨

    @classmethod
    def info(cls, message: str, show_toast: bool = False, icon: str | None = None):
        """일반 정보 알림"""
        cls._notify(message, "info", show_toast=False, icon=icon)

    @classmethod
    def success(cls, message: str, show_toast: bool = False, icon: str | None = None):
        """성공 알림"""
        cls._notify(message, "success", show_toast=False, icon=icon)

    @classmethod
    def warning(cls, message: str, show_toast: bool = False):
        """경고 알림"""
        cls._notify(message, "warning", show_toast=False)

    @classmethod
    def error(cls, message: str, details: str | None = None, show_toast: bool = False):
        """에러 알림"""
        full_msg = f"{message}: {details}" if details else message
        cls._notify(full_msg, "error", show_toast=False)
        SessionManager.set("last_error", full_msg)

    @classmethod
    def loading(cls, message: str, show_toast: bool = False):
        """로딩/작업 진행 중 알림"""
        cls._notify(message, "info", show_toast=False, icon=cls.ICONS["loading"])

    @classmethod
    def model_load(cls, model_name: str, device: str = "GPU"):
        """모델 로딩 전용 알림"""
        icon = cls.ICONS["brain"] if device == "GPU" else cls.ICONS["setting"]
        msg = f"추론 모델 로드 시작 ({device})"
        cls._notify(msg, "info", show_toast=False, icon=icon)
