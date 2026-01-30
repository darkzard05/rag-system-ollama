"""
시스템 알림 통합 관리 모듈
Streamlit Toast, 상태 로그(Status Box), 로거(Logger)를 일원화하여 관리합니다.
"""

import logging

import streamlit as st

from core.session import SessionManager

logger = logging.getLogger(__name__)


class SystemNotifier:
    """
    시스템 상태 알림 및 로그를 중앙에서 관리하는 클래스
    UI 컴포넌트(Toast, Status Box)와 백엔드 로깅을 동시에 처리합니다.
    """

    # 기본 아이콘 매핑
    ICONS = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "loading": "⏳",
        "brain": "🧠",
        "file": "📄",
        "setting": "⚙️",
    }

    @classmethod
    def _notify(
        cls,
        message: str,
        level: str = "info",
        show_toast: bool = False,
        icon: str | None = None,
        duration: int = 4000,
    ) -> None:
        """내부 통합 알림 처리 로직"""

        # 1. 아이콘 결정
        if not icon:
            icon = cls.ICONS.get(level, "ℹ️")

        # 2. 백엔드 로깅 (콘솔/파일)
        log_msg = f"[{level.upper()}] {message}"
        if level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)

        # 3. 세션 상태 로그 추가 (UI Status Box용)
        # 로딩 중이거나 짧은 상태 메시지인 경우 기존 로그를 대체할지 여부 결정 로직 추가 가능
        SessionManager.add_status_log(message)

        # 4. Streamlit Toast 알림 (옵션)
        if show_toast:
            try:
                st.toast(message, icon=icon)
            except Exception as e:
                logger.debug(f"Toast 표시 실패 (비 UI 스레드 가능성): {e}")

        # 5. UI 강제 동기화 (선택적)
        # 상태 박스가 있는 컨테이너를 즉시 업데이트하려면 여기서 콜백을 호출할 수 있음
        # 하지만 성능을 위해 호출자가 명시적으로 처리하는 것을 권장

    @classmethod
    def info(cls, message: str, show_toast: bool = False, icon: str | None = None):
        """일반 정보 알림"""
        cls._notify(message, "info", show_toast, icon)

    @classmethod
    def success(cls, message: str, show_toast: bool = True, icon: str | None = None):
        """성공 알림 (기본적으로 Toast 표시)"""
        cls._notify(message, "success", show_toast, icon)

    @classmethod
    def warning(cls, message: str, show_toast: bool = True):
        """경고 알림"""
        cls._notify(message, "warning", show_toast)

    @classmethod
    def error(
        cls, message: str, details: str | None = None, show_toast: bool = True
    ):
        """에러 알림"""
        full_msg = f"{message}: {details}" if details else message
        cls._notify(full_msg, "error", show_toast)
        # 에러는 세션에도 별도 기록 가능
        SessionManager.set("last_error", full_msg)

    @classmethod
    def loading(cls, message: str, show_toast: bool = False):
        """로딩/작업 진행 중 알림"""
        cls._notify(message, "info", show_toast, icon=cls.ICONS["loading"])

    @classmethod
    def model_load(cls, model_name: str, device: str = "GPU"):
        """모델 로딩 전용 알림"""
        icon = cls.ICONS["brain"] if device == "GPU" else cls.ICONS["setting"]
        msg = f"모델 로드 시작 ({device}): {model_name}"
        cls._notify(msg, "info", show_toast=True, icon=icon)
