import logging
from typing import Any

import streamlit as st

from core.session import SessionManager
from core.session.context import ContextManager
from ui.widget_keys import INTERACTIVE_KEYS


# Keys that are directly bound to Streamlit widgets and should not be overwritten by the background sync
# to prevent UI flickering (e.g., cursor jumping or input resetting).
class SyncRegistry:
    """
    Registry for keys that should be skipped during session synchronization
    to prevent UI flickering or input resets.
    """

    _interactive_keys: set[str] = set(INTERACTIVE_KEYS)

    @classmethod
    def is_interactive(cls, key: str) -> bool:
        """Checks if a key is registered as interactive."""
        return key in cls._interactive_keys


logger = logging.getLogger(__name__)


class UIBridge:
    """
    Streamlit UI와 백그라운드 세션 저장소 간의 데이터 동기화를 담당하는 브릿지 클래스.
    @st.fragment를 사용하여 전체 페이지 리런 없이 주기적으로 세션 상태를 업데이트합니다.
    """

    @classmethod
    def sync_session(cls) -> None:
        """
        SessionManager를 통해 세션 상태를 st.session_state로 동기화합니다.

        세션 저장소(SSoT)에 미러링할 변경이 없으면 스냅샷/복원/동기화 작업을
        생략하여 매 rerun마다 발생하는 불필요한 동기화 비용을 제거합니다.
        변경 유무는 ``has_pending_ui_sync``(``_dirty_keys``) 신호로 판별하며,
        인터랙티브 키 보호(SyncRegistry)와 무관하게 동작합니다.
        """
        session_id = ContextManager.get_current_session_id()

        # 세션 ID가 없거나 기본값인 경우 동기화 스킵
        if not session_id or session_id == "default":
            return

        # 변경 사항이 없으면 동기화 불필요 (rerun/render 비용 절감)
        if not SessionManager.has_pending_ui_sync(session_id):
            return

        try:
            # 동기화 전 대화형 키의 현재 값을 저장 (UI flickering 방지)
            interactive_snapshots: dict[str, Any] = {
                key: st.session_state[key]
                for key in SyncRegistry._interactive_keys
                if key in st.session_state
            }

            # SessionManager를 통해 핵심 상태를 st.session_state에 동기화
            SessionManager.sync_to_streamlit(session_id)

            # 대화형 키가 동기화에 의해 덮어쓰여졌다면 원래 값으로 복원
            for key, value in interactive_snapshots.items():
                st.session_state[key] = value

        except (RuntimeError, KeyError, ValueError) as e:
            # 프래그먼트 내부의 오류가 전체 앱을 중단시키지 않도록 예외 처리
            logger.error(f"[UIBridge] 세션 동기화 중 오류 발생: {e}", exc_info=True)
