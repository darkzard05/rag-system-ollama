import logging

from core.session import SessionManager

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

        세션 저장소(SSoT)에 미러링할 변경이 없으면 동기화 작업을
        생략하여 매 rerun마다 발생하는 불필요한 동기화 비용을 제거합니다.
        변경 유무는 ``has_pending_ui_sync``(``_dirty_keys``) 신호로 판별하며,
        위젯 키는 절대 저장소에 의해 더티 처리되지 않습니다 (검증됨: 어떤
        ``SessionManager.set`` 호출도 ``INTERACTIVE_KEYS`` 맴버를 사용하지 않음),
        따라서 스냅샷/복원이 필요 없습니다 — 동기화는 저장소 키만 미러링하고
        위젯 키는 절대 덮어쓰지 않습니다 (Streamlit 1.54는 스크립트 측 위젯 키
        대입을 금지).
        """
        session_id = SessionManager.get_session_id()

        # 세션 ID가 없거나 기본값인 경우 동기화 스킵
        if not session_id or session_id == "default":
            return

        # 변경 사항이 없으면 동기화 불필요 (rerun/render 비용 절감)
        if not SessionManager.has_pending_ui_sync(session_id):
            return

        try:
            # SessionManager를 통해 핵심 상태를 st.session_state에 동기화
            SessionManager.sync_to_streamlit(session_id)

        except (RuntimeError, KeyError, ValueError) as e:
            # 프래그먼트 내부의 오류가 전체 앱을 중단시키지 않도록 예외 처리
            logger.error(f"[UIBridge] 세션 동기화 중 오류 발생: {e}", exc_info=True)
