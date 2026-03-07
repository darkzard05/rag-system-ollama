"""
시스템 상태 박스(Status Box) UI 컴포넌트 - Native Streamlit Refactoring
"""

import streamlit as st

from core.session import SessionManager


def render_status_box(container):
    """시스템 상태 로그 박스를 Streamlit 네이티브 컴포넌트로 렌더링합니다."""
    if container is None:
        return

    try:
        status_logs = SessionManager.get("status_logs", [])
    except Exception:
        status_logs = []

    if not status_logs:
        container.info("시스템 준비 중...")
        return

    # [최적화] HTML 대신 네이티브 expander와 caption 사용 (테마 호환성 100%)
    with container.expander("📊 시스템 작업 이력", expanded=True):
        # 최신 로그 순으로 출력
        reversed_logs = status_logs[::-1]

        for i, log in enumerate(reversed_logs):
            # 텍스트 정제 (제어 문자 제거 등)
            clean_log = log.strip()
            if not clean_log:
                continue

            if i == 0:
                # 최신 로그는 강조 표시
                st.markdown(f"**🔵 {clean_log}**")
            else:
                # 이전 로그들은 연하게 표시
                st.caption(f"⚪ {clean_log}")

            # 너무 많은 로그가 쌓였을 경우 가독성을 위해 제한 (최근 15개)
            if i >= 14:
                st.caption("... (이전 로그 생략)")
                break
