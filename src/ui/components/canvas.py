import streamlit as st
from src.core.session import SessionManager


def render_knowledge_canvas():
    """
    AI가 생성한 구조화된 결과물(Artifacts)을 렌더링하는 지식 캔버스 컴포넌트입니다.
    """
    # 세션에서 현재 활성화된 아티팩트 정보를 가져옵니다.
    artifact = SessionManager.get("active_artifact")

    st.markdown("### 🎨 Knowledge Canvas")

    if not artifact:
        st.info(
            "아직 생성된 결과물이 없습니다. AI에게 요약, 표 작성, 또는 구조 분석을 요청해 보세요!"
        )
        return

    # 아티팩트 타입에 따른 렌더링 분기
    # artifact shape: {"type": "summary" | "table" | "graph", "content": "...", "title": "..."}
    a_type = artifact.get("type", "summary")
    a_content = artifact.get("content", "")
    a_title = artifact.get("title", "분석 결과")

    with st.container(border=True):
        st.markdown(f"#### {a_title}")

        if a_type == "summary":
            st.markdown(a_content)
        elif a_type == "table":
            try:
                # 마크다운 표 형태인 경우 streamlit의 dataframe이나 markdown으로 처리
                st.markdown(a_content)
            except Exception:
                st.error("표를 렌더링하는 중 오류가 발생했습니다.")
        elif a_type == "graph":
            st.warning(
                "그래프 뷰어는 현재 준비 중입니다. 텍스트 기반 구조로 표시합니다."
            )
            st.markdown(a_content)
        else:
            st.markdown(a_content)

        # 하단 액션 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 캔버스 비우기", use_container_width=True):
                SessionManager.set("active_artifact", None)
                st.rerun()
        with col2:
            if st.button("📋 복사하기", use_container_width=True):
                # 실제 클립보드 복사는 JS가 필요하므로 여기서는 알림만 표시
                st.toast("내용이 클립보드에 복사되었습니다. (시뮬레이션)")
