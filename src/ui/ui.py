"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface
from ui.components.sidebar import render_sidebar
from ui.components.viewer import render_pdf_viewer


def render_left_column():
    """왼쪽 컬럼 (채팅) 렌더링"""
    render_chat_interface()


def render_right_column():
    """오른쪽 컬럼 (PDF 뷰어) 렌더링"""
    render_pdf_viewer()


@st.cache_resource
def inject_custom_css():
    """앱 전반에 걸친 최소한의 커스텀 CSS 및 JS 주입 (캐시됨)."""
    st.markdown(
        """
    <style>
    /* 1. 전체 앱 컨테이너 고정 및 스크롤 방지 */
    [data-testid="stAppViewContainer"] {
        height: 100vh !important;
        overflow: hidden !important;
    }

    /* 2. 메인 영역 패딩 최적화 */
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        height: 100vh !important;
    }

    /* 3. 사이드바 상단 정렬 */
    [data-testid="stSidebarContent"] {
        padding-top: 2rem !important;
    }

    /* 4. 고정 높이 컨테이너 내부 스크롤바 스타일링 */
    [data-testid="stVerticalBlockBorderWrapper"] > div:nth-child(1) {
        height: 100% !important;
        overflow-y: auto !important;
        scrollbar-width: thin;
    }

    /* 5. 불필요한 JS 실행 요소 숨김 */
    div.element-container:has(iframe[title="streamlit_javascript.st_javascript"]) {
        display: none !important;
    }

    /* 6. 사고 과정(Thought) 컨테이너 디자인 */
    .thought-container {
        font-style: italic;
        color: #555;
        background-color: #f8f9fa;
        border-left: 4px solid #dee2e6;
        padding: 12px;
        margin: 10px 0;
        border-radius: 4px;
        font-size: 0.95rem;
    }

    /* 7. 반응형 헬퍼: 화면 높이에 따른 자동 조절 */
    @media (min-height: 800px) {
        .fixed-height-container { height: 70vh !important; }
    }
    @media (max-height: 799px) {
        .fixed-height-container { height: 60vh !important; }
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def update_window_height():
    """[최적화] 창 높이 추적 시 불필요한 반복 호출을 방지합니다."""
    from streamlit_javascript import st_javascript

    if (
        "last_valid_height" in st.session_state
        and st.session_state.last_valid_height > 0
    ):
        return

    win_h = st_javascript("window.innerHeight", key="height_tracker")
    if win_h and win_h > 100:
        st.session_state.last_valid_height = int(win_h)
