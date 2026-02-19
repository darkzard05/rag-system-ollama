"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
"""

from __future__ import annotations

import streamlit as st

from common.utils import safe_cache_resource
from ui.components.chat import render_chat_interface
from ui.components.sidebar import render_sidebar as _render_sidebar
from ui.components.viewer import render_pdf_viewer


def render_left_column():
    """왼쪽 컬럼 (채팅) 렌더링"""
    render_chat_interface()


def render_right_column():
    """오른쪽 컬럼 (PDF 뷰어) 렌더링"""
    render_pdf_viewer()


def render_sidebar(**kwargs):
    """사이드바 렌더링 위임"""
    return _render_sidebar(**kwargs)


@safe_cache_resource
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

    /* 2. 메인 영역 패딩 최적화 (사이드바와 균형 조정) */
    [data-testid="stMainBlockContainer"] {
        padding-top: 5rem !important;
        padding-bottom: 1rem !important;
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

    /* 6. 헤더 크기 통일 (사이드바 & 메인 영역) */
    [data-testid="stSidebar"] h2,
    [data-testid="stMainBlockContainer"] h2 {
        font-size: 1.25rem !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }

    /* 7. 사고 과정(Thought) 컨테이너 디자인 */
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
