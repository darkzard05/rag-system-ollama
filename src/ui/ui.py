"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface
from ui.components.sidebar import render_sidebar as _render_sidebar


def render_left_column():
    """메인 채팅 영역 렌더링"""
    render_chat_interface()


def render_sidebar(**kwargs):
    """사이드바 렌더링 위임"""
    return _render_sidebar(**kwargs)


def inject_custom_css(is_expanded: bool = False):
    """
    레이아웃 CSS: 두 column 높이 동기화 및 공간 최대화
    - 헤더 & toolbar 숨김
    - 상단/좌측/우측 여백 제거
    - Column 높이 고정하되, 내부 Streamlit 위젯 구조는 보존
    """
    st.markdown(
        """
    <style>
    /* 헤더 & 배포 메뉴 바 숨김 */
    header {
        display: none;
    }

    [data-testid="stToolbar"] {
        display: none;
    }

    /* 전역 여백 제거 */
    html, body {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* 메인 컨테이너 여백 제거 */
    .appview-container {
        margin: 0 !important;
        padding: 0 !important;
    }

    .stMainBlockContainer {
        padding: 0 !important;
        margin: 0 !important;
    }

    .main {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
        margin: 0 !important;
    }

    /* Sidebar 패딩 유지 */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    /* 2열 컨테이너: 독립적 스크롤 레이아웃 */
    [data-testid="stHorizontalBlock"] {
        display: flex;
        height: 100vh;
    }

    /* 개별 column: 전체 높이 차지 및 내부 플렉스 설정 */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        height: 100vh;
        overflow: hidden;
    }

    /* st.container 내부의 stVerticalBlockBorderWrapper: 가변 높이 및 스크롤 허용 */
    [data-testid="column"] [data-testid="stVerticalBlockBorderWrapper"] {
        flex: 1;
        height: 0;
        min-height: 0;
    }

    /* 하단 고정 영역 */
    .fixed-bottom-area {
        flex-shrink: 0;
        padding: 8px;
        background: white;
        border-top: 1px solid #eee;
    }

    .pdf-control-bar {
        flex-shrink: 0;
        height: 60px;
        padding: 8px 12px;
        border-bottom: 1px solid #e0e0e0;
        display: flex;
        align-items: center;
        margin: 0 !important;
    }

    .chat-input-area {
        flex-shrink: 0;
        height: 70px;
        padding: 8px 4px !important;
        border-top: 1px solid #e0e0e0;
        display: flex;
        align-items: flex-end;
        margin: 0 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
