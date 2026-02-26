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
    사이드바 너비를 동적으로 제어하고 2열 레이아웃을 최적화합니다.
    """
    # 문서가 있으면 1040px (300 + 40(gap) + 700), 없으면 300px 고정
    sidebar_width = "1040px" if is_expanded else "300px"

    st.markdown(
        f"""
    <style>
    /* 1. 사이드바 전체 너비 강제 고정 및 애니메이션 */
    [data-testid="stSidebar"] {{
        width: {sidebar_width} !important;
        min-width: {sidebar_width} !important;
        max-width: {sidebar_width} !important;
        transition: width 0.3s ease-in-out;
    }}

    [data-testid="stSidebarContent"] {{
        width: {sidebar_width} !important;
        overflow-x: hidden !important;
    }}

    /* 2. 내부 2열 레이아웃 및 간격(Gap) 제어 */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {{
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        gap: {"40px" if is_expanded else "0px"} !important; /* 확장 시 40px 간격 적용 */
    }}

    /* 제1열: 설정 영역 (300px 절대 고정) */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) {{
        flex: 0 0 300px !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }}

    /* 제2열: 미리보기 영역 */
    [data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) {{
        display: {"block" if is_expanded else "none"} !important;
        flex: {"0 0 700px" if is_expanded else "0 0 0px"} !important;
        width: {"700px" if is_expanded else "0px"} !important;
        min-width: {"700px" if is_expanded else "0px"} !important;
        overflow-x: hidden !important;
    }}

    /* 3. 잔상 제거 */
    [data-testid="stSidebarContent"] {{
        background-color: var(--secondary-background-color) !important;
    }}

    /* 4. 메인 컨텐츠 영역 */
    [data-testid="stMainBlockContainer"] {{
        max-width: 100% !important;
        padding-top: 1.5rem !important;
        padding-bottom: 6rem !important; /* 하단 입력창 공간 확보 */
    }}

    /* 5. 기타 컴포넌트 스타일 및 헤더 정렬 */
    .sidebar-header {{
        margin-top: 0px !important;
        margin-bottom: 20px !important;
        padding-top: 0px !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        display: flex;
        align-items: center;
        height: 40px; /* 고정 높이로 수직 정렬 보장 */
    }}

    /* 사이드바 익스팬더(고급 설정 등)는 기본 스타일 존중 */
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * {{
        color: inherit !important;
        font-weight: normal !important;
    }}

    .thought-container {{
        font-style: italic;
        color: var(--text-color) !important;
        background-color: rgba(128, 128, 128, 0.1) !important;
        border-left: 4px solid #1e88e5 !important;
        padding: 15px !important;
        margin: 10px 0 !important;
        border-radius: 4px !important;
        line-height: 1.6 !important;
    }}
    .timeline-container {{
        border: 1px solid rgba(128, 128, 128, 0.3) !important;
        border-radius: 8px !important;
        padding: 12px !important;
        background: transparent !important;
        color: var(--text-color) !important;
    }}
    /* 타임라인 로그 글씨 색상 보정 */
    .timeline-container div {{
        color: var(--text-color) !important;
    }}

    /* 6. 인용구 하이라이트 및 툴팁 스타일 */
    .citation-tooltip {{
        color: #1e88e5 !important;
        font-weight: bold !important;
        cursor: help !important;
        border-bottom: 2px solid rgba(30, 136, 229, 0.3) !important;
        padding: 0 2px !important;
        transition: all 0.2s ease !important;
        position: relative;
        display: inline-block;
    }}

    .citation-tooltip:hover {{
        background-color: rgba(30, 136, 229, 0.1) !important;
        border-bottom: 2px solid #1e88e5 !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
