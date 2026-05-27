"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(정공법 해결: JS 제거, 상단 여백 확보, Native 버튼 노출)
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css(is_expanded: bool = False):
    """
    CSS-Only Layout (No-JS):
    상단 여백을 확보하여 Native 헤더와 사이드바 버튼을 물리적으로 노출시킵니다.
    """
    padding_left = "15px" if is_expanded else "60px"

    st.markdown(
        f"""
    <style>
    /* 1. 전역 레이아웃 및 뷰포트 고정 */
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
        height: 100dvh !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* 2. 메인 컨테이너 최적화 (상단 여백 50px 확보가 핵심) */
    [data-testid="stMainBlockContainer"] {{
        padding-top: 50px !important; /* 헤더 영역 확보 */
        padding-left: {padding_left} !important;
        padding-right: 15px !important;
        padding-bottom: 0px !important;
        max-width: 100% !important;
        height: 100dvh !important;
        display: flex;
        flex-direction: column;
        box-sizing: border-box;
    }}

    /* 3. [접근성] 레이블 시각적 숨김 (브라우저 인식 유지) */
    [data-testid="stWidgetLabel"] {{
        clip: rect(0 0 0 0);
        clip-path: inset(50%);
        height: 1px;
        overflow: hidden;
        position: absolute;
        white-space: nowrap;
        width: 1px;
    }}

    /* 4. Native 헤더 및 버튼 노출 및 스타일링 */
    header {{
        display: flex !important;
        visibility: visible !important;
        height: 50px !important;
        background-color: transparent !important;
        z-index: 999999 !important;
        pointer-events: auto !important;
    }}

    /* 사이드바 확장 버튼 스타일링 */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[aria-label="Open sidebar"] {{
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 1000000 !important;
        background-color: #007bff !important;
        border-radius: 8px !important;
        width: 40px !important;
        height: 32px !important;
        justify-content: center !important;
        align-items: center !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
        border: 2px solid white !important;
        opacity: 1 !important;
    }}

    /* 버튼 아이콘 색상 */
    [data-testid="stSidebarCollapsedControl"] svg,
    button[aria-label="Open sidebar"] svg {{
        fill: white !important;
        color: white !important;
    }}

    /* 사이드바 내부 닫기 버튼은 숨김 */
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}

    /* 불필요한 요소 제거 */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* 5. 기타 스타일링 (스크롤바) */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(136, 136, 136, 0.3); border-radius: 10px; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    """메인 영역의 채팅 인터페이스를 렌더링합니다."""
    return render_chat_interface()
