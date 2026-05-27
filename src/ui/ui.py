"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(근본적 해결: JS 의존성 제거, Native 버튼 복구, 접근성 강화)
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css(is_expanded: bool = False):
    """
    CSS-Only Layout Fix:
    JS를 배제하고 오직 CSS 선택자만으로 사이드바 버튼을 살려내고 접근성을 개선합니다.
    """
    padding_left = "15px" if is_expanded else "60px"

    st.markdown(
        f"""
    <style>
    /* 1. 전역 뷰포트 고정 및 배경 설정 */
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
        height: 100dvh !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* 2. 메인 컨테이너 최적화 (Padding) */
    [data-testid="stMainBlockContainer"] {{
        padding: 0px 15px 0px {padding_left} !important;
        max-width: 100% !important;
        height: 100dvh !important;
        display: flex;
        flex-direction: column;
        gap: 0 !important;
        box-sizing: border-box;
    }}

    /* 3. [접근성] 레이블 시각적 숨김 (DOM에는 남겨두어 브라우저 경고 해결) */
    /* .sr-only 스타일 적용 */
    [data-testid="stWidgetLabel"] {{
        clip: rect(0 0 0 0);
        clip-path: inset(50%);
        height: 1px;
        overflow: hidden;
        position: absolute;
        white-space: nowrap;
        width: 1px;
    }}

    /* 4. Native 헤더 및 버튼 복구 및 스타일링 */
    header {{
        display: flex !important;
        visibility: visible !important;
        height: 48px !important;
        background: transparent !important;
        z-index: 999999 !important;
    }}

    /* 사이드바 확장 버튼 (접혔을 때 나타남) 강제 노출 */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    button[aria-label="Open sidebar"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        z-index: 1000000 !important;
        background-color: #007bff !important;
        border-radius: 8px !important;
        width: 40px !important;
        height: 32px !important;
        justify-content: center !important;
        align-items: center !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
        border: 2px solid white !important;
    }}

    /* 버튼 내부 아이콘(SVG) 색상 강제 */
    [data-testid="stSidebarCollapsedControl"] svg,
    button[aria-label="Open sidebar"] svg {{
        fill: white !important;
        color: white !important;
    }}

    /* 사이드바 내부 닫기 버튼은 숨김 (깔끔한 UI 유지) */
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}

    /* 불필요한 장식 제거 */
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* 5. 기타 UI 보강 (스크롤바 등) */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(136, 136, 136, 0.3); border-radius: 10px; }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    """메인 영역의 채팅 인터페이스를 렌더링합니다."""
    return render_chat_interface()
