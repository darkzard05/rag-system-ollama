"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
"""

from __future__ import annotations

import streamlit as st

from core.session import SessionManager
from ui.components.sidebar import render_sidebar as _render_sidebar


def render_sidebar(
    file_uploader_callback,
    model_selector_callback,
    embedding_selector_callback,
    is_generating=False,
    current_file_name=None,
    available_models=None,
):
    """사이드바 렌더링 위임"""
    return _render_sidebar(
        file_uploader_callback=file_uploader_callback,
        model_selector_callback=model_selector_callback,
        embedding_selector_callback=embedding_selector_callback,
        is_generating=is_generating,
        current_file_name=current_file_name,
        available_models=available_models,
    )


@st.fragment(run_every="1s")
def render_global_status_bar():
    """
    최상단에 고정된 전역 상태 표시줄을 렌더링합니다.
    1초마다 세션 상태를 체크하여 실시간 업데이트를 제공합니다.
    """
    status_msg = SessionManager.get("global_status", "✅ 시스템 준비 완료")
    status_level = SessionManager.get("status_level", "success")

    # 상태별 배경색 정의
    colors = {
        "success": "#28a745",  # Green
        "info": "#007bff",  # Blue
        "warning": "#ffc107",  # Yellow
        "error": "#dc3545",  # Red
    }
    bg_color = colors.get(status_level, "#212529")
    text_color = "black" if status_level == "warning" else "white"

    # HTML 주입 (최상단 고정 레이어)
    st.markdown(
        f"""
        <div class="global-status-bar" style="background-color: {bg_color}; color: {text_color};">
            <span class="status-dot">●</span> {status_msg}
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_custom_css():
    """
    레이아웃 CSS: 상단 안전 영역 확보 및 App-Shell 레이아웃 최적화
    - 전역 상태 바 스타일 정의
    - 메인 컨테이너 패딩 복원 (28px)
    - 컬럼 높이 정밀 계산 (calc(100vh - 28px))
    """
    st.markdown(
        """
    <style>
    /* 1. 전역 상태 표시줄 스타일 (최상단 고정) */
    .global-status-bar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 28px;
        z-index: 999999;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 15px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        transition: background-color 0.3s ease;
    }

    /* 진행률 표시줄 컨테이너 */
    .status-progress-container {
        position: fixed;
        top: 28px;
        left: 0;
        width: 100%;
        height: 4px;
        background-color: rgba(0,0,0,0.1);
        z-index: 999999;
    }

    /* 진행률 표시줄 */
    .status-progress-bar {
        height: 100%;
        background-color: #28a745;
        transition: width 0.3s ease;
    }

    .status-dot {
        margin-right: 8px;
        font-size: 8px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* 2. Streamlit 기본 요소 숨김 및 여백 조정 */
    header { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }

    /* 전역 여백 제거 */
    html, body { margin: 0 !important; padding: 0 !important; }

    /* 메인 컨테이너: 상단 안전 영역(28px) 확보 */
    .stApp {
        padding-top: 28px !important;
    }

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

    /* 3. 사이드바 조정 (상단 바와 일치) */
    [data-testid="stSidebar"] {
        padding-top: 0 !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.5rem;
    }

    /* 4. 2열 App-Shell 레이아웃: 높이 정밀 계산 */
    [data-testid="stHorizontalBlock"] {
        display: flex;
        height: calc(100vh - 28px) !important;
    }

    [data-testid="column"] {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 28px) !important;
        overflow: hidden;
    }

    /* 컨테이너 내부 스크롤 허용 */
    [data-testid="column"] div[data-testid="stVerticalBlockBorderWrapper"]:has(div[style*="overflow-y: auto"]) {
        flex: 1;
        height: 0;
        min-height: 0;
    }

    /* 하단 고정 영역 (Sticky Footer) */
    [data-testid="column"] > div > div:last-child {
        position: sticky !important;
        bottom: 0 !important;
        z-index: 99 !important;
        background-color: white !important;
        padding: 12px 8px !important;
        border-top: 1px solid #e0e0e0 !important;
        width: 100% !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
