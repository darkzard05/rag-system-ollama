"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(원상복구 버전: 네이티브 기능 복원 및 최소 스타일 적용)
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        """
    <style>
    /* 1. 표준 레이아웃 유지 */
    html, body, [data-testid="stAppViewContainer"] {
        height: 100dvh !important;
    }

    /* 2. 접근성: 레이블 시각적 숨김 */
    [data-testid="stWidgetLabel"] {
        clip: rect(0 0 0 0);
        clip-path: inset(50%);
        height: 1px;
        overflow: hidden;
        position: absolute;
        white-space: nowrap;
        width: 1px;
    }

    /* 3. 네이티브 헤더 및 버튼 복구 */
    header {
        visibility: visible !important;
        background: transparent !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        background-color: #007bff !important;
        color: white !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    button[data-testid="stSidebarCollapseButton"] svg {
        fill: white !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    return render_chat_interface()
