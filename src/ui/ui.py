"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
"""

from __future__ import annotations

import streamlit as st

from core.session import SessionManager
from ui.components.chat import render_chat_interface


@st.fragment(run_every="1s")
def render_global_status_bar():
    """
    최상단에 고정된 전역 상태 표시줄을 렌더링합니다.
    1초마다 세션 상태를 체크하여 실시간 업데이트를 제공합니다.
    """
    if SessionManager.get("rag_build_complete_flag"):
        SessionManager.set("rag_build_complete_flag", False)
        st.rerun()

    status_msg = SessionManager.get("global_status", "✅ 시스템 준비 완료")
    status_level = SessionManager.get("status_level", "success")
    progress = SessionManager.get("global_progress", 0)

    colors = {
        "success": "#28a745",
        "info": "#007bff",
        "warning": "#ffc107",
        "error": "#dc3545",
    }
    bg_color = colors.get(status_level, "#212529")
    text_color = "black" if status_level == "warning" else "white"

    progress_html = ""
    if status_level != "success" and 0 < progress < 100:
        progress_html = f"""
        <div class="status-progress-container">
            <div class="status-progress-bar" style="width: {progress}%;"></div>
        </div>
        """

    st.markdown(
        f"""
        <div class="global-status-bar" style="background-color: {bg_color}; color: {text_color};">
            <span class="status-dot">●</span> {status_msg}
        </div>
        {progress_html}
        """,
        unsafe_allow_html=True,
    )


def inject_custom_css(is_expanded: bool = False):
    """
    Pixel-Perfect App-Shell 레이아웃 CSS:
    dvh 도입, Flexbox 최적화 및 범용 컴포넌트 디자인 보호를 적용합니다.
    """
    st.markdown(
        """
    <style>
    /* 1. 전역 뷰포트 고정 (dvh 사용으로 모바일 대응) */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100dvh !important;
        margin: 0 !important;
        padding: 0 !important;
        scrollbar-gutter: stable; /* 스크롤바 생성 시 레이아웃 밀림 방지 */
    }

    /* 2. 표준 스크롤바 및 커스텀 디자인 */
    * {
        scrollbar-width: thin;
        scrollbar-color: rgba(136, 136, 136, 0.3) transparent;
    }

    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(136, 136, 136, 0.3); border-radius: 10px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(136, 136, 136, 0.8); }

    /* 3. 상단 상태 바 영역 확보 및 메인 컨테이너 최적화 */
    .stApp {
        padding-top: 28px !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding: 10px 20px !important;
        max-width: 100% !important;
        height: calc(100dvh - 35px) !important; /* 상단바 28px + 여백 고려 */
        display: flex;
        flex-direction: column;
        gap: 0 !important;
        box-sizing: border-box;
    }

    /* 4. 컬럼 내부 레이아웃 최적화 (단순화) */
    [data-testid="stColumn"] {
        height: 100% !important;
        overflow: hidden !important;
    }

    /* 컨테이너 및 래퍼가 고정 높이를 유지하되 범위를 벗어나지 않도록 조정 */
    [data-testid="stMainBlockContainer"] [data-testid="stVerticalBlockBorderWrapper"] {
        border: none !important;
    }

    /* 실제 스크롤이 발생하는 내부 블록 */
    [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
        overflow-y: auto !important;
    }

    /* 5. 독립 스크롤 영역 정밀 타겟팅 (범용 컴포넌트 보호) */

    /* 사고 과정(Thought) 및 상태창(Status), 익스팬더 테두리 복원 */
    [data-testid="stStatusWidget"], .thought-container, [data-testid="stExpander"], [data-testid="stHeader"] {
        flex-grow: 0 !important;
        height: auto !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    /* 6. UI 요소 최적화 및 가림 방지 */
    header { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    [data-testid="stDecoration"] { display: none !important; }

    .global-status-bar {
        width: 100%;
        height: 28px;
        display: flex;
        align-items: center;
        padding: 0 15px;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-radius: 6px;
    }

    .thought-container {
        font-size: 0.85rem;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 10px;
    }

    /* 7. 사이드바 로고 디자인 (Classic Split) */
    .sidebar-logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 15px 12px;
        background: rgba(0, 123, 255, 0.1);
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid rgba(0, 123, 255, 0.2);
    }
    .logo-icon-wrapper {
        background: #007bff;
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 123, 255, 0.4);
    }
    .logo-text-wrapper {
        display: flex;
        flex-direction: column;
        line-height: 1.2;
    }
    .logo-main-row {
        display: flex;
        align-items: baseline;
        gap: 4px;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .logo-rag {
        color: #ffffff;
    }
    .logo-system {
        color: #007bff;
    }
    .logo-ollama {
        font-size: 0.75rem;
        font-weight: 700;
        color: #888;
        letter-spacing: 2px;
        margin-top: 2px;
        padding-top: 2px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* 8. 시스템 상태 모니터 디자인 */
    .status-container {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px;
        margin-top: 20px;
    }
    .status-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 12px;
    }
    .status-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #888;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .live-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.7rem;
        color: #4caf50;
        font-weight: 700;
    }
    .live-dot {
        width: 6px;
        height: 6px;
        background: #4caf50;
        border-radius: 50%;
        box-shadow: 0 0 8px #4caf50;
        animation: pulse-live 2s infinite;
    }
    @keyframes pulse-live {
        0% { opacity: 1; }
        50% { opacity: 0.4; }
        100% { opacity: 1; }
    }
    .metric-row {
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    .metric-item {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .metric-label-row {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        color: #ccc;
    }
    .progress-track {
        background: rgba(255, 255, 255, 0.1);
        height: 4px;
        border-radius: 2px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: #007bff;
        border-radius: 2px;
        transition: width 0.5s ease-in-out;
    }

    /* 9. 고급 설정 섹션 정밀 스타일링 */
    .settings-label {
        font-size: 0.7rem;
        font-weight: 700;
        color: #777;
        margin-bottom: 8px;
        display: block;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        padding-bottom: 3px;
    }
    .settings-sublabel {
        font-size: 0.68rem;
        color: #666;
        margin-bottom: 4px;
        font-weight: 600;
        margin-top: 6px;
    }

    /* Expander 전체 스타일 강제 적용 */
    [data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        background-color: rgba(255, 255, 255, 0.01) !important;
        margin-top: 0px !important;
    }

    /* Expander 헤더(Summary) 커스텀 */
    [data-testid="stExpander"] summary {
        padding: 6px 12px !important;
        color: #bbb !important;
        min-height: 0 !important;
    }

    /* Expander 내부 컨텐츠 공백 최소화 */
    [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
        padding: 0px 5px 10px 5px !important;
        gap: 0px !important;
    }

    /* Selectbox 내부 스타일 강제 주입 */
    div[data-testid="stSelectbox"] div[role="button"] {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 6px !important;
        font-size: 0.8rem !important;
        padding: 0px 8px !important;
        min-height: 1.8rem !important;
    }

    /* 버튼 변형 */
    div.stButton > button {
        height: 1.8rem !important;
        padding: 0 10px !important;
        line-height: 1 !important;
    }

    /* 10. 인터랙티브 인용구 애니메이션 */
    .citation-highlight {
        transition: all 0.2s ease;
        padding: 1px 3px;
        border-radius: 4px;
    }
    .citation-highlight:hover {
        background-color: rgba(0, 123, 255, 0.1);
        box-shadow: 0 0 0 1px rgba(0, 123, 255, 0.3);
    }
    </style>

    <script>
    // 인용구 클릭 이벤트 리스너 (이벤트 위임 사용)
    document.addEventListener('click', function(e) {
        const citation = e.target.closest('.citation-highlight');
        if (citation) {
            const page = citation.getAttribute('data-page');
            if (page) {
                // Streamlit의 hidden input 또는 전역 상태 업데이트를 위한 트릭
                // 1. URL 해시 변경 (Streamlit rerun 유도 가능)
                // window.location.hash = 'page=' + page;

                // 2. Streamlit 위젯 조작 (더 확실한 방법)
                // 페이지 번호를 입력하는 input 위젯을 찾아 값을 변경하고 엔터 키 이벤트를 발생시킵니다.
                const navInputs = window.parent.document.querySelectorAll('input[aria-label*="Page"]');
                navInputs.forEach(input => {
                    input.value = page;
                    input.dispatchEvent(new Event('input', { bubbles: true }));
                    input.dispatchEvent(new KeyboardEvent('keydown', { bubbles: true, key: 'Enter', keyCode: 13 }));
                });
            }
        }
    });
    </script>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    """메인 영역의 채팅 인터페이스를 렌더링합니다."""
    return render_chat_interface()
