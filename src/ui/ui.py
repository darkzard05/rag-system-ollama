"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css(is_expanded: bool = False):
    """
    Pixel-Perfect App-Shell 레이아웃 CSS:
    dvh 도입, Flexbox 최적화 및 범용 컴포넌트 디자인 보호를 적용합니다.
    """
    padding_left = "15px" if is_expanded else "60px"

    # 1. CSS 주입 (스타일 전용)
    st.markdown(
        f"""
    <style>
    /* 1. 전역 뷰포트 고정 */
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
        height: 100dvh !important;
        margin: 0 !important;
        padding: 0 !important;
        scrollbar-gutter: stable;
    }}

    /* 2. 표준 스크롤바 */
    * {{
        scrollbar-width: thin;
        scrollbar-color: rgba(136, 136, 136, 0.3) transparent;
    }}

    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(136, 136, 136, 0.3); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: rgba(136, 136, 136, 0.8); }}

    /* 3. 메인 컨테이너 최적화 (Padding 동적 적용) */
    .stApp {{
        padding-top: 0 !important;
        height: 100dvh !important;
        overflow: hidden !important;
    }}

    [data-testid="stMainBlockContainer"] {{
        padding: 0px 15px 0px {padding_left} !important;
        max-width: 100% !important;
        height: 100dvh !important;
        display: flex;
        flex-direction: column;
        gap: 0 !important;
        box-sizing: border-box;
        overflow: hidden !important;
    }}

    /* 4. 컬럼 내부 Flexbox 레이아웃 */
    [data-testid="stColumn"] {{
        height: 100% !important;
        min-height: 0 !important;
        display: flex;
        flex-direction: column;
        overflow: hidden !important;
        gap: 0 !important;
    }}

    [data-testid="stColumn"] > div {{
        flex-grow: 1 !important;
        display: flex;
        flex-direction: column;
        height: 100% !important;
        min-height: 0 !important;
        gap: 0 !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(div[style*="overflow-y: auto"]) {{
        flex-grow: 1 !important;
        flex-shrink: 1 !important;
        height: auto !important;
        min-height: 0 !important;
        margin-bottom: 0 !important;
    }}

    [data-testid="stVerticalBlockBorderWrapper"]:has(div[style*="overflow-y: auto"]) > div > [data-testid="stVerticalBlock"] {{
        height: 100% !important;
        overflow-y: auto !important;
        gap: 0 !important;
    }}

    /* 채팅 입력창 고정 */
    [data-testid="stChatInput"] {{
        padding-bottom: 0px !important;
        background-color: transparent !important;
    }}

    /* 5. 컴포넌트 디자인 보호 */
    .thought-container {{
        font-size: 0.85rem;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 10px;
    }}

    /* 헤더 노출 (버튼 가림 방지를 위해 0 대신 최소 높이 부여 시도) */
    header {{
        height: 40px !important;
        background: transparent !important;
        overflow: visible !important;
        z-index: 99999 !important;
    }}

    /* 사이드바 확장 버튼: CSS만으로도 보이게 설정 */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {{
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        top: 10px !important;
        left: 10px !important;
        z-index: 10000000 !important;
        background-color: #007bff !important;
        border-radius: 8px !important;
        width: 40px !important;
        height: 32px !important;
        justify-content: center !important;
        align-items: center !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.5) !important;
        border: 2px solid white !important;
        cursor: pointer !important;
    }}

    /* 아이콘 색상 강제 */
    [data-testid="stSidebarCollapsedControl"] svg,
    [data-testid="stSidebarCollapseButton"] svg {{
        fill: white !important;
        color: white !important;
    }}

    /* 사이드바 내부 닫기 버튼만 숨김 */
    section[data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] {{
        display: none !important;
    }}

    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    /* 6. 토스트 메시지 위치 */
    div[data-testid="stToastContainer"] {{
        bottom: 30px !important;
        left: 30px !important;
        top: auto !important;
        right: auto !important;
        width: 300px !important;
        z-index: 1000000 !important;
    }}

    div[data-testid="stToast"] {{
        background-color: rgba(26, 26, 26, 0.95) !important;
        border: 1px solid rgba(0, 123, 255, 0.3) !important;
        border-radius: 8px !important;
        color: white !important;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 2. JavaScript 주입 (별도 실행 및 단순화)
    st.markdown(
        """
    <script>
    (function() {
        console.log("Sidebar Fix Script Running...");
        
        function applyFix() {
            var btn = document.querySelector('[data-testid="stSidebarCollapsedControl"]')
                   || document.querySelector('[data-testid="stSidebarCollapseButton"]');
            
            // 사이드바 외부 버튼만 처리
            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (btn && (!sidebar || !sidebar.contains(btn))) {
                if (btn.parentElement !== document.body) {
                    document.body.appendChild(btn);
                    console.log("Button moved to body");
                }
                btn.style.display = 'flex';
                btn.style.visibility = 'visible';
                btn.style.zIndex = '10000000';
            }
        }

        // 초기 실행 및 관찰
        setTimeout(applyFix, 1000);
        new MutationObserver(applyFix).observe(document.documentElement, { childList: true, subtree: true });
    })();
    </script>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    """메인 영역의 채팅 인터페이스를 렌더링합니다."""
    return render_chat_interface()
