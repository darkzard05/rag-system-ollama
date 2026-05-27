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

    # 1. CSS 주입
    st.markdown(
        f"""
    <style>
    /* 1. 전역 뷰포트 고정 */
    html, body, [data-testid="stAppViewContainer"] {{
        overflow: hidden !important;
        height: 100dvh !important;
        margin: 0 !important;
        padding: 0 !important;
    }}

    /* 2. 표준 스크롤바 */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-thumb {{ background: rgba(136, 136, 136, 0.3); border-radius: 10px; }}

    /* 3. 메인 컨테이너 최적화 */
    [data-testid="stMainBlockContainer"] {{
        padding: 0px 15px 0px {padding_left} !important;
        max-width: 100% !important;
        height: 100dvh !important;
        display: flex;
        flex-direction: column;
        overflow: hidden !important;
    }}

    /* [접근성] 레이블 시각적 숨김 처리 (DOM에는 유지) */
    [data-testid="stWidgetLabel"] {{
        display: none !important;
    }}

    /* [중요] 기존 Streamlit 모든 사이드바 컨트롤 숨김 (커스텀 버튼 사용 예정) */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    header {{
        display: none !important;
    }}

    /* 4. 커스텀 확장 버튼 스타일 (JS로 주입됨) */
    #__custom_sidebar_trigger__ {{
        position: fixed;
        top: 20px;
        left: 20px;
        width: 44px;
        height: 44px;
        background-color: #007bff;
        color: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 2147483647; /* 최상단 보장 */
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border: 2px solid white;
        transition: transform 0.2s;
        font-size: 20px;
        font-weight: bold;
        user-select: none;
    }}
    #__custom_sidebar_trigger__:hover {{
        transform: scale(1.1);
        background-color: #0056b3;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # 2. JavaScript 주입 (강력한 커스텀 버튼 주입 로직)
    st.markdown(
        """
    <script>
    (function() {
        var TRIGGER_ID = '__custom_sidebar_trigger__';
        
        function injectTrigger() {
            if (document.getElementById(TRIGGER_ID)) return;

            var trigger = document.createElement('div');
            trigger.id = TRIGGER_ID;
            trigger.innerHTML = '&#9776;'; // 햄버거 메뉴 아이콘
            trigger.title = 'Open Settings';
            
            trigger.onclick = function(e) {
                e.preventDefault();
                e.stopPropagation();
                // Streamlit의 원본 버튼을 찾아 클릭 이벤트 전달
                var nativeBtn = document.querySelector('[data-testid="stSidebarCollapseButton"]')
                             || document.querySelector('[data-testid="stSidebarCollapsedControl"]')
                             || document.querySelector('[data-testid="collapsedControl"]');
                if (nativeBtn) {
                    nativeBtn.click();
                } else {
                    console.log("Native sidebar button not found");
                }
            };

            document.body.appendChild(trigger);
        }

        // 사이드바 상태에 따라 커스텀 버튼 표시/숨김
        function syncTriggerVisibility() {
            var trigger = document.getElementById(TRIGGER_ID);
            if (!trigger) return;

            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            var isCollapsed = !sidebar || sidebar.getAttribute('aria-expanded') === 'false';
            
            trigger.style.display = isCollapsed ? 'flex' : 'none';
        }

        // 초기 실행 및 지속 관찰
        setInterval(function() {
            injectTrigger();
            syncTriggerVisibility();
        }, 500);
    })();
    </script>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    """메인 영역의 채팅 인터페이스를 렌더링합니다."""
    return render_chat_interface()
