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
    st.markdown(
        """
    <style>
    /* 1. 전역 뷰포트 고정 (dvh 사용으로 모바일 대응) */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
        height: 100dvh !important;
        margin: 0 !important;
        padding: 0 !important;
        scrollbar-gutter: stable;
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

    /* 3. 상단 안전 영역 확보 및 메인 컨테이너 최적화 */
    .stApp {
        padding-top: 0 !important; /* 상단바 제거에 따른 패딩 제거 */
        height: 100dvh !important;
        overflow: hidden !important;
    }

    [data-testid="stMainBlockContainer"] {
        padding: 0px 15px 0px __PADDING_LEFT__ !important;
        max-width: 100% !important;
        height: 100dvh !important;
        display: flex;
        flex-direction: column;
        gap: 0 !important;
        box-sizing: border-box;
        overflow: hidden !important;
    }



    /* 4. 컬럼 내부 Flexbox 레이아웃 최적화 */
    [data-testid="stColumn"] {
        height: 100% !important;
        min-height: 0 !important;
        display: flex;
        flex-direction: column;
        overflow: hidden !important;
        gap: 0 !important; /* 컬럼 내 요소 간 간격 제거 */
    }

    [data-testid="stColumn"] > div {
        flex-grow: 1 !important;
        display: flex;
        flex-direction: column;
        height: 100% !important;
        min-height: 0 !important;
        gap: 0 !important;
    }

    /* 컨테이너의 고정 높이를 무효화하고 남은 공간을 채우도록(flex-grow) 설정 */
    [data-testid="stVerticalBlockBorderWrapper"]:has(div[style*="overflow-y: auto"]) {
        flex-grow: 1 !important;
        flex-shrink: 1 !important;
        height: auto !important;
        min-height: 0 !important;
        margin-bottom: 0 !important;
    }

    /* 실제 스크롤이 발생하는 내부 블록 */
    [data-testid="stVerticalBlockBorderWrapper"]:has(div[style*="overflow-y: auto"]) > div > [data-testid="stVerticalBlock"] {
        height: 100% !important;
        overflow-y: auto !important;
        gap: 0 !important;
    }

    /* 채팅 입력창 고정 및 가림 방지 */
    [data-testid="stChatInput"] {
        padding-bottom: 0px !important; /* 하단 여백 제거 */
        background-color: transparent !important;
    }



    /* 5. 컴포넌트 디자인 보호 및 최적화 */
    .thought-container {
        font-size: 0.85rem;
        background-color: rgba(255, 255, 255, 0.03);
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin-bottom: 10px;
    }

    /* 헤더: 높이 0 + overflow visible → 공간 제거하되 내부 버튼은 밖으로 넘쳐 보임 */
    header {
        height: 0 !important;
        overflow: visible !important;
        padding: 0 !important;
        margin: 0 !important;
        background: transparent !important;
    }

    /* 사이드바 확장 버튼 (접혔을 때 나타남): 화면 좌상단 고정 배치 */
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
        visibility: visible !important;
        position: fixed !important;
        top: 15px !important;
        left: 15px !important;
        z-index: 1000001 !important;
        background-color: #007bff !important;
        border-radius: 10px !important;
        width: 44px !important;
        height: 36px !important;
        justify-content: center !important;
        align-items: center !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
        cursor: pointer !important;
    }

    /* 사이드바 내부의 축소 버튼(X) 숨김 (1.54.0+ 에서는 확장 버튼과 ID 공유하므로 범위 한정) */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
    }


    /* 6. 토스트 메시지 위치 조정 (사이드바 하단) */
    div[data-testid="stToastContainer"] {
        bottom: 30px !important;
        left: 30px !important;
        top: auto !important;
        right: auto !important;
        width: 300px !important;
        z-index: 1000000 !important;
    }

    div[data-testid="stToast"] {
        background-color: rgba(26, 26, 26, 0.95) !important;
        border: 1px solid rgba(0, 123, 255, 0.3) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    </style>

    <script>
    (function() {
        var BTN_ID = '__sidebar_expand_btn__';

        function fixSidebarBtn() {
            // Streamlit 원본 확장 버튼 탐색 (1.54.0 대응)
            var original = document.querySelector('[data-testid="stSidebarCollapsedControl"]')
                        || document.querySelector('[data-testid="collapsedControl"]');

            // 1.54.0에서 동일한 data-testid를 사용하는 버튼 처리
            if (!original) {
                var buttons = document.querySelectorAll('[data-testid="stSidebarCollapseButton"]');
                var sidebar = document.querySelector('[data-testid="stSidebar"]');
                for (var i = 0; i < buttons.length; i++) {
                    // 사이드바 내부에 있지 않은 버튼이 확장 버튼임
                    if (sidebar && !sidebar.contains(buttons[i])) {
                        original = buttons[i];
                        break;
                    }
                }
            }

            if (original) {
                // body로 이동시켜 header 클리핑에서 완전히 벗어남
                if (original.parentElement !== document.body) {
                    document.body.appendChild(original);
                }
                Object.assign(original.style, {
                    display: 'flex',
                    visibility: 'visible',
                    position: 'fixed',
                    top: '15px',
                    left: '15px',
                    zIndex: '1000001',
                    backgroundColor: '#007bff',
                    borderRadius: '10px',
                    width: '44px',
                    height: '36px',
                    justifyContent: 'center',
                    alignItems: 'center',
                    boxShadow: '0 4px 10px rgba(0,0,0,0.4)',
                    cursor: 'pointer',
                    border: '2px solid rgba(255, 255, 255, 0.2)'
                });
            }
        }

        // DOM 변경 감시 (Streamlit 리렌더 대응)
        var observer = new MutationObserver(fixSidebarBtn);
        observer.observe(document.documentElement, { childList: true, subtree: true });

        // 초기 실행 (딜레이로 Streamlit 렌더 완료 대기)
        setTimeout(fixSidebarBtn, 500);
        setTimeout(fixSidebarBtn, 1500);
    })();
    </script>

    """.replace("__PADDING_LEFT__", padding_left),
        unsafe_allow_html=True,
    )


def render_left_column():
    """메인 영역의 채팅 인터페이스를 렌더링합니다."""
    return render_chat_interface()
