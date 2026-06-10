"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(스타일 최적화, 이중 스크롤 방지 및 접근성 준수 버전)
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        f"""
    <style>
    /* is_expanded: {is_expanded} */
    /* 전체 화면 스크롤 차단 및 높이 고정 */
    .stApp {{
        height: 100dvh;
        overflow: hidden;
    }}
    /* 메인 컨테이너 패딩 최적화: 헤더 공간 확보 및 하단 여백 제거 */
    .block-container {{
        padding-top: 3.5rem !important;
        padding-bottom: 0px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        height: 100%;
        display: flex;
        flex-direction: column;
    }}

    /* Streamlit 고정 높이 컨테이너를 뷰포트 기반 반응형으로 변환 */
    /* .block-container 내부에 위치한 스크롤 컨테이너만 타겟팅하여 사이드바 등 타 요소 보호 */
    .block-container [data-testid="stVerticalBlockBorderWrapper"] {{
        height: calc(100dvh - 11rem) !important;
        min-height: 200px !important;
        border: none !important;
    }}

    /* 채팅창 내부 메시지 영역 스크롤바 디자인 */
    .block-container [data-testid="stVerticalBlockBorderWrapper"] > div {{
        overflow-y: auto !important;
    }}

    /* PDF 뷰어 전용 보정 (첫 번째 컬럼에 위치하며 상단 컨트롤러 존재 대응) */
    .block-container [data-testid="stColumn"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"] {{
        height: calc(100dvh - 13.5rem) !important;
    }}

    /* 1. 사이드바 확장 버튼 가시성 강제 확보 (Invisible 이슈 해결) */
    [data-testid="stSidebarCollapseButton"] {{
        z-index: 100000 !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: color-mix(in srgb, var(--background-color), transparent 20%) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border-radius: 50% !important;
    }}

    /* 2. 상단 헤더 복구 및 Glassmorphism 적용 (Transparent 이슈 해결) */
    header[data-testid="stHeader"] {{
        background: color-mix(in srgb, var(--background-color), transparent 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--faded-text-color), transparent 80%) !important;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05) !important;
        z-index: 99999 !important;
        display: flex !important; /* display: none 제거 효과 */
        visibility: visible !important;
    }}

    /* 3. 불필요한 배포 버튼 등은 숨김 유지 (선택적) */
    .stAppDeployButton {{
        display: none !important;
    }}

    /* 4. 레이아웃 보정: 헤더와 본문 겹침 방지 - 제거 (상단에서 통합 관리) */

    /* 5. 사고 과정(Thought Process) 컴포넌트 모던 UI */
    details.thought-expander {{
        background-color: var(--secondary-background-color);
        border: 1px solid var(--faded-text-color);
        border-radius: 8px;
        margin: 8px 0 16px 0;
        padding: 8px 12px;
        transition: all 0.2s ease-in-out;
    }}
    details.thought-expander:hover {{
        border-color: var(--primary-color);
    }}
    details.thought-expander summary {{
        cursor: pointer;
        font-size: 0.9em;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.8;
        outline: none;
        list-style: none;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    details.thought-expander summary::-webkit-details-marker {{
        display: none;
    }}
    details.thought-expander summary::before {{
        content: "▶";
        font-size: 0.8em;
        transition: transform 0.2s;
    }}
    details[open].thought-expander summary::before {{
        transform: rotate(90deg);
    }}
    .thought-container {{
        border-left: 3px solid var(--primary-color);
        padding: 10px 15px;
        margin-top: 12px;
        font-size: 0.85em;
        color: var(--text-color);
        opacity: 0.85;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
        background-color: color-mix(in srgb, var(--background-color) 50%, transparent);
        border-radius: 0 4px 4px 0;
    }}

    /* 6. 인용 가독성 효과 */
    .citation-highlight {{
        background-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
        border-bottom: 2px dashed var(--primary-color);
        padding: 0 4px;
        border-radius: 3px;
        color: var(--primary-color);
        font-weight: 600;
        cursor: help;
        transition: background-color 0.2s;
    }}
    .citation-highlight:hover {{
        background-color: color-mix(in srgb, var(--primary-color) 30%, transparent);
    }}

    /* 7. 스트리밍 대기 애니메이션 (Pulse) */
    .streaming-pulse {{
        animation: pulse 1.5s infinite ease-in-out;
        min-height: 24px;
    }}
    @keyframes pulse {{
        0% {{ opacity: 0.4; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.4; }}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )


def render_left_column():
    return render_chat_interface()


def inject_sidebar_closer():
    """사이드바를 프로그램적으로 닫기 위한 JS 인젝션"""
    js = """
    <script>
        const collapseSidebar = () => {
            const sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
            const collapseButton = window.parent.document.querySelector('[data-testid="stSidebarCollapseButton"]');
            if (sidebar && sidebar.getAttribute('aria-expanded') === 'true' && collapseButton) {
                collapseButton.click();
            }
        };
        // 실행 지연을 주어 Streamlit 렌더링 후 동작하도록 함
        setTimeout(collapseSidebar, 500);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)
