# src/ui/ui.py
"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(하이브리드 레이아웃 전략: Flexbox 기반 가변 높이 및 반응형 칩 레이아웃 적용)
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css(is_expanded: bool = False):
    st.markdown(
        f"""
    <style>
    /* 1. Viewport Locking */
    .stApp {{
        height: 100vh !important;
        overflow: hidden !important;
    }}

    .block-container {{
        padding-top: 3rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
        height: 100vh !important;
        display: flex;
        flex-direction: column;
    }}

    /* 2. Flex Chain - Intermediate containers */
    /* Streamlit structure: .block-container > div > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"] */
    .block-container > div {{
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        min-height: 0;
    }}

    [data-testid="stVerticalBlock"] {{
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        min-height: 0;
    }}

    [data-testid="stHorizontalBlock"] {{
        display: flex;
        flex-grow: 1;
        min-height: 0;
    }}

    /* 3. Independent Columns */
    [data-testid="stColumn"] {{
        height: 100% !important; 
        overflow-y: auto !important;
        overflow-x: hidden !important;
        padding-right: 5px;
        scroll-behavior: smooth;
        display: flex;
        flex-direction: column;
    }}

    /* 스크롤바 스타일링 */
    [data-testid="stColumn"]::-webkit-scrollbar {{
        width: 6px;
    }}
    [data-testid="stColumn"]::-webkit-scrollbar-track {{
        background: transparent;
    }}
    [data-testid="stColumn"]::-webkit-scrollbar-thumb {{
        background-color: color-mix(in srgb, var(--text-color) 20%, transparent);
        border-radius: 10px;
    }}
    [data-testid="stColumn"]::-webkit-scrollbar-thumb:hover {{
        background-color: color-mix(in srgb, var(--text-color) 40%, transparent);
    }}

    /* 4. Scoped Chat Input */
    [data-testid="stChatInputContainer"] {{
        position: fixed !important;
        bottom: 0 !important;
        right: 0 !important;
        left: 50% !important; /* Start from middle */
        width: 50% !important; /* Take right half */
        padding-bottom: 1.5rem !important;
        padding-top: 0.5rem !important;
        background-color: var(--background-color) !important;
        z-index: 100;
        border-top: 1px solid color-mix(in srgb, var(--faded-text-color) 10%, transparent);
    }}

    /* Ensure column has enough bottom space so messages aren't hidden by the input */
    [data-testid="stColumn"]:last-child > div {{
        padding-bottom: 120px !important;
    }}

    [data-testid="stChatMessage"]:last-child {{
        margin-bottom: 2rem !important;
    }}

    /* 4. 사이드바 확장 버튼 가시성 확보 */
    [data-testid="stSidebarCollapseButton"] {{
        z-index: 100000 !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: color-mix(in srgb, var(--background-color), transparent 20%) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border-radius: 50% !important;
    }}

    /* 5. 상단 헤더 Glassmorphism */
    header[data-testid="stHeader"] {{
        background: color-mix(in srgb, var(--background-color), transparent 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--faded-text-color), transparent 80%) !important;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05) !important;
        z-index: 99999 !important;
        display: flex !important;
        visibility: visible !important;
    }}

    .stAppDeployButton {{
        display: none !important;
    }}

    /* 6. 사고 과정(Thought Process) UI - Layout Shift 방지 및 여백 압축 */
    details.thought-expander {{
        background-color: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--faded-text-color) 30%, transparent);
        border-radius: 8px;
        /* [수정] 아래쪽 마진을 16px에서 6px로 대폭 줄여 텍스트와의 간격 압축 */
        margin: 0px 0 6px 0;
        padding: 8px 12px;
        transition: all 0.2s ease-in-out;
    }}
    details.thought-expander summary {{
        cursor: pointer;
        font-size: 0.85em;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.7;
        outline: none;
        list-style: none;
        display: flex;
        align-items: center;
        gap: 8px;
        user-select: none;
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
        border-left: 3px solid color-mix(in srgb, var(--primary-color) 70%, transparent);
        padding: 10px 15px;
        margin-top: 8px; /* [수정] 내부 마진 축소 */
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

    /* 7. 인용 가독성 및 툴팁 */
    .citation-highlight {{
        background-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
        border-bottom: 2px dashed var(--primary-color);
        padding: 0 4px;
        border-radius: 3px;
        color: var(--primary-color);
        font-weight: 600;
        cursor: help;
        transition: background-color 0.2s, transform 0.1s;
        display: inline-block;
        position: relative;
    }}
    .citation-highlight:active::after {{
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: var(--text-color);
        color: var(--background-color);
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85em;
        font-weight: normal;
        white-space: pre-wrap;
        width: max-content;
        max-width: 250px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        pointer-events: none;
    }}

    /* 8. 스트리밍 애니메이션 (빌드 상태 표시에도 공용 사용) */
    .streaming-pulse {{
        animation: pulse 1.5s infinite ease-in-out;
        min-height: 24px;
        font-size: 0.9em;
        color: var(--primary-color);
        font-weight: 500;
        /* [수정] 마진 축소 */
        margin-bottom: 4px;
    }}
    @keyframes pulse {{
        0% {{ opacity: 0.4; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.4; }}
    }}

    /* 9. 성능 지표 HUD - 컴팩트 디자인 */
    .perf-details {{
        margin-top: 2px !important;
        padding-top: 0px !important;
    }}
    .perf-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 8px;
        margin-top: 8px;
    }}
    .perf-card {{
        background-color: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--faded-text-color) 15%, transparent);
        border-radius: 6px;
        padding: 6px 8px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}
    .perf-card-title {{
        font-size: 10px;
        color: var(--faded-text-color);
        margin-bottom: 2px;
        font-weight: 500;
    }}
    .perf-card-value {{
        font-size: 13px;
        font-weight: 700;
        color: var(--text-color);
    }}
    .perf-card-desc {{
        font-size: 9px;
        color: var(--faded-text-color);
        margin-top: 1px;
    }}
    
    /* 10. 참조 페이지 버튼 반응형 랩핑 시스템 */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: wrap !important;
        gap: 8px !important;
    }}
    div[data-testid="column"] {{
        min-width: 60px !important;
        flex: 0 1 auto !important;
        margin: 0 !important;
    }}
    div[data-testid="column"] button {{
        border-radius: 20px !important;
        padding: 4px 12px !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid color-mix(in srgb, var(--faded-text-color) 20%, transparent) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }}
    div[data-testid="column"] button:hover {{
        border-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
        background-color: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
    }}

    /* 11. 채팅 메시지 내부 수직 유격 압축 */
    [data-testid="stChatMessage"] div[data-testid="stVerticalBlock"] {{
        gap: 0px !important; /* [수정] 내부 블록 간격 완벽 제거 */
    }}

    /* 12. Mobile Responsiveness */
    @media (max-width: 768px) {{
        .stApp, .block-container {{
            height: auto !important;
            overflow: visible !important;
        }}
        [data-testid="stColumn"] {{
            height: auto !important;
            overflow: visible !important;
        }}
        [data-testid="stChatInputContainer"] {{
            left: 0 !important;
            width: 100% !important;
        }}
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
        setTimeout(collapseSidebar, 500);
    </script>
    """
    st.components.v1.html(js, height=0, width=0)