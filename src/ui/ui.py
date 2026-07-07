# src/ui/ui.py
"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.
(하이브리드 레이아웃 전략: Flexbox 기반 가변 높이 및 반응형 칩 레이아웃 적용)
"""

from __future__ import annotations

import streamlit as st

from ui.components.chat import render_chat_interface


def inject_custom_css():
    st.markdown(
        """
<style>
/* ════════════════════════════════════════════
   0. GLOBAL VIEWPORT LOCK
   ════════════════════════════════════════════ */
.stApp, [data-testid="stAppViewContainer"] {
    --header-h: 60px;
    height: 100vh !important;
    height: 100dvh !important;
    overflow: hidden !important;
}

/* ════════════════════════════════════════════
   1-8. FLEX CHAIN (DOM 검증 기반 7-step)
   ════════════════════════════════════════════ */

/* 1. Main container: 헤더 제외 높이 + flex 컬럼 */
[data-testid="stMainBlockContainer"] {
    height: calc(100vh - var(--header-h, 60px)) !important;
    height: calc(100dvh - var(--header-h, 60px)) !important;
    display: flex !important;
    flex-direction: column !important;
    overflow: hidden !important;
}

/* 2. stMainBlockContainer의 직계 stVerticalBlock이 flex 공간 채움 */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    min-height: 0 !important;
    min-width: 0 !important;
}

/* 3. stLayoutWrapper (stMainBlockContainer > stVerticalBlock 내부) */
[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
    flex: 1 !important;
    min-height: 0 !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

/* 4. stHorizontalBlock: fill remaining space */
[data-testid="stHorizontalBlock"] {
    height: 100% !important;
    min-height: 0 !important;
    flex-wrap: nowrap !important;
    gap: 0.25rem !important;
    overflow: hidden !important;
}

/* 5. stColumn: flex container */
[data-testid="stColumn"] {
    flex: 1 1 0px !important;
    min-width: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
    position: relative !important;
}

/* 6. Column 내부 1단계 stVerticalBlock: flex 확장 */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    min-height: 0 !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

/* 7. Column 내부 stLayoutWrapper: flex 확장 */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] {
    flex: 1 !important;
    min-height: 0 !important;
    min-width: 0 !important;
    overflow: hidden !important;
}

/* 8. ★ SCROLLABLE: Column 내부 2단계 stVerticalBlock (stLayoutWrapper 내부) */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
    flex: 1 !important;
    min-height: 0 !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
}

/* 9. 우측 컬럼 하단 패딩 (fixed 입력창이 메시지 가리지 않도록) */
[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child
> [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]
> [data-testid="stVerticalBlock"] {
    padding-bottom: 4rem !important;
}

/* 10. ★ FIXED CHAT INPUT (bypasses DOM nesting issues - position:fixed on element directly) */
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    right: 0 !important;
    left: 62% !important;
    width: 32% !important;
    z-index: 1000 !important;
    background-color: var(--background-color) !important;
    border-top: 1px solid color-mix(in srgb, var(--border-color, #ccc) 30%, transparent) !important;
}

/* ════════════════════════════════════════════
   EXISTING STYLES TO KEEP (thought expander, citation, streaming, etc.)
   ════════════════════════════════════════════ */
[data-testid="stMain"] { overflow: hidden !important; }
[data-testid="stSidebarCollapseButton"] { z-index: 100000 !important; visibility: visible !important; opacity: 1 !important; background-color: color-mix(in srgb, var(--background-color), transparent 20%) !important; }
    /* Sidebar Logo Styles - Minimalist & Native */
    .sidebar-logo-container {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin: 0 0 16px 0 !important;
    }
    .logo-brand {
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        color: var(--text-color) !important;
    }
    .logo-subtitle {
        font-size: 0.75rem !important;
        color: var(--primary-color) !important;
        opacity: 0.8 !important;
    }

    /* Sidebar Settings Labels - Minimalist & Native */
    .settings-label {
        display: block !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-color) !important;
        opacity: 0.9 !important;
        margin: 16px 0 4px 0 !important;
    }
    .settings-sublabel {
        display: block !important;
        font-size: 0.75rem !important;
        color: var(--text-color) !important;
        opacity: 0.6 !important;
        margin: 0 0 8px 0 !important;
    }


header[data-testid="stHeader"] { background: color-mix(in srgb, var(--background-color), transparent 30%) !important; backdrop-filter: blur(12px) !important; z-index: 99999 !important; display: flex !important; }
.stAppDeployButton { display: none !important; }
details.thought-expander { background-color: var(--secondary-background-color); border: 1px solid color-mix(in srgb, var(--faded-text-color) 30%, transparent); border-radius: 8px; margin: 0 0 6px 0; padding: 8px 12px; }
details.thought-expander summary { cursor: pointer; font-size: 0.85em; font-weight: 600; color: var(--text-color); display: flex; align-items: center; gap: 8px; }
details.thought-expander summary::before { content: "\25B6"; font-size: 0.8em; transition: transform 0.2s; }
details[open].thought-expander summary::before { transform: rotate(90deg); }
.thought-container { border-left: 3px solid color-mix(in srgb, var(--primary-color) 70%, transparent); padding: 10px 15px; margin-top: 8px; font-size: 0.85em; color: var(--text-color); opacity: 0.85; max-height: 300px; overflow-y: auto; background-color: color-mix(in srgb, var(--background-color) 50%, transparent); border-radius: 0 4px 4px 0; }
.citation-highlight { background-color: color-mix(in srgb, var(--primary-color) 15%, transparent); border-bottom: 2px dashed var(--primary-color); padding: 0 4px; border-radius: 3px; color: var(--primary-color); cursor: help; display: inline-block; position: relative; }
.streaming-pulse { animation: pulse 1.5s infinite ease-in-out; min-height: 24px; font-size: 0.9em; color: var(--primary-color); margin-bottom: 4px; }
@keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
    .perf-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.8rem;
        margin: 4px 0;
    }
    .perf-row {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
        border-bottom: 1px solid color-mix(in srgb, var(--border-color, #ccc) 20%, transparent);
    }
    .perf-label {
        color: var(--text-color);
        opacity: 0.7;
    }
    .perf-value {
        font-weight: 600;
        color: var(--text-color);
    }
    .perf-status {
        font-size: 0.7rem;
        padding: 1px 4px;
        border-radius: 4px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .status-excellent {
        background-color: color-mix(in srgb, #2ecc71 20%, transparent);
        color: #2ecc71;
    }
    .status-stable {
        background-color: color-mix(in srgb, #f1c40f 20%, transparent);
        color: #f1c40f;
    }
    .status-poor {
        background-color: color-mix(in srgb, #e74c3c 20%, transparent);
        color: #e74c3c;
    }
    .perf-details { margin-top: 2px !important; }
.perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; }
[data-testid="stChatMessage"] div[data-testid="stVerticalBlock"] { gap: 0px !important; }

/* ════════════════════════════════════════════
   12. MOBILE RESPONSIVE
   ════════════════════════════════════════════ */
@media (max-width: 768px) {
    .stApp, [data-testid="stAppViewContainer"] {
        height: auto !important;
        overflow: visible !important;
    }
    [data-testid="stMainBlockContainer"] {
        height: auto !important;
        display: block !important;
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stColumn"] {
        display: block !important;
    }
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {
        max-height: none !important;
        overflow-y: visible !important;
        padding-bottom: 4rem !important;
    }
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        width: 100% !important;
        z-index: 1000 !important;
    }
}
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
