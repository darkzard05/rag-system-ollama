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
        """
<style>
.stApp, [data-testid="stAppViewContainer"] {
    height: 100vh !important;
    overflow: hidden !important;
}
[data-testid="stMainBlockContainer"] {
    height: calc(100vh - var(--header-h, 60px)) !important;
}
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}
[data-testid="stChatInput"] {
    position: fixed !important;
    bottom: 0 !important;
    right: 0 !important;
    left: 50% !important;
    width: 50% !important;
    z-index: 1000;
    background-color: var(--background-color) !important;
    padding-bottom: 1rem !important;
    padding-top: 0.5rem !important;
}
[data-testid="stMain"] { overflow: hidden !important; }
[data-testid="stSidebarCollapseButton"] { z-index: 100000 !important; visibility: visible !important; opacity: 1 !important; background-color: color-mix(in srgb, var(--background-color), transparent 20%) !important; }
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
.perf-details { margin-top: 2px !important; }
.perf-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 8px; }
[data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; gap: 8px !important; }
div[data-testid="stColumn"] { min-width: 60px !important; flex: 0 1 auto !important; margin: 0 !important; }
div[data-testid="stColumn"] button { border-radius: 20px !important; padding: 4px 12px !important; font-size: 12px !important; width: 100% !important; }
[data-testid="stChatMessage"] div[data-testid="stVerticalBlock"] { gap: 0px !important; }
html body [data-testid="stMainBlockContainer"] {
    height: calc(100vh - var(--header-h, 60px)) !important;
    height: calc(100dvh - var(--header-h, 60px)) !important;
    overflow: hidden !important;
}
[data-testid="stColumn"] {
    display: flex !important;
    flex-direction: column !important;
    flex: 1 1 0px !important;
    min-height: 0 !important;
    position: relative !important;
}
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    overflow-y: auto !important;
    padding-bottom: 80px !important;
}
[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"] {
    padding-bottom: 80px !important;
}
@media (max-width: 768px) {
    .stApp, [data-testid="stAppViewContainer"] { height: auto !important; overflow: visible !important; }
    [data-testid="stMainBlockContainer"] { height: auto !important; }
    [data-testid="stColumn"] > [data-testid="stVerticalBlock"] { max-height: none !important; overflow-y: visible !important; }
    [data-testid="stChatInput"] { left: 0 !important; width: 100% !important; position: fixed !important; }
}
</style>
""",
        unsafe_allow_html=True,
    )
    inject_column_inline_styles()


def inject_column_inline_styles():
    st.components.v1.html(
        """
    <script>
    (function() {
        function apply() {
            var w = window.parent;
            if (!w) return;
            var cols = w.document.querySelectorAll('[data-testid="stColumn"]');
            for (var i = 0; i < cols.length; i++) {
                cols[i].style.setProperty('display', 'flex', 'important');
                cols[i].style.setProperty('flex-direction', 'column', 'important');
                cols[i].style.setProperty('flex', '1 1 0px', 'important');
                cols[i].style.setProperty('min-height', '0', 'important');
            }
            var vbs = w.document.querySelectorAll('[data-testid="stColumn"] > [data-testid="stVerticalBlock"]');
            for (var i = 0; i < vbs.length; i++) {
                vbs[i].style.setProperty('overflow-y', 'auto', 'important');
                vbs[i].style.setProperty('overflow-x', 'hidden', 'important');
                vbs[i].style.setProperty('flex', '1', 'important');
                vbs[i].style.setProperty('min-height', '0', 'important');
            }
            var last = w.document.querySelector('[data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]');
            if (last) last.style.setProperty('padding-bottom', '80px', 'important');
        }
        if (w.document.readyState === 'loading') {
            w.document.addEventListener('DOMContentLoaded', apply);
        } else {
            apply();
        }
        var timer = null;
        var obs = new MutationObserver(function() {
            if (timer) clearTimeout(timer);
            timer = setTimeout(apply, 200);
        });
        obs.observe(w.document.body, { childList: true, subtree: true });
    })();
    </script>
    """,
        height=0,
        width=0,
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
