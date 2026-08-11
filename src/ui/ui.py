# src/ui/ui.py
"""
Streamlit UI 컴포넌트들을 조립하여 전체 레이아웃을 구성하는 메인 UI 파일.

싱글-로우 레이아웃 (개선됨):
- 컨텐츠 로우: PDF 뷰어(왼쪽) + 채팅 메시지(오른쪽), flex: 1로 viewport 채움
- PDF 컨트롤은 PDF 열 하단(뷰어 아래)에 위치
- 채팅 입력은 채팅 열 내부 하단에 고정 (sticky position)
- 더 이상 절대 위치 오버레이 독이 없음
- 헤더 높이는 JS로 실시간 감지하여 CSS 변수(--header-h)에 반영
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

_COLUMN_RATIO: list[int] = [42, 58]


def inject_header_height_script() -> None:
    """Inject JS to detect Streamlit header height and set CSS variable.

    Uses window.parent.document to reach the parent page from the iframe,
    bypassing Streamlit's innerHTML sanitization on st.markdown.
    """
    import streamlit.components.v1 as components

    # deprecated but functional; st.html sandbox blocks parent-frame access — see p1_sthtml_verdict.md
    components.html(
        """
        <script>
        try {
            const doc = window.parent.document;
            // Pass-1 fail-safe: apply the 60px fallback before the header even exists.
            doc.documentElement.style.setProperty('--header-h', '3.75rem');
            const header = doc.querySelector('[data-testid="stHeader"]');
            if (header) {
                const h = header.offsetHeight;
                doc.documentElement.style.setProperty('--header-h', h + 'px');
            }
            window.addEventListener('resize', () => {
                const h2 = doc.querySelector('[data-testid="stHeader"]')?.offsetHeight || 60;
                doc.documentElement.style.setProperty('--header-h', h2 + 'px');
            });
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', () => {
                    const h3 = doc.querySelector('[data-testid="stHeader"]')?.offsetHeight || 60;
                    doc.documentElement.style.setProperty('--header-h', h3 + 'px');
                });
            }
        } catch (e) {
            // Best-effort: layout defaults remain intact if the parent frame is unreachable.
        }
        </script>
        """,
        height=0,
    )


@st.cache_data(show_spinner=False)
def _load_css() -> str:
    css_path = Path(__file__).parent / "styles" / "main.css"
    return css_path.read_text(encoding="utf-8")


def inject_custom_css() -> None:
    if "css_script_injected" not in st.session_state:
        inject_header_height_script()  # 세션당 1회만 주입 (리스너는 부모 window에 유지)
        st.session_state.css_script_injected = True
    try:
        css_content = _load_css()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except (FileNotFoundError, PermissionError, OSError) as e:
        st.error(f"Failed to load custom CSS: {e}")


def render_main_content() -> None:
    """Render the two-column layout with chat input inside the chat column.

    ┌────────────────────────┬──────────────────────────┐
    │  PDF Viewer            │  Chat Messages           │  ← content row (flex: 1)
    │  ─── PDF Controls ───  │  ① Context Strip         │
    │                        │  ② Status Banner         │
    │                        │  Messages (flex: 1)      │
    │                        │  ─── Chat Input ───      │  ← sticky bottom inside column
    └────────────────────────┴──────────────────────────┘
    """
    from ui.components.chat import render_chat_input_area, render_chat_messages_area
    from ui.components.viewer import render_pdf_area

    col_pdf, col_chat = st.columns(_COLUMN_RATIO, gap="small")
    with col_pdf:
        render_pdf_area()
    with col_chat:
        render_chat_messages_area()
        render_chat_input_area()
