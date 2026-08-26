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

import json
import logging
import time
from pathlib import Path

import streamlit as st

logger = logging.getLogger(__name__)

# Header height fallback, shared by the JS probe (ui.py) and the CSS contract
# (main.css --header-h) so the two never drift (C5). 60px == 3.75rem.
HEADER_H_FALLBACK_PX: int = 60
HEADER_H_FALLBACK_REM: str = "3.75rem"

_COLUMN_RATIO: list[int] = [42, 58]


def inject_header_height_script() -> None:
    """Inject JS to detect Streamlit header height and set CSS variable.

    Uses window.parent.document to reach the parent page from the iframe,
    bypassing Streamlit's innerHTML sanitization on st.markdown.
    """
    # st.iframe embeds the HTML in an iframe with same-origin access and JS
    # execution — required here because the script reaches window.parent.document
    # (st.html's sandbox would block that). The content is a <script> only, so
    # height="content" auto-measures to ~0px (effectively invisible), matching
    # the old components.html(height=0). Plain string (not f-string): the JS is
    # heavily braced. Inject the single
    # fallback value via replace so JS braces stay literal.
    js = """
        <script>
        try {
            const doc = window.parent.document;
            // Pass-1 fail-safe: apply the fallback before the header even exists.
            doc.documentElement.style.setProperty('--header-h', '__HEADER_H_FALLBACK__');
            const header = doc.querySelector('[data-testid="stHeader"]');
            if (header) {
                const h = header.offsetHeight;
                doc.documentElement.style.setProperty('--header-h', h + 'px');
            }
            window.addEventListener('resize', () => {
                const h2 = doc.querySelector('[data-testid="stHeader"]')?.offsetHeight || __HEADER_H_FALLBACK_PX__;
                doc.documentElement.style.setProperty('--header-h', h2 + 'px');
            });
            if (window.visualViewport) {
                window.visualViewport.addEventListener('resize', () => {
                    const h3 = doc.querySelector('[data-testid="stHeader"]')?.offsetHeight || __HEADER_H_FALLBACK_PX__;
                    doc.documentElement.style.setProperty('--header-h', h3 + 'px');
                });
            }
        } catch (e) {
            // Best-effort: layout defaults remain intact if the parent frame is unreachable.
        }
        </script>
        """.replace("__HEADER_H_FALLBACK__", HEADER_H_FALLBACK_REM).replace(
        "__HEADER_H_FALLBACK_PX__", str(HEADER_H_FALLBACK_PX)
    )
    st.iframe(js, height="content")


@st.cache_data(show_spinner=False)
def _load_css() -> str:
    css_path = Path(__file__).parent / "styles" / "main.css"
    return css_path.read_text(encoding="utf-8")


def inject_custom_css() -> None:
    """문서 <head>에 커스텀 CSS + 헤더 높이 감지 스크립트를 영구 주입한다.

    ⚠️ 결정적 버그 수정 (업로드 전환 붕괴):
    이전 구현은 ``CSS_INJECTED_KEY`` 가드로 ``st.iframe`` 호출을 "세션당 1회"만
    했었다. 그 결과 첫 렌더는 ``stVerticalBlock`` 자식이
    ``[stIFrame, stIFrame, contentRow]``(3개) 였으나, 업로드 on_change 풀 rerun
    시 가드는 iframe 호출을 **생략** → ``[contentRow]``(1개)만 방출되었다.
    Streamlit은 줄어든 최상위 요소 수를 슬롯 재활용으로 맞추는데, 비워진
    iframe 슬롯이 ``stLayoutWrapper`` 로 재탄생하며 PDF 네비("⬅️ 이전 다음") 등
    컬럼 자식이 그곳으로 호이스팅되어 **형제 wrapper 2개 → flex:1 50:50 분할
    → 840→420 붕괴** 가 발생했다 (실측: main_content_wrapper_h 840→420, appRoots=1).

    해결: iframe을 **렌더마다 항상 동일하게 방출**하여 최상위 요소 수/순서를
    결정론적으로 고정한다. <head> 주입 스크립트 자체가 멱등(idempotent)이라
    중복 방출해도 리스너 누수/스타일 덮어쓰기 없이 안전하다. 가드 변수는
    제거한다(매 렌더 동일 슬롯 점유 보장).
    """
    try:
        css_content = _load_css()
    except (FileNotFoundError, PermissionError, OSError) as e:
        st.error(f"Failed to load custom CSS: {e}")
        return
    # [UX] CSS를 본문 <style> 마크업이 아니라 문서 <head>에 영구 주입한다.
    # 본문 마크업 방식은 업로드 on_change 풀 rerun 시 레이아웃 delta가
    # <style> delta보다 먼저 플러시되어 flex:1 규칙이 늦게 적용되고, 그 사이
    # 2열 컨테이너가 콘텐츠 높이(≈420px)로 붕괴하는 깜빡임이 발생한다.
    # <head>에 주입하면 세션 내 재런에서도 스타일이 유지되어 첫 페인트 전에
    # 항상 적용되므로 붕괴 창이 사라진다. window.parent 접근이 필요하므로
    # sandbox가 막는 st.html 대신 st.iframe(높이 0, JS 실행 가능)을 사용한다.

    # 헤더 높이 감지 스크립트 - 매 렌더 동일 슬롯 점유(결정론적 최상위 구조).
    inject_header_height_script()

    css_json = json.dumps(css_content)
    js = f"""
        <script>
        try {{
            const css = {css_json};
            const doc = window.parent.document;
            let el = doc.getElementById('main-app-css');
            if (!el) {{
                el = doc.createElement('style');
                el.id = 'main-app-css';
                doc.head.appendChild(el);
            }}
            el.textContent = css;
        }} catch (e) {{}}
        </script>
    """
    st.iframe(js, height="content")


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
        t0 = time.perf_counter()
        render_pdf_area()
        logger.debug("[PERF] render_pdf_area took %.3fs", time.perf_counter() - t0)
    with col_chat:
        # [FIX-STREAM-INPUT] Render the input BEFORE the messages area on purpose.
        # render_chat_messages_area() -> _run_active_stream_in_timeline() runs a
        # BLOCKING synchronous stream loop inside this same script run; st.chat_input()
        # must be created BEFORE that loop starts so the widget is already in the DOM
        # while tokens stream (otherwise the input vanishes for the whole generation and
        # reappears only after). CSS `order:1` on the input wrapper re-pins it visually to
        # the column bottom, so reordering the DOM does not disturb the bottom-pin layout.
        t0 = time.perf_counter()
        render_chat_input_area()
        logger.debug(
            "[PERF] render_chat_input_area took %.3fs", time.perf_counter() - t0
        )
        render_chat_messages_area()
        logger.debug(
            "[PERF] render_chat_messages_area took %.3fs", time.perf_counter() - t0
        )
