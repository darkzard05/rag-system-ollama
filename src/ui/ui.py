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
        # CSS 로드 실패해도 프로브 슬롯 방출은 유지한다: 최상위 요소 수/순서를
        # 결정론적으로 고정해야 업로드 전환 붕괴가 재발하지 않는다. 빈 CSS로
        # 폴백하고 파일 로그에만 남긴다 (st.error는 DOM에 요소를 추가해 같은
        # 붕괴를 유발하므로 여기서 쓰지 않는다).
        logger.error("Failed to load custom CSS: %s", e)
        css_content = ""
    # [UX] CSS를 본문 <style> 마크업이 아니라 문서 <head>에 영구 주입한다.
    # 본문 마크업 방식은 업로드 on_change 풀 rerun 시 레이아웃 delta가
    # <style> delta보다 먼저 플러시되어 flex:1 규칙이 늦게 적용되고, 그 사이
    # 2열 컨테이너가 콘텐츠 높이(≈420px)로 붕괴하는 깜빡임이 발생한다.
    # <head>에 주입하면 세션 내 재런에서도 스타일이 유지되어 첫 페인트 전에
    # 항상 적용되므로 붕괴 창이 사라진다. window.parent 접근이 필요하므로
    # sandbox가 막는 st.html 대신 st.iframe(높이 0, JS 실행 가능)을 사용한다.

    # 헤더 높이 감지 스크립트 - 매 렌더 동일 슬롯 점유(결정론적 최상위 구조).
    inject_header_height_script()

    from ui.components.autoscroll_stick import BOTTOM_STICK_PX

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

            // =========================================================================
            // 채팅창 스마트 자동 스크롤러 (MutationObserver + Wheel Event 감지)
            // =========================================================================
            if (!window.parent.__chatScrollState) {{
                window.parent.__chatScrollState = {{
                    userScrolledUp: false,
                    observer: null,
                    isTicking: false,
                    yankTarget: -1,
                    initialYanked: false
                }};
            }}
            const scrollState = window.parent.__chatScrollState;

            // 기존 옵저버가 있으면 해제 후 재생성 (Rerun 간 메모리 누수 방지)
            if (scrollState.observer) {{
                scrollState.observer.disconnect();
                scrollState.observer = null;
            }}

            // Anchor on the chat input, NOT `:last-child`. With a real PDF
            // loaded the main block holds 5 stColumn nodes (gutter/spacer
            // columns included) and `:last-child` lands on a 211x40 sliver
            // whose scrollHeight === clientHeight, so every scroll is a
            // permanent no-op. stChatInput is always present in the chat
            // column and lives inside the scroller itself.
            const getChatContainer = () => {{
                const input = doc.querySelector('[data-testid="stChatInput"]');
                if (input) {{
                    const col = input.closest('[data-testid="stColumn"]');
                    const vb = col && col.querySelector(':scope > [data-testid="stVerticalBlock"]');
                    if (vb) return vb;
                }}
                return doc.querySelector(
                    '[data-testid="stMainBlockContainer"] [data-testid="stColumn"]:last-child > [data-testid="stVerticalBlock"]'
                );
            }};

            const scrollToBottom = (force = false) => {{
                const container = getChatContainer();
                if (!container) return;

                // 사용자가 휠을 위로 올려 과거 대화를 보고 있다면 강제 스크롤이 아닌 한 멈춤
                if (!force && scrollState.userScrolledUp) return;

                if (!scrollState.isTicking) {{
                    scrollState.isTicking = true;
                    window.parent.requestAnimationFrame(() => {{
                        // 렌더링 프레임에서는 scroll 이벤트가 rAF 콜백보다 먼저
                        // 처리되므로, 예약 이후에 latch가 갱신될 수 있다.
                        // 갱신된 latch를 다시 확인해 사용자가 위로 올린 경우를 존중한다.
                        if (!force && scrollState.userScrolledUp) {{
                            scrollState.isTicking = false;
                            return;
                        }}
                        scrollState.yankTarget = container.scrollHeight;
                        container.scrollTop = scrollState.yankTarget;
                        scrollState.isTicking = false;
                    }});
                }}
            }};

            const setupChatObserver = () => {{
                const container = getChatContainer();
                if (!container) {{
                    setTimeout(setupChatObserver, 100);
                    return;
                }}

                // 1. 실제 사용자의 마우스 휠 동작 감지
                container.removeEventListener('wheel', container.__wheelHandler);
                container.__wheelHandler = (e) => {{
                    if (e.deltaY < 0) {{
                        // 위로 휠을 굴렸을 때만 스크롤 고정
                        scrollState.userScrolledUp = true;
                    }} else {{
                        // 아래로 휠을 굴려 바닥(BOTTOM_STICK_PX 이내)에 닿으면 자동 스크롤 재개
                        const dist = container.scrollHeight - container.scrollTop - container.clientHeight;
                        if (dist <= {BOTTOM_STICK_PX}) {{
                            scrollState.userScrolledUp = false;
                        }} else if (!container.__wheelResumePending) {{
                            // wheel 이벤트는 스크롤 적용 전(pre-scroll)에
                            // 동기적으로 발생하므로, 이번 펄스로 바닥에
                            // 도달해도 dist는 여전히 임계값을 초과한다.
                            // 스크롤이 착륙한 뒤(post-layout)에 거리 재측정.
                            // __wheelResumePending 가드로 제스처당 1회만 예약.
                            container.__wheelResumePending = true;
                            setTimeout(() => {{
                                container.__wheelResumePending = false;
                                const landed = container.scrollHeight - container.scrollTop - container.clientHeight;
                                if (landed <= {BOTTOM_STICK_PX}) {{
                                    scrollState.userScrolledUp = false;
                                }}
                            }}, 80);
                        }}
                    }}
                }};
                container.addEventListener('wheel', container.__wheelHandler, {{ passive: true }});

                // 스크롤바 드래그·PageUp/PageDown·터치·방향키는 wheel 이벤트를
                // 발생시키지 않는다. wheel 리스너만으로는 latch가 갱신되지 않아
                // 사용자가 과거 대화를 읽으려 올려도 스트리밍마다 바닥으로 끌려간다.
                // scroll 이벤트는 스크롤이 실제로 적용된 뒤 firing되므로
                // 위치를 그대로 신뢰할 수 있어, 어떤 입력 수단으로도 latch를 동기화한다.
                container.removeEventListener('scroll', container.__scrollSyncHandler);
                container.__lastScrollTop = container.scrollTop;
                container.__scrollSyncHandler = () => {{
                    if (container.__lastScrollTop === container.scrollTop) return;
                    container.__lastScrollTop = container.scrollTop;
                    // auto-follow가 스스로 만든 스크롤은 사용자 의도가 아니다.
                    // scroll 이벤트는 메인 스레드가 바쁜 streaming 동안 지연되어
                    // dispatch 시점의 scrollHeight만 최신이고 scrollTop은 yank
                    // 이전 값으로 남아, phantom dist로 latch가 걸린다(실측 t=11749ms
                    // dist=30 → auto-follow 영구 정지). yank가 놓은 위치에 그대로면
                    // 우리 own scroll이므로 무시한다. 사용자가 실제로 위로 올리면
                    // 이 조건을 벗어나 latch가 정상적으로 걸린다.
                    // 임계값은 BOTTOM_STICK_PX 단일 상수 (wheel 해제와 동일 기준).
                    const yank = scrollState.yankTarget;
                    if (yank >= 0 && container.scrollTop >= yank - container.clientHeight - {BOTTOM_STICK_PX}) {{
                        return;
                    }}
                    scrollState.yankTarget = -1;
                    const dist = container.scrollHeight - container.scrollTop - container.clientHeight;
                    scrollState.userScrolledUp = dist > {BOTTOM_STICK_PX};
                }};
                container.addEventListener('scroll', container.__scrollSyncHandler, {{ passive: true }});

                // 2. 새 메시지 및 st.write_stream 토큰 스트리밍 DOM 변화 실시간 감시
                scrollState.observer = new MutationObserver(() => {{
                    scrollToBottom(false);
                }});

                scrollState.observer.observe(container, {{
                    childList: true,
                    subtree: true,
                    characterData: true
                }});

                // 3. 입력창 엔터 키/전송 버튼 클릭 시 무조건 락 해제 및 최하단 스크롤
                const chatInputArea = doc.querySelector('[data-testid="stChatInput"]');
                if (chatInputArea && !chatInputArea.__hasSubmitHook) {{
                    chatInputArea.__hasSubmitHook = true;
                    chatInputArea.addEventListener('keydown', (e) => {{
                        if (e.key === 'Enter' && !e.shiftKey) {{
                            scrollState.userScrolledUp = false;
                            setTimeout(() => scrollToBottom(true), 50);
                        }}
                    }});
                    const sendBtn = chatInputArea.querySelector('button');
                    if (sendBtn) {{
                        sendBtn.addEventListener('click', () => {{
                            scrollState.userScrolledUp = false;
                            setTimeout(() => scrollToBottom(true), 50);
                        }});
                    }}
                }}

                // 초기 진입 1회만 강제 하단 이동한다. 매 rerun 강제 yank는
                // 과거 대화를 읽던 사용자를 바닥으로 끌어내리므로, 이후에는
                // latch를 존중하는 비강제 모드로 전환한다 (Enter/전송 버튼의
                // 명시적 force는 사용자 의도이므로 유지).
                if (!scrollState.initialYanked) {{
                    scrollState.initialYanked = true;
                    scrollToBottom(true);
                }} else {{
                    scrollToBottom(false);
                }}
            }};

            setupChatObserver();
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
    from ui.components.chat import (
        render_chat_input_area,
        render_chat_messages_area,
    )
    from ui.components.viewer import render_pdf_area

    col_pdf, col_chat = st.columns(_COLUMN_RATIO, gap="small")
    with col_pdf:
        t0 = time.perf_counter()
        render_pdf_area()
        logger.info("[PERF] render_pdf_area took %.3fs", time.perf_counter() - t0)
    with col_chat:
        # [TASK5-a VERDICT: stream-DOM 제약 검증 결론 — DOM 순서 유지 확정]
        # render_chat_input_area()를 render_chat_messages_area()보다 먼저 호출해야
        # 한다: 스트리밍이 같은 script run에서 st.write_stream 루프로 돌므로,
        # input 위젯이 그 루프 시작 전에 DOM에 있어야 생성 내내 사라지지 않는다
        # (뒤집으면 messages-then-input + sticky만으로 충분하겠지만, 스트리밍 중
        # 입력창이 통째로 사라져 되돌릴 수 없으므로 DOM 순서 유지를 선택).
        # 시각 순서는 main.css order:1이, 키보드 접근성은 :focus-visible 링이 담당.
        # Trade-off 기록: Tab 순서는 DOM 순서(input 우선)를 따르므로 시각 순서와
        # 다를 수 있다 — 스트리밍 안정성이 우선하므로 허용하고 주석으로 고정한다.
        # (기존 [FIX-STREAM-INPUT] 주석의 결론을 TASK 5에서 재검증 후 확정)

        t0 = time.perf_counter()
        render_chat_input_area()
        logger.info(
            "[PERF] render_chat_input_area took %.3fs", time.perf_counter() - t0
        )

        render_chat_messages_area()
        logger.info(
            "[PERF] render_chat_messages_area took %.3fs", time.perf_counter() - t0
        )
