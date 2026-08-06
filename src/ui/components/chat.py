"""
채팅 인터페이스 컴포넌트 - 통합 타임라인 버전.

모든 이벤트(문서 업로드, 분석 진행, 대화 메시지, 생각 과정)를
단일 연대기순 타임라인으로 렌더링합니다.
"""

import contextlib
import html
import logging
import time
from collections.abc import Callable
from typing import Any

import streamlit as st

from common.config import MSG_CHAT_GUIDE, UI_TIMELINE_POLL_SECONDS
from common.utils import (
    apply_tooltips_to_response,
    normalize_latex_delimiters,
)
from core.session import SessionManager
from ui.widget_keys import cancel_rebuild_key, jump_key

logger = logging.getLogger(__name__)


def _handle_page_jump(p: int) -> None:
    """참조 페이지 이동 버튼 콜백입니다."""
    SessionManager.set(
        "pdf_target_page",
        {"page": int(p), "source": "manual", "ts": time.time()},
    )
    SessionManager.set("current_page", p)
    st.toast(f"📄 {p}페이지로 이동 중...", icon="📄")
    # 전체 리런이 필요하다 (뷰어 fragment가 run_every=2.0으로 폴링 중이어도
    # popover 점프는 즉시 반영되어야 하므로 st.rerun()으로 전체 재실행).
    st.rerun()


def _extract_reference_pages(documents: list[Any]) -> list[int]:
    """문서 메타데이터에서 참조 페이지 번호 목록을 추출합니다."""
    pages: set[int] = set()
    for d in documents:
        meta = (
            getattr(d, "metadata", {})
            if hasattr(d, "metadata")
            else d.get("metadata", {})
        )
        with contextlib.suppress(ValueError, TypeError):
            page = meta.get("page")
            if page is not None:
                pages.add(int(page))
            for pg in meta.get("pages") or []:
                pages.add(int(pg))
    return sorted(pages)


def _render_references_popover(
    msg_id: str,
    documents: list[Any] | None,
    on_page_jump: Callable[[int], None] | None = None,
) -> None:
    """참조 페이지 popover와 페이지 이동 버튼을 렌더링합니다."""
    with st.popover("📑 참조 페이지", use_container_width=False):
        if not documents:
            return

        pages = _extract_reference_pages(documents)
        if not pages:
            return
        cols = st.columns(min(len(pages), 5))
        for idx, p in enumerate(pages):
            clicked = cols[idx % len(cols)].button(
                f"{p}p",
                key=jump_key(msg_id, p, idx),
                use_container_width=True,
            )
            if clicked and on_page_jump is not None:
                on_page_jump(p)


def render_message(
    role: str,
    content: str,
    thought: str | None = None,
    documents: list[Any] | None = None,
    metrics: dict | None = None,
    processed_content: str | None = None,
    msg_type: str = "general",
    wrap_in_container: bool = True,
    msg_index: int = 0,
    is_latest: bool = False,
    **kwargs,
) -> None:
    """메시지를 렌더링하는 통합 엔진."""
    avatar_icon = "🤖" if role == "assistant" else "👤"
    msg_id = kwargs.get("msg_id", f"msg_{msg_index}")

    with (
        st.chat_message(role, avatar=avatar_icon)
        if wrap_in_container
        else st.container()
    ):
        # 오류 메시지: 부분 답변이 있으면 먼저 본문을 보존하고 오류를 안내한다
        error = kwargs.get("error")
        if error:
            has_content = bool(processed_content or content)
            if has_content:
                if processed_content:
                    st.markdown(processed_content, unsafe_allow_html=True)
                else:
                    display_text = (
                        html.escape(content) if role == "assistant" else content
                    )
                    display_text = normalize_latex_delimiters(display_text)
                    st.markdown(display_text, unsafe_allow_html=(role == "assistant"))
            st.error(f"⚠️ {error}")
            return

        # 본문 내용
        if processed_content:
            st.markdown(processed_content, unsafe_allow_html=True)
        else:
            display_text = content
            if role == "assistant":
                display_text = html.escape(display_text)

            display_text = normalize_latex_delimiters(display_text)
            if role == "assistant" and documents:
                display_text = apply_tooltips_to_response(display_text, documents)
            st.markdown(display_text, unsafe_allow_html=(role == "assistant"))

        # 완료된 어시스턴트 메시지의 하단 정보
        if role == "assistant" and msg_type == "general":
            # 완료 요약 + 출처 미리보기 (기본 노출).
            # 스트리밍 중 st.status 피드백이 완료 시 사라지는 문제를 완화한다.
            if (content or processed_content or "").strip():
                if documents:
                    pages = _extract_reference_pages(documents)
                    page_txt = " · ".join(f"p.{p}" for p in pages[:4])
                    if len(pages) > 4:
                        page_txt += f" 외 {len(pages) - 4}건"
                    st.caption(
                        f"✅ 답변 생성 완료 · 📚 참조 {len(documents)}건"
                        + (f" · {page_txt}" if page_txt else "")
                    )
                else:
                    st.caption("✅ 답변 생성 완료")

            # 참조 페이지 popover는 documents가 있는 모든 완료된 어시스턴트
            # 메시지에 렌더링한다. 버튼 키는 msg_id 기반이므로 이전 답변과
            # 공존해도 안전하다 (최신 여부(is_latest)와 무관).
            if documents:
                _render_references_popover(
                    msg_id, documents, on_page_jump=_handle_page_jump
                )

            # 성능 지표
            if metrics:
                total_time = metrics.get("total_time", 0)
                retrieved = metrics.get("retrieved_chunks", 0)
                model = (
                    kwargs.get("model", "")
                    or SessionManager.get("last_selected_model", "")
                    or "model"
                )
                st.caption(f"⚡ {total_time:.1f}s · 🔍 {retrieved} chunks · 🤖 {model}")

        # 상세 사고 과정 (LLM reasoning 로그) — 답변·출처 아래, 기본 접힘.
        if (
            role == "assistant"
            and thought
            and thought.strip()
            and msg_type != "streaming"
        ):
            with st.expander("🧠 상세 사고 과정", expanded=False):
                st.markdown(thought)


def _cancel_rebuild(sid: str) -> None:
    """문서 분석 재구축 취소 요청 콜백입니다."""
    SessionManager.set("rebuild_cancelled", True, session_id=sid)
    st.rerun()


def _handle_stop_generation(sid: str) -> None:
    """스트리밍 중단 요청 콜백입니다. 소비 스레드가 부분 결과를 확정하도록 합니다."""
    SessionManager.set("generation_cancel", True, session_id=sid)
    st.rerun()


def _render_guidance_panel() -> None:
    """빈 대화 상태의 단일 가이드 메시지를 렌더링합니다."""
    st.chat_message("system", avatar="⚙️").markdown(MSG_CHAT_GUIDE)


def _render_doc_context_inline(sid: str) -> None:
    """문서 컨텍스트를 타임라인 첫 메시지로 렌더링합니다 (네이티브)."""
    file_name = str(SessionManager.get("last_uploaded_file_name", "", sid) or "")
    if not file_name:
        pdf_path = str(SessionManager.get("pdf_file_path", "", sid) or "")
        if pdf_path:
            file_name = pdf_path.replace("\\", "/").rsplit("/", 1)[-1]
    if not file_name:
        return

    is_building = bool(SessionManager.get("is_building_rag", False, sid))
    is_ready = SessionManager.is_ready_for_chat(session_id=sid)
    has_error = bool(SessionManager.get("pdf_processing_error", "", sid))

    doc_stats = SessionManager.get("doc_stats", {}, sid) or {}
    doc_loaded = bool(SessionManager.get("pdf_processed", False, sid))
    cache_tag = ""
    if doc_loaded and doc_stats:
        cache_tag = " [캐시]" if doc_stats.get("cache_used") else " [신규]"

    with st.chat_message("system", avatar="📄"):
        if is_building:
            st.caption(f"📎 {file_name} · 분석 중...{cache_tag}")
            # 진행 상황은 메시지 루프에서 build_progress 타입으로 처리
        elif has_error:
            st.caption(f"📎 {file_name} · 오류 발생{cache_tag}")
        elif is_ready:
            st.caption(f"📎 {file_name} · 준비 완료{cache_tag}")
        else:
            st.caption(f"📎 {file_name} · 대기 중{cache_tag}")


@st.fragment(run_every=UI_TIMELINE_POLL_SECONDS)
def _render_unified_timeline(current_sid: str) -> None:
    """
    통합 타임라인 렌더링 (단일 패스).
    메시지 리스트에 저장된 모든 타입의 메시지를 시간 순서대로 렌더링.
    """
    messages = SessionManager.get_messages() or []

    # 빈 대화일 때: 문서 컨텍스트가 있으면 표시, 없으면 가이드
    if not messages:
        if SessionManager.get("last_uploaded_file_name", "", current_sid):
            _render_doc_context_inline(current_sid)
        else:
            _render_guidance_panel()
        return

    # 문서 컨텍스트가 있고 메시지가 있으면, 첫 메시지로 문서 상태 표시
    has_doc_context = bool(
        SessionManager.get("last_uploaded_file_name", "", current_sid)
    )
    doc_rendered = False

    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        mtype = msg.get("msg_type", "general")
        is_latest = i == len(messages) - 1

        # 시스템/로그 메시지 처리
        if role == "system":
            if mtype == "build_progress":
                # 빌드 진행 상황
                progress = msg.get("progress", 0)
                status_text = msg.get("status", "진행 중...")

                # 상태를 생성 시점에 결정한다. (2초 폴링이 매번
                # running 상태로 생성 후 complete로 전환하면 깜빡임이 발생)
                if msg.get("error"):
                    label, state, expanded = "❌ 분석 실패/취소", "error", True
                elif progress >= 100 or msg.get("done"):
                    label, state, expanded = "✅ 분석 완료", "complete", False
                else:
                    label, state, expanded = (
                        f"문서 분석 중: {status_text}",
                        "running",
                        True,
                    )

                with (
                    st.chat_message("system", avatar="🔄"),
                    st.status(label, expanded=expanded, state=state),
                ):
                    st.progress(progress / 100)
                    st.caption(f"{progress}% 완료")
                    if state == "running" and msg.get("cancelable", True):
                        st.button(
                            "분석 취소",
                            key=cancel_rebuild_key(current_sid),
                            on_click=_cancel_rebuild,
                            args=(current_sid,),
                            use_container_width=True,
                        )

                    # 진행 로그 표시
                    if msg.get("logs"):
                        with st.expander("진행 로그", expanded=False):
                            for log in msg.get("logs", [])[-10:]:
                                st.text(f"▹ {log}")
                continue

            elif mtype == "build_error":
                # 빌드 에러
                with st.chat_message("system", avatar="❌"):
                    st.error(msg.get("error", "알 수 없는 오류"))
                continue

            elif mtype == "log":
                # 상태 로그 (작은 캡션으로)
                st.caption(f"ℹ️ {content}")
                continue

            # 일반 시스템 메시지 (문서 업로드 알림 등)
            if not doc_rendered and has_doc_context:
                _render_doc_context_inline(current_sid)
                doc_rendered = True

            with st.chat_message("system", avatar="⚙️"):
                st.markdown(content)
            continue

        # READY_FOR_QUERY 같은 내부 메시지는 스킵
        if content == "READY_FOR_QUERY":
            continue

        # 스트리밍 중인 메시지
        if mtype == "streaming":
            # 스트리밍 중 오류가 실린 메시지는 즉시 표면화
            if msg.get("error"):
                with st.chat_message("assistant", avatar="🤖"):
                    st.error(str(msg.get("error")))
                continue
            status_text = msg.get("status", "생성 중...")
            thought = msg.get("thought", "")

            with st.chat_message("assistant", avatar="🤖"):
                # 실시간 상태 표시
                with st.status(
                    f"🤔 {status_text}", expanded=True, state="running"
                ) as status:
                    if thought:
                        status.write(thought)

                # 스트리밍 내용 표시 (커서 포함)
                display_content = msg.get("content", "")
                if display_content:
                    display_content = normalize_latex_delimiters(
                        html.escape(display_content)
                    )
                    st.markdown(display_content + " ▌", unsafe_allow_html=True)

                # 중단 버튼 (취소 요청 → 소비 스레드가 누적 부분 콘텐츠를 확정)
                if not SessionManager.get("generation_cancel", False, current_sid):
                    st.button(
                        "⏹ 중단",
                        key=f"stop_gen_{msg.get('msg_id')}",
                        on_click=_handle_stop_generation,
                        args=(current_sid,),
                        use_container_width=True,
                    )
            continue

        # 일반 완료된 메시지 (사용자/어시스턴트)
        render_message(
            role=role,
            content=content,
            thought=msg.get("thought"),
            documents=msg.get("documents"),
            metrics=msg.get("metrics"),
            processed_content=msg.get("processed_content"),
            msg_type=mtype,
            msg_index=i,
            msg_id=msg.get("msg_id"),
            is_latest=is_latest,
            error=msg.get("error"),
        )


# ---------------------------------------------------------------------------
# 메인 렌더링
# ---------------------------------------------------------------------------


def render_chat_messages_area() -> None:
    """Renders the chat column: unified timeline + sticky input."""
    current_sid = SessionManager.get_session_id()

    # 통합 타임라인 fragment (단일 진입점, 2초 폴링)
    _render_unified_timeline(current_sid)


def _resolve_chat_input_state(sid: str) -> tuple[str, bool]:
    """채팅 입력의 placeholder/disabled 상태를 결정하는 순수 함수입니다."""
    is_generating = bool(SessionManager.get("is_generating_answer", False, sid))
    is_ready = SessionManager.is_ready_for_chat(session_id=sid)

    if is_generating:
        return "AI가 답변 생성 중입니다...", True
    if not is_ready:
        return MSG_CHAT_GUIDE, True
    return "추가 질문을 입력하세요...", False


def render_chat_input_area() -> None:
    """Renders the native st.chat_input() at the bottom of the chat column."""
    current_sid = SessionManager.get_session_id()
    input_placeholder, input_disabled = _resolve_chat_input_state(current_sid)

    user_query = st.chat_input(
        input_placeholder, disabled=input_disabled, key="main_chat_input"
    )

    if user_query and not input_disabled:
        query_text = user_query.strip()
        if query_text:
            from ui.components.streaming import (
                friendly_error_message,
                start_streaming_turn,
            )

            SessionManager.add_message("user", query_text, session_id=current_sid)
            SessionManager.set("is_generating_answer", True, current_sid)
            try:
                model_name = (
                    SessionManager.get("last_selected_model", session_id=current_sid)
                    or ""
                )
                start_streaming_turn(current_sid, query_text, model_name)
            except Exception as exc:
                # 보장: 배경 소비자 시작 전 예외가 발생해도 플래그가 남지 않도록
                # 해제하고, 타임라인에 친화적 오류 메시지를 표면화한다.
                logger.exception("[CHAT] 답변 생성 시작 실패: %s", exc)
                SessionManager.set("is_generating_answer", False, current_sid)
                SessionManager.set("generation_cancel", True, current_sid)
                SessionManager.add_message(
                    "assistant",
                    "",
                    msg_type="general",
                    error=friendly_error_message(exc),
                    session_id=current_sid,
                )
            st.rerun()
