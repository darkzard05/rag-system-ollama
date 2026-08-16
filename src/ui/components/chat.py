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

from common.config import DEFAULT_OLLAMA_MODEL, MSG_CHAT_GUIDE, UI_TIMELINE_POLL_SECONDS
from common.utils import (
    apply_tooltips_to_response,
    fast_hash,
    normalize_latex_delimiters,
)
from core.session import SessionManager
from ui.components.common import (
    AVATARS,
    get_doc_metadata,
    navigate_to_page,
    status_line,
    ui_error,
)
from ui.widget_keys import (
    MAIN_CHAT_INPUT_KEY,
    cancel_rebuild_key,
    jump_key,
)

logger = logging.getLogger(__name__)


def _handle_page_jump(p: int) -> None:
    """참조 페이지 이동 버튼 콜백입니다."""
    SessionManager.set(
        "pdf_target_page",
        {"page": int(p), "source": "manual", "ts": time.time()},
    )
    navigate_to_page(int(p))
    st.toast(f"Moving to page {p}...")
    # 전체 리런이 필요하다 (뷰어 fragment가 run_every=2.0으로 폴링 중이어도
    # popover 점프는 즉시 반영되어야 하므로 st.rerun()으로 전체 재실행).
    st.rerun()


def _doc_stable_id(doc: object) -> str:
    """문서의 안정 식별자(doc_id 메타 또는 content 해시)를 반환합니다."""
    meta = get_doc_metadata(doc)
    content = ""
    if hasattr(doc, "page_content"):
        content = getattr(doc, "page_content", "") or ""
    else:
        content = doc.get("page_content", "") if isinstance(doc, dict) else ""
    doc_id = meta.get("doc_id")
    if doc_id is not None:
        return str(doc_id)
    return fast_hash(content)


def _handle_doc_jump(doc_id: str) -> None:
    """인용의 안정 doc_id로 문서를 찾아 첫 페이지로 이동합니다."""
    docs = SessionManager.get("documents", []) or []
    target_page = 1
    found = False
    for d in docs:
        if _doc_stable_id(d) == doc_id:
            found = True
            page = get_doc_metadata(d).get("page")
            with contextlib.suppress(ValueError, TypeError):
                if page is not None:
                    target_page = int(page)
            break
    if not found:
        # doc_id만으로 페이지를 알 수 없으면 1p 기준으로 이동한다.
        target_page = 1
    SessionManager.set(
        "pdf_target_page",
        {"page": target_page, "source": "citation", "ts": time.time()},
    )
    navigate_to_page(target_page)
    st.toast("Moving to cited document...")
    st.rerun()


def _render_citation_anchors(
    citations: list[dict[str, Any]], documents: list[Any] | None
) -> None:
    """citations[] 배열을 클릭 가능한 출처 앵커로 렌더합니다.

    PRIMARY 소스는 citations[] (안정 doc_id 기반)이며, 인라인 [doc:N] 폴백은
    apply_tooltips_to_response가 담당합니다. doc_id가 documents에서 실제 문서를
    가리키면 점프 버튼을 노출합니다.
    """
    if not citations:
        return
    doc_ids = {_doc_stable_id(d) for d in (documents or [])}
    with st.container():
        st.caption("Sources")
        for idx, cit in enumerate(citations):
            sid = str(cit.get("doc_id", ""))
            span = cit.get("text_span") or cit.get("section") or f"Source {idx + 1}"
            label = html.escape(str(span))[:160]
            if sid in doc_ids:
                if st.button(
                    f"{idx + 1}. {label}",
                    key=f"cit_doc_{sid}_{idx}",
                    use_container_width=True,
                ):
                    _handle_doc_jump(sid)
            else:
                st.markdown(
                    f'<span data-doc-id="{html.escape(sid)}">{idx + 1}. {label}</span>',
                    unsafe_allow_html=True,
                )


def _extract_reference_pages(documents: list[Any]) -> list[int]:
    """문서 메타데이터에서 참조 페이지 번호 목록을 추출합니다."""
    pages: set[int] = set()
    for d in documents:
        meta = get_doc_metadata(d)
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
    citations: list[dict[str, Any]] | None = None,
) -> None:
    """참조 popover: 페이지 이동 버튼 + doc 기반 인용 점프를 렌더링합니다."""
    with st.popover("References", use_container_width=False):
        if not documents and not citations:
            return

        pages = _extract_reference_pages(documents or [])
        if pages:
            st.caption("By page")
            cols = st.columns(min(len(pages), 5))
            for idx, p in enumerate(pages):
                clicked = cols[idx % len(cols)].button(
                    f"{p}p",
                    key=jump_key(msg_id, p, idx),
                    use_container_width=True,
                )
                if clicked and on_page_jump is not None:
                    on_page_jump(p)

        # P3: citations[] 기반 doc 점프 (안정 doc_id).
        doc_citations = [c for c in (citations or []) if c.get("doc_id") is not None]
        if doc_citations:
            doc_ids = {_doc_stable_id(d) for d in (documents or [])}
            st.caption("By doc")
            for idx, cit in enumerate(doc_citations):
                sid = str(cit.get("doc_id"))
                if sid in doc_ids:
                    label = cit.get("section") or cit.get("text_span") or f"doc {sid}"
                    if st.button(
                        f"{idx + 1}. {label}",
                        key=f"pop_doc_{msg_id}_{sid}_{idx}",
                        use_container_width=True,
                    ):
                        _handle_doc_jump(sid)


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
    process: dict | None = None,
    citations: list[dict[str, Any]] | None = None,
    **kwargs,
) -> None:
    """메시지를 렌더링하는 통합 엔진."""
    avatar_icon = AVATARS["assistant"] if role == "assistant" else AVATARS["user"]
    msg_id = kwargs.get("msg_id", f"msg_{msg_index}")
    citations = citations or kwargs.get("citations")

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
            ui_error(f"Error: {error}")
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
                display_text = apply_tooltips_to_response(
                    display_text, documents, citations=citations
                )
            st.markdown(display_text, unsafe_allow_html=(role == "assistant"))

        # 완료된 어시스턴트 메시지의 하단 정보
        if role == "assistant" and msg_type == "general":
            cancelled = bool(kwargs.get("cancelled", False))
            # 완료 요약 + 출처 미리보기 (기본 노출).
            # 스트리밍 중 st.status 피드백이 완료 시 사라지는 문제를 완화한다.
            if (content or processed_content or "").strip():
                if cancelled:
                    # 중단 확정 상태: 부분 답변 보존 안내 (uiux-fix-p1 INT-2)
                    st.caption("Stopped · Partial answer preserved")
                elif documents:
                    pages = _extract_reference_pages(documents)
                    page_txt = status_line(*(f"p.{p}" for p in pages[:4]))
                    if len(pages) > 4:
                        page_txt += f" +{len(pages) - 4} more"
                    st.caption(
                        status_line(
                            "Answer complete",
                            f"{len(documents)} references",
                            page_txt,
                        )
                    )
                else:
                    st.caption("Answer complete")

            # 참조 페이지 popover는 documents가 있는 모든 완료된 어시스턴트
            # 메시지에 렌더링한다. 버튼 키는 msg_id 기반이므로 이전 답변과
            # 공존해도 안전하다 (최신 여부(is_latest)와 무관).
            if documents or citations:
                _render_references_popover(
                    msg_id,
                    documents,
                    on_page_jump=_handle_page_jump,
                    citations=citations,
                )

                # 성능 지표 (UX-3: 기본 접힘)
                if metrics:
                    total_time = metrics.get("total_time", 0)
                    # 성능 메트릭에 검색 청크 수 키가 없으므로 메시지에 실린 documents
                    # 길이(실제 답변에 사용된 청크 수)를 우선 사용한다.
                    retrieved = (
                        len(documents)
                        if documents
                        else metrics.get("retrieved_chunks", 0)
                    )
                    model = (
                        kwargs.get("model", "")
                        or SessionManager.get("last_selected_model", "")
                        or DEFAULT_OLLAMA_MODEL
                    )
                    with st.expander("Metrics", expanded=False):
                        st.caption(
                            status_line(
                                f"Time: {total_time:.1f}s",
                                f"Retrieved: {retrieved} chunks",
                                f"Model: {model}",
                            )
                        )

        # 상세 사고 과정/답변 프로세스 (LLM reasoning 로그 + 파이프라인 진행
        # 구조) — 답변·출처 아래, 기본 접힘.
        if (
            role == "assistant"
            and msg_type != "streaming"
            and (
                (thought and thought.strip())
                or bool(
                    process
                    and (
                        process.get("steps")
                        or process.get("sections")
                        or process.get("top_scores")
                        or process.get("perf")
                    )
                )
            )
        ):
            with st.expander("Detailed thinking", expanded=False):
                process = process or {}
                steps = process.get("steps") or []
                sections = process.get("sections") or []
                top_scores = process.get("top_scores") or []
                perf = process.get("perf") or {}

                if steps:
                    st.markdown(" · ".join(steps))
                if sections:
                    st.caption(" · ".join(sections))
                if top_scores:
                    st.caption(
                        ", ".join(
                            f"{s['section']} {s['score']:.3f}" for s in top_scores
                        )
                    )
                parts = []
                if isinstance(perf.get("total_time"), (int, float)):
                    parts.append(f"{perf['total_time']:.1f}s")
                if isinstance(perf.get("tps"), (int, float)):
                    parts.append(f"{perf['tps']:.1f} tok/s")
                if perf.get("input_token_count") is not None:
                    parts.append(f"{perf['input_token_count']} tok")
                if parts:
                    st.caption(status_line(*parts))

                if thought and thought.strip():
                    st.markdown("**Thinking process**")
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
    st.chat_message("system").markdown(MSG_CHAT_GUIDE)


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
        cache_tag = " [cached]" if doc_stats.get("cache_used") else " [new]"

    with st.chat_message("system", avatar=AVATARS["document"]):
        if is_building:
            st.caption(status_line(file_name, f"Analyzing...{cache_tag}"))
            # 진행 상황은 메시지 루프에서 build_progress 타입으로 처리
        elif has_error:
            st.caption(status_line(file_name, f"Error{cache_tag}"))
        elif is_ready:
            st.caption(status_line(file_name, f"Ready{cache_tag}"))
        else:
            st.caption(status_line(file_name, f"Waiting...{cache_tag}"))


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
                status_text = msg.get("status", "Processing...")

                # 상태를 생성 시점에 결정한다. (2초 폴링이 매번
                # running 상태로 생성 후 complete로 전환하면 깜빡임이 발생)
                if msg.get("error"):
                    label, state, expanded = "Analysis failed/cancelled", "error", True
                elif progress >= 100 or msg.get("done"):
                    label, state, expanded = "Analysis complete", "complete", False
                else:
                    label, state, expanded = (
                        f"Analyzing document: {status_text}",
                        "running",
                        True,
                    )

                with (
                    st.chat_message("system", avatar=AVATARS["building"]),
                    st.status(label, expanded=expanded, state=state),
                ):
                    st.progress(progress / 100)
                    st.caption(f"{progress}% complete")
                    if state == "running" and msg.get("cancelable", True):
                        st.button(
                            "Cancel Analysis",
                            key=cancel_rebuild_key(current_sid),
                            on_click=_cancel_rebuild,
                            args=(current_sid,),
                            use_container_width=True,
                        )

                    # 진행 로그 표시
                    if msg.get("logs"):
                        with st.expander("Progress log", expanded=False):
                            for log in msg.get("logs", [])[-10:]:
                                st.text(log)
                continue

            elif mtype == "build_error":
                # 빌드 에러
                with st.chat_message("system", avatar=AVATARS["error"]):
                    ui_error(msg.get("error", "Unknown error"))
                continue

            elif mtype == "log":
                # 상태 로그 (작은 캡션으로)
                st.caption(content)
                continue

            # 일반 시스템 메시지 (문서 업로드 알림 등)
            if not doc_rendered and has_doc_context:
                _render_doc_context_inline(current_sid)
                doc_rendered = True

            with st.chat_message("system"):
                st.markdown(content)
            continue

        # READY_FOR_QUERY 같은 내부 메시지는 스킵
        if content == "READY_FOR_QUERY":
            continue

        # 스트리밍 중인 메시지
        if mtype == "streaming":
            # 스트리밍 중 오류가 실린 메시지는 즉시 표면화
            if msg.get("error"):
                with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                    ui_error(str(msg.get("error")))
                continue
            status_text = msg.get("status", "Generating...")
            thought = msg.get("thought", "")
            # 중단 요청 접수 시 즉시 "Stopping..." 피드백 (uiux-fix-p1 INT-2)
            cancel_requested = bool(
                SessionManager.get("generation_cancel", False, current_sid)
            )
            if cancel_requested:
                status_text = "Stopping..."

            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                # 실시간 상태 표시
                with st.status(
                    f"{status_text}", expanded=True, state="running"
                ) as status:
                    if thought:
                        status.write(thought)
                    # ui.components 내부 순환 의존을 피하기 위해 lazy import
                    # (모듈 레벨로 올리지 말 것 — chat.py:456과 동일한 이유).
                    process_steps = msg.get("process_steps", [])
                    if process_steps:
                        from ui.components.streaming import _build_process

                        process = _build_process(msg)
                        if process.get("steps"):
                            status.write(" · ".join(process["steps"]))
                    # 중단 요청 접수 시 "Stopping..." 안내를 즉시 표시한다 (INT-2)
                    if cancel_requested:
                        status.caption("Stopping...")
                    # 빈 박스 방지: thought·단계가 모두 없어도 "Preparing..."
                    # placeholder로 본문이 절대 비어 보이지 않게 한다.
                    elif not thought and not process_steps:
                        status.caption("Preparing...")

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
                        "Stop",
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
            process=msg.get("process"),
            cancelled=msg.get("cancelled", False),
            citations=msg.get("citations"),
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
        return "AI is generating your answer...", True
    if not is_ready:
        return MSG_CHAT_GUIDE, True
    return "Ask a follow-up question...", False


def render_chat_input_area() -> None:
    """Renders the native st.chat_input() at the bottom of the chat column."""
    current_sid = SessionManager.get_session_id()
    input_placeholder, input_disabled = _resolve_chat_input_state(current_sid)

    user_query = st.chat_input(
        input_placeholder, disabled=input_disabled, key=MAIN_CHAT_INPUT_KEY
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
