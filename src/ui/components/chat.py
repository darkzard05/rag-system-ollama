"""
채팅 인터페이스 컴포넌트 - 통합 타임라인 버전.

모든 이벤트(문서 업로드, 분석 진행, 대화 메시지, 생각 과정)를
단일 연대기순 타임라인으로 렌더링합니다.
"""

import contextlib
import html
import logging
import time
import uuid
from collections.abc import Callable
from typing import Any

import streamlit as st

from common.config import DEFAULT_OLLAMA_MODEL, MSG_CHAT_GUIDE
from common.utils import (
    apply_tooltips_to_response,
    fast_hash,
    normalize_latex_delimiters,
    strip_context_tokens,
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

# 본문 하단 "Answer complete" 캡션에 노출할 참조 페이지 미리보기 최대 수.
PREVIEW_PAGES_MAX = 4


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


def _render_references_content(
    msg_id: str,
    documents: list[Any] | None,
    on_page_jump: Callable[[int], None] | None = None,
    citations: list[dict[str, Any]] | None = None,
    generating: bool = False,
) -> bool:
    """참조 콘텐츠(페이지/doc 점프 버튼)를 렌더링합니다.

    통합 익스팬더 내부에서 직접 호출되므로 popover 래퍼 없이 본문만 그립니다.
    렌더된 참조가 있으면 True, 없으면 False를 반환합니다.

    generating=True(스트리밍 중)에는 매 chunk rerun마다 동일 위젯이 재생성되므로
    key를 소비하는 st.button 대신 정적 markdown으로 페이지/doc을 표시합니다.
    key가 필요한 상호작용 점프 버튼은 완료 후(generating=False, 단일 rerun)에만
    렌더하므로 StreamlitDuplicateElementKey 충돌을 피합니다.
    """
    rendered = False
    if not documents and not citations:
        return rendered

    pages = _extract_reference_pages(documents or [])
    if pages:
        st.caption("By page")
        if generating:
            # 스트리밍 중: key 없는 정적 표시 (중복 등록 방지).
            st.markdown(" · ".join(f"`{p}p`" for p in pages))
        else:
            cols = st.columns(min(len(pages), 5))
            for idx, p in enumerate(pages):
                clicked = cols[idx % len(cols)].button(
                    f"{p}p",
                    key=jump_key(msg_id, p, idx),
                    use_container_width=True,
                )
                if clicked and on_page_jump is not None:
                    on_page_jump(p)
        rendered = True

    # P3: citations[] 기반 doc 점프 (안정 doc_id).
    doc_citations = [c for c in (citations or []) if c.get("doc_id") is not None]
    if doc_citations:
        doc_ids = {_doc_stable_id(d) for d in (documents or [])}
        st.caption("By doc")
        for idx, cit in enumerate(doc_citations):
            sid = str(cit.get("doc_id"))
            if sid in doc_ids:
                label = cit.get("section") or cit.get("text_span") or f"doc {sid}"
                if generating:
                    # 스트리밍 중: key 없는 정적 표시.
                    st.markdown(
                        f'{idx + 1}. <span data-doc-id="{html.escape(sid)}">'
                        f"{html.escape(label)}</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    if st.button(
                        f"{idx + 1}. {label}",
                        key=f"pop_doc_{msg_id}_{sid}_{idx}",
                        use_container_width=True,
                    ):
                        _handle_doc_jump(sid)
        rendered = True
    return rendered


def render_generation_expander(
    msg: dict[str, Any],
    *,
    expanded: bool,
    generating: bool,
    status_text: str = "Answer generation",
    process_override: dict[str, Any] | None = None,
) -> None:
    """답변 말풍선 상단(질문↔답변 사이)에 **단일 고정 익스팬더**를 렌더합니다.

    메트릭·생성 단계·상위 점수·사고 과정·참조를 모두 이 익스팬더 안에 수납해
    산개되던 부가 정보(별도 Metrics 익스팬더, References popover, 완료 후
    generation 익스팬더)를 하나로 통합한다. 기본값은 접힘(expanded=False).

    스트리밍 중과 완료 후 동일 위젯을 재사용해, 생성 완료 시 상태 박스가
    증발하던 문제를 해결한다. 본문은 매 렌더 **무조건** 작성하므로 fragment
    폴링(0.5s)으로 st.expander가 재생성되어도 내용이 비지 않는다.

    - generating=True: 기본 접힘 유지, 내부 st.spinner로 진행 표시
    - generating=False: 접은 상태(완료 후 유지), 정적 헤더만
    - cancelled 메시지는 추론 로그를 감춰 전체 추론 완료로 오인되지 않게 함
    - process_override: 완료 메시지처럼 이미 계산된 process dict가 있으면
      재파생(process_steps 의존) 대신 직접 사용한다.
    """
    thought = msg.get("thought", "") or ""
    cancelled = bool(msg.get("cancelled", False))
    show_thought = bool(thought and thought.strip() and not cancelled)
    documents = msg.get("documents") or []
    citations = msg.get("citations") or []
    metrics = msg.get("metrics") or {}
    msg_id = msg.get("msg_id") or ""

    # ui.components 내부 순환 의존을 피하기 위해 lazy import
    from ui.components.streaming import _build_process

    process = process_override or _build_process(msg) or {}
    steps = process.get("steps") or []
    sections = process.get("sections") or []
    top_scores = [
        s
        for s in (process.get("top_scores") or [])
        if isinstance(s, dict) and "section" in s and "score" in s
    ]
    perf = process.get("perf") or {}

    # 메트릭(완료 메시지에 실린 metrics)도 익스팬더 수납 대상.
    retrieved = (
        len(documents) if documents else (process or {}).get("retrieved_count", 0)
    )
    total_time = metrics.get("total_time", 0)
    has_metrics = bool(total_time or retrieved)
    has_block = bool(
        steps or sections or top_scores or perf or show_thought or has_metrics
    )

    # 완료 메시지인데 표시할 내용이 없으면 익스팬더 자체를 렌더하지 않는다.
    # 빈 익스팬더 헤더가 대화 줄 간격(패딩+익스팬더)을 키워 간격 과대를 유발한다.
    # 생성 중(generating=True)에는 항상 익스팬더를 열어 진행 표시/깜빡임을 방지한다.
    if not generating and not (has_block or documents or citations):
        return

    with st.expander("Answer details", expanded=expanded):
        if generating:
            with st.spinner(status_text):
                pass  # spinner는 헤더 아래 진행 표시용(본문은 아래 즉시 작성)

        if not (has_block or documents or citations):
            # 생성 중인데 아직 표시할 내용이 없으면 진행 캡션만 노출.
            st.caption("Preparing...")
            return

        if steps:
            st.markdown(" · ".join(steps))
        if sections:
            st.caption(" · ".join(sections))
        if top_scores:
            st.caption(
                ", ".join(f"{s['section']} {s['score']:.3f}" for s in top_scores)
            )

        # 메트릭: Time / Retrieved / Model (UX-3: 기본 접힘 익스팬더 내 수납).
        parts = []
        if isinstance(total_time, (int, float)):
            parts.append(f"Time: {total_time:.1f}s")
        if retrieved:
            parts.append(f"Retrieved: {retrieved} chunks")
        model = (
            msg.get("model", "")
            or SessionManager.get("last_selected_model", "")
            or DEFAULT_OLLAMA_MODEL
        )
        if model:
            parts.append(f"Model: {model}")
        if parts:
            st.caption(status_line(*parts))

        if show_thought:
            st.markdown("**Thinking process**")
            st.markdown(thought)

        # 참조(페이지/doc 점프) — 기존 References popover 내용을 익스팬더 안으로 통합.
        if documents or citations:
            st.divider()
            st.caption("References")
            _render_references_content(
                msg_id,
                documents,
                on_page_jump=_handle_page_jump,
                citations=citations,
                generating=generating,
            )


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
    process_steps: list[str] | None = None,
    **kwargs,
) -> None:
    """메시지를 렌더링하는 통합 엔진.

    신뢰 경계: `processed_content`는 호출자가 이미 HTML 이스케이프한 안전한
    마크다운/HTML만 전달해야 한다(예: streaming.py의 생산 경로는 content를
    html.escape 후 apply_tooltips_to_response로 <span>을 주입). 이 인자는
    unsafe_allow_html=True로 렌더되므로 원시 LLM 출력을 그대로 넘기지 않는다.
    원시 텍스트는 `content`로 전달하면 본문 경로에서 자동 이스케이프된다.
    """
    avatar_icon = AVATARS["assistant"] if role == "assistant" else AVATARS["user"]
    msg_id = kwargs.get("msg_id", f"msg_{msg_index}")
    citations = citations or kwargs.get("citations")
    cancelled = bool(kwargs.get("cancelled", False))
    process_steps = process_steps or kwargs.get("process_steps") or []

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
                    if role == "assistant":
                        display_text = strip_context_tokens(display_text)
                    st.markdown(display_text, unsafe_allow_html=(role == "assistant"))
            ui_error(f"Error: {error}")
            return

        # 통합 익스팬더(메트릭·단계·사고·참조) — 질문↔답변 사이(답변 말풍선 상단) 고정.
        # 생성중(generating=True) 슬롯 경로와 완료 후 타임라인 경로가 동일 위젯을
        # 그려 위치 점프를 제거한다.
        if role == "assistant":
            render_generation_expander(
                {
                    "thought": thought,
                    "documents": documents or [],
                    "citations": citations,
                    "metrics": metrics,
                    "model": kwargs.get("model", ""),
                    "process_steps": process_steps,
                    "cancelled": cancelled,
                    "msg_id": msg_id,
                },
                expanded=False,
                generating=False,
                process_override=process or None,
            )

        # 본문 내용
        if processed_content:
            st.markdown(processed_content, unsafe_allow_html=True)
        else:
            display_text = content
            if role == "assistant":
                display_text = html.escape(display_text)

            display_text = normalize_latex_delimiters(display_text)
            # [F5] 본문에 컨텍스트 메타토큰([doc:..] [score:..] 등)이 노출되지 않도록 제거
            if role == "assistant":
                display_text = strip_context_tokens(display_text)
            if role == "assistant" and documents:
                # RC-A: 완료된 메시지 본문은 매 script run마다 변하지 않으므로
                # tooltips 주입 결과(문서 스캔 O(docs))를 msg_id로 캐시해 재수행을
                # 제거한다. st.markdown 호출은 매 run 그대로 실행되므로 위젯/참조
                # popover 재렌더는 보장된다. content/documents/citations가 동일하면 재사용.
                _tip_key = f"_msg_render_{msg_id}"
                _tip_cached = st.session_state.get(_tip_key)
                _tip_sig = (content, id(documents), id(citations))
                if _tip_cached is not None and _tip_cached[0] == _tip_sig:
                    display_text = _tip_cached[1]
                else:
                    display_text = apply_tooltips_to_response(
                        display_text, documents, citations=citations
                    )
                    st.session_state[_tip_key] = (_tip_sig, display_text)
            st.markdown(display_text, unsafe_allow_html=(role == "assistant"))

        # 완료된 어시스턴트 메시지의 하단 상태줄 (기본 노출, 부가 정보는 상단 익스팬더로 통합).
        if (
            role == "assistant"
            and msg_type == "general"
            and (content or processed_content or "").strip()
        ):
            if cancelled:
                st.caption("Stopped · Partial answer preserved")
            elif documents:
                pages = _extract_reference_pages(documents)
                page_txt = status_line(*(f"p.{p}" for p in pages[:PREVIEW_PAGES_MAX]))
                if len(pages) > PREVIEW_PAGES_MAX:
                    page_txt += f" +{len(pages) - PREVIEW_PAGES_MAX} more"
                st.caption(
                    status_line(
                        "Answer complete",
                        f"{len(documents)} references",
                        page_txt,
                    )
                )
            else:
                st.caption("Answer complete")


def _cancel_rebuild(sid: str) -> None:
    """문서 분석 재구축 취소 요청 콜백입니다."""
    SessionManager.set("rebuild_cancelled", True, session_id=sid)
    st.rerun()


def _render_build_progress_block(sid: str) -> None:
    """문서 분석 상태 블록을 단독 렌더합니다 (전용 폴링 fragment가 호출).

    리팩터링에서 타임라인 폴링이 제거된 뒤, 빌드 도중에는 전체 rerun이
    발생하지 않아 ``st.progress`` 가 0%에 고착되던 결함(진행 바 동결)을
    해결하기 위해 분리했다. ``_report_progress``(main.py)가 갱신하는
    ``rebuild_progress`` 상태만 읽어 주기적(``run_every``)으로 다시 그린다.

    분석 블록은 대화의 일부로 영구 잔존한다(빌드 완료/취소/에러 후에도 남아
    타임라인 기록으로 남는다). 빌드가 한 번도 시작되지 않은 초기 상태에서만
    렌더하지 않는다.
    """
    is_building = bool(SessionManager.get("is_building_rag", False, sid))
    is_cancelling = bool(SessionManager.get("rebuild_cancelled", False, sid))
    is_done = bool(SessionManager.get("rebuild_done", False, sid))
    has_doc = bool(SessionManager.get("last_uploaded_file_name", "", sid))

    # 빌드가 한 번도 시작되지 않은 초기 상태(업로드 전)에서는 노출하지 않는다.
    if not (is_building or is_cancelling or is_done or has_doc):
        return

    progress = int(SessionManager.get("rebuild_progress", 0, sid))
    status_text = str(SessionManager.get("rebuild_status", "", sid) or "")
    error = SessionManager.get("pdf_processing_error", "", sid) or ""

    if error:
        label, state, expanded = "Analysis failed/cancelled", "error", True
    elif progress >= 100 or is_done:
        label, state, expanded = "Analysis complete", "complete", False
    elif is_cancelling:
        label, state, expanded = "Cancelling analysis...", "running", True
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
        if state == "running" and not is_cancelling:
            st.button(
                "Cancel Analysis",
                key=cancel_rebuild_key(sid),
                on_click=_cancel_rebuild,
                args=(sid,),
                use_container_width=True,
            )


@st.fragment(run_every=1.5)
def _render_build_progress_fragment(sid: str) -> None:
    """빌드 상태 블록 전용 폴링 fragment.

    전체 rerun 없이 1.5초마다 ``rebuild_progress`` 를 다시 읽어 진행 바를
    갱신한다(타임라인 폴링이 제거된 빈틈을 메움). 빌드 완료 후에는
    ``run_in_background_worker._on_complete`` 의 rerun이 최종 100%를 확정한다.
    완료/취소/에러 상태에서도 블록은 그대로 남아 대화 기록으로 잔존한다.
    """
    _render_build_progress_block(sid)


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


def _render_unified_timeline(current_sid: str) -> None:
    """
    통합 타임라인 렌더링 (단일 패스).
    메시지 리스트에 저장된 모든 타입의 메시지를 시간 순서대로 렌더링.

    타임라인은 ``render_chat_messages_area`` 호출 시점(전체 rerun)에만 갱신되며,
    스트리밍 중 토큰 갱신은 submit 핸들러의 ``st.empty()`` 플레이스홀더가 담당해
    전체 컬럼 재렌더(깜빡임)를 유발하지 않는다. 빌드 진행 표시(build_progress)는
    타임라인이 아닌 전용 폴링 fragment(``_render_build_progress_fragment``)가
    담당한다(전체 rerun 없이 1.5초마다 갱신).
    """
    _t_tl = time.perf_counter()
    messages = SessionManager.get_messages() or []
    n_msgs = len(messages)

    # 빈 대화일 때: 문서 컨텍스트가 있으면 표시, 없으면 가이드
    if not messages:
        if SessionManager.get("last_uploaded_file_name", "", current_sid):
            # 문서가 있으면 분석 블록(전용 폴링 fragment)이 이미 별도로
            # 렌더되므로 시작 가이드 패널("Upload a PDF...")은 노출하지 않는다.
            # (빌드 진행/완료 블록과 중복되는 것을 방지)
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
                # 빌드 진행 표시는 전용 폴링 fragment
                # (_render_build_progress_fragment)가 담당하므로 타임라인에서
                # 제외한다(빌드 도중 전체 rerun 없이도 갱신됨).
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

        # 스트리밍 중인 메시지: 단일 pass로 직접 렌더.
        # 슬롯(st.empty)을 쓰지 않고 매 렌더 msg 딕셔너리를 통째로 다시 그리므로
        # 익스팬더가 항상 본문 위에 고정된다.
        if mtype == "streaming":
            # 활성 스트리밍 턴(입력창 아래 아닌 대화 안에 렌더)은 이 타임라인
            # 브랜치에서 라이브 렌더와 스트림 소비를 함께 수행한다.
            is_active = bool(
                SessionManager.get("is_generating_answer", False, current_sid)
                and SessionManager.get("active_stream_msg_id", "", current_sid)
                == msg.get("msg_id")
            )
            if is_active:
                # [DEFENSE-IN-DEPTH] 핵심 타임아웃 회수(watchdog)와 병행하는
                # best-effort 보강: 스트림 소비 경로가 L922/938 리셋에 도달하기
                # 전에 비정상 종료(setup 예외 등)하면 플래그가 True로 고착된다.
                # 정상/오류 경로는 이미 False로 리셋하므로 가드로 no-op 처리되고,
                # 비정상 탈출 시에만 강제 리셋해 입력창 고착을 막는다.
                try:
                    _run_active_stream_in_timeline(
                        msg, current_sid, query=msg.get("query", "")
                    )
                finally:
                    if SessionManager.get("is_generating_answer", False, current_sid):
                        SessionManager.set(
                            "is_generating_answer", False, current_sid=current_sid
                        )
                continue
            with st.chat_message("assistant", avatar=AVATARS["assistant"]):
                _draw_streaming_message(msg, current_sid)
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
            process_steps=msg.get("process_steps"),
        )

    logger.debug(
        "[PERF] _render_unified_timeline: rendered %d msg(s) in %.3fs",
        n_msgs,
        time.perf_counter() - _t_tl,
    )


# ---------------------------------------------------------------------------
# 스트리밍 전용 렌더 (단일 pass, 폴링 없음)
# ---------------------------------------------------------------------------


def _draw_streaming_message(msg: dict[str, Any], current_sid: str) -> None:
    """스트리밍 메시지를 그립니다 (단일 pass 렌더).

    표준 리팩터 이후 스트리밍은 별도 스레드/fragment 폴링 없이 단일 script run
    안에서 ``_run_standard_streaming_turn``이 ``stream_chunks``를 동기 소비하며
    본문을 갱신한다. 따라서 이 함수는 렌더 시점에 msg 딕셔너리를 한 번 읽어
    익스팬더("Answer details")를 먼저 그리고 그 아래에 본문을 그린다.
    """
    # 스트리밍 중 오류가 실린 메시지는 즉시 표면화
    if msg.get("error"):
        st.error(str(msg.get("error")))
        return

    status_text = msg.get("status", "Generating...")
    cancel_requested = bool(SessionManager.get("generation_cancel", False, current_sid))
    if cancel_requested:
        status_text = "Stopping..."

    # 실시간 상태 표시 — 영속 익스팬더(본문 위에 고정, 생성 완료 후에도 유지).
    render_generation_expander(
        msg, expanded=False, generating=True, status_text=status_text
    )

    # 스트리밍 내용 표시 (커서 포함).
    raw_content = msg.get("content", "")
    msg_id = msg.get("msg_id", "")
    cache_key = f"_stream_html_{msg_id}"
    if raw_content:
        cached = st.session_state.get(cache_key)
        if not cached or cached["raw"] != raw_content:
            processed = normalize_latex_delimiters(html.escape(raw_content))
            st.session_state[cache_key] = {
                "raw": raw_content,
                "html": processed,
            }
        else:
            processed = cached["html"]
        st.markdown(processed + " ▌", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 메인 렌더링
# ---------------------------------------------------------------------------


def render_chat_messages_area() -> None:
    """Renders the chat column: unified timeline (streaming included)."""
    current_sid = SessionManager.get_session_id()

    # 분석 상태 블록을 타임라인 흐름 안에 둔다(별도 위치 배치 시 시작 메시지
    # 말풍선이 블록 아래로 밀려 흐리게 노출되는 레이아웃 충돌을 방지).
    _render_build_progress_fragment(current_sid)

    # 통합 타임라인 렌더. 스트리밍 메시지도 단일 pass로 직접 렌더하므로
    # 익스팬더 위치가 안정적으로 유지된다.
    _render_unified_timeline(current_sid)


def _resolve_chat_input_state(sid: str) -> tuple[str, bool]:
    """채팅 입력의 placeholder/disabled 상태를 결정하는 순수 함수입니다."""
    is_generating = bool(SessionManager.get("is_generating_answer", False, sid))
    is_ready = SessionManager.is_ready_for_chat(session_id=sid)
    is_swapping = bool(SessionManager.get("is_swapping_model", False, sid))

    if is_generating:
        return "AI is generating your answer...", True
    if is_swapping:
        return "Switching models, please wait...", True
    if not is_ready:
        return MSG_CHAT_GUIDE, True
    return "Ask a follow-up question...", False


def render_chat_input_area() -> None:
    """Renders the native st.chat_input() at the bottom of the chat column.

    입력창 영역에는 폴링 fragment를 쓰지 않는다. disabled 상태는
    ``_resolve_chat_input_state``가 ``is_generating_answer`` 플래그로 결정하며,
    생성 완료/예외 시 submit 핸들러가 ``st.rerun()`` 1회로 입력창을 정상
    활성화한다(INT-입력동결 방지). 빌드 진행 바는 이 영역이 아닌 별도의
    ``_render_build_progress_fragment``(1.5초 폴링)가 담당한다.
    """
    current_sid = SessionManager.get_session_id()

    # 생성 중에도 위젯을 disabled로 계속 렌더(입력창 소실 방지).
    input_placeholder, input_disabled = _resolve_chat_input_state(current_sid)

    user_query = st.chat_input(
        input_placeholder, disabled=input_disabled, key=MAIN_CHAT_INPUT_KEY
    )

    if user_query and not input_disabled:
        query_text = user_query.strip()
        if query_text:
            _t_submit = time.perf_counter()
            SessionManager.add_message("user", query_text, session_id=current_sid)

            # [FIX-ORDER] 스트리밍 버블을 입력창 아래(분리)가 아닌 대화 타임라인
            # 안(질문 바로 아래)에 그리려면, 스트리밍 루프를 입력 영역에서 직접
            # 돌리지 않는다. 대신 `streaming` 타입 플레이스홀더를 추가하고 플래그를
            # 세운 뒤 1회 rerun. 이후 타임라인의 `mtype=="streaming"` 브랜치가
            # 동일 script run에서 라이브 렌더 + 스트림 소비를 함께 수행한다
            # (입력 영역은 DOM상 메시지 스크롤 컨테이너보다 뒤에 방출되므로,
            #  여기서 렌더하면 입력창 아래에 붙는 원인이었다).
            stream_msg_id = str(uuid.uuid4())
            SessionManager.add_message(
                "assistant",
                "",
                msg_type="streaming",
                msg_id=stream_msg_id,
                query=query_text,
                thought="",
                documents=[],
                metrics={},
                citations=[],
                processed_content=None,
                session_id=current_sid,
            )
            SessionManager.set("is_generating_answer", True, current_sid)
            SessionManager.set("active_stream_msg_id", stream_msg_id, current_sid)
            logger.debug(
                "[PERF] submit handler: setup took %.3fs (before st.rerun)",
                time.perf_counter() - _t_submit,
            )
            # 타임라인이 라이브 스트리밍을 렌더하도록 명시적 rerun 1회.
            st.rerun()


def _friendly_stream_error(exc: Exception) -> str:
    """원시 예외를 사용자 친화 메시지로 매핑합니다 (lazy import로 순환 의존 회피)."""
    from ui.components.streaming import friendly_error_message

    return friendly_error_message(exc)


def _run_active_stream_in_timeline(
    msg: dict[str, Any], current_sid: str, query: str
) -> None:
    """활성 스트리밍 메시지를 대화 타임라인 안에서 라이브 렌더한다.

    입력 영역(입력창 아래)이 아니라 메시지 스크롤 컨테이너 내부에 렌더하므로
    질문 → "Answer details" 익스팬더 → 스트리밍 본문 순서가 보장된다. 렌더와
    스트림 소비를 동일 script run에서 함께 수행해 깜빡임을 막는다.
    """
    from ui.components.streaming import stream_chunks

    msg_id = msg.get("msg_id", "")
    model_name = SessionManager.get("last_selected_model", session_id=current_sid) or ""

    accumulated = ""
    thought = ""
    documents: list[Any] = []
    metrics: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    process_steps: list[str] = []
    _raw_json_parts: list[str] = []
    _fa_scan_pos = 0

    # 라이브 렌더 컨테이너: 매 chunk 본문/익스팬더를 갱신.
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        aux_ph = st.empty()  # 부가 정보(thought/docs/metrics) 고정 슬롯
        body_ph = st.empty()  # 본문 고정 슬롯

        def _render_aux() -> None:
            aux_ph.empty()
            with aux_ph:
                render_generation_expander(
                    {
                        "thought": thought,
                        "documents": documents or [],
                        "citations": citations,
                        "metrics": metrics,
                        "model": model_name,
                        "process_steps": process_steps[-10:],
                        "cancelled": False,
                        "msg_id": msg_id,
                    },
                    expanded=False,
                    generating=True,
                )

        def _persist() -> None:
            SessionManager.add_message(
                "assistant",
                accumulated,
                msg_type="streaming",
                msg_id=msg_id,
                thought=thought,
                documents=documents,
                metrics=metrics,
                citations=citations,
                processed_content=None,
                session_id=current_sid,
            )

        # 초기 프레임: 빈 본문이라도 익스팬더를 바로 붙여 순서를 고정.
        _render_aux()
        body_ph.markdown("", unsafe_allow_html=False)
        _persist()

        try:
            for chunk in stream_chunks(query, model_name, current_sid):
                if chunk.content:
                    if getattr(chunk, "raw_json", False):
                        from ui.components.streaming import (
                            _extract_final_answer_delta,
                        )

                        _raw_json_parts.append(chunk.content)
                        blob = "".join(_raw_json_parts)
                        delta, _fa_scan_pos = _extract_final_answer_delta(
                            blob, _fa_scan_pos
                        )
                        accumulated += delta
                    else:
                        accumulated += chunk.content
                    body_ph.markdown(accumulated + " ▌", unsafe_allow_html=False)
                if chunk.thought:
                    thought += chunk.thought
                if chunk.status:
                    step = chunk.status
                    if not process_steps or process_steps[-1] != step:
                        process_steps.append(step)
                _meta = chunk.metadata or {}
                if _meta.get("documents"):
                    documents = _meta["documents"]
                if chunk.performance:
                    metrics = chunk.performance
                if getattr(chunk, "citations", None):
                    citations = chunk.citations or []
                if (
                    chunk.thought
                    or _meta.get("documents")
                    or chunk.performance
                    or getattr(chunk, "citations", None)
                ):
                    _render_aux()
                _persist()
            # 루프 정상 종료: 마지막 청크 이후에야 확정되는 메타데이터(문서/측정값/생각)가
            # 있으면 누락 없이 익스팬더에 반영한다. (스트림 consumer는 metadata를 늦게
            # 내보내는 경우가 있어, 루프 중 마지막 _render_aux 호출만으로는 불완전할 수 있음)
            _render_aux()
        except Exception as exc:  # noqa: BLE001 - 스트림 레벨 오류를 사용자에게 노출
            logger.exception("[CHAT] 스트리밍 중 오류: %s", exc)
            SessionManager.set("is_generating_answer", False, current_sid=current_sid)
            SessionManager.add_message(
                "assistant",
                accumulated or "",
                msg_type="general",
                msg_id=msg_id,
                thought=thought,
                documents=documents,
                metrics=metrics,
                citations=citations,
                error=_friendly_stream_error(exc),
                session_id=current_sid,
            )
            return

    # 스트리밍 정상 완료: 최종 스냅샷을 ``general``로 확정(폴링 대신 명시적 저장).
    SessionManager.set("is_generating_answer", False, current_sid=current_sid)
    # [FIX-STREAM-COMPLETE] 활성 run이 끝나면 이 run의 프레임이 화면에 고착된다.
    # 그 프레임은 아직 `msg_type="streaming"` 확정 전(또는 활성 disabled 입력창) 상태라,
    # "Answer details" 익스팬더(측정값/문서/생각)가 반영되지 않은 채 커서(▌)만 남고,
    # 입력창도 disabled로 굳어 다음 질문 때까지 갱신되지 않는다(실측 재현 확인).
    # 폴링 fragment 제거 이후 결정론적 전환은 명시적 rerun 1회로 보장한다.
    # 이 rerun으로 타임라인이 `general` 브랜치를 타며 익스팬더를 즉시 렌더하고,
    # 입력창도 `is_generating_answer=False`에 맞춰 정상 활성화된다.
    SessionManager.add_message(
        "assistant",
        accumulated,
        msg_type="general",
        msg_id=msg_id,
        thought=thought,
        documents=documents,
        metrics=metrics,
        citations=citations,
        process_steps=process_steps[-10:],
        processed_content=None,
        session_id=current_sid,
    )
    st.rerun()
