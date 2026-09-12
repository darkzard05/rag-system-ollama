"""
채팅 인터페이스 컴포넌트 - 통합 타임라인 버전.

모든 이벤트(문서 업로드, 분석 진행, 대화 메시지, 생각 과정)를
단일 연대기순 타임라인으로 렌더링합니다.

구조 (P3 모듈 분리):
- ``chat_references``: 참조 (페이지/doc 점프) 렌더.
- ``chat_build``: 문서 분석 (빌드) 진행 렌더.
- 본 모듈: 메시지/타임라인/스트리밍 렌더 및 위 모듈 재-export.
"""

import html
import logging
import time
import uuid
from typing import Any

import streamlit as st

from common.config import MSG_CHAT_GUIDE
from common.utils import (
    apply_tooltips_to_response,
    normalize_latex_delimiters,
    strip_context_tokens,
)
from core.session import SessionManager
from ui.components.common import AVATARS, status_line, ui_error
from ui.components.streaming import (
    _AUX_STATE_KEY,
    _clear_aux_state,
    _content_generator,
    _finalize_pdf_side_effects,
)
from ui.widget_keys import MAIN_CHAT_INPUT_KEY

logger = logging.getLogger(__name__)

# 본문 하단 "Answer complete" 캡션에 노출할 참조 페이지 미리보기 최대 수.
PREVIEW_PAGES_MAX = 4


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
                    _render_streaming_with_write_stream(
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
    안에서 라이브 렌더 셸(``_run_active_stream_in_timeline``)이 순수 코어
    ``consume_stream_into_message``를 ``on_chunk`` 콜백과 함께 동기 소비하며
    본문을 갱신한다.

    현재 이 함수는 비활성/폴백 브랜치에서 렌더 시점의 msg 스냅샷을 한 번 읽어
    익스팬더("Answer details")를 먼저 그리고 그 아래에 본문을 그린다. 내용이
    없는(■ 중지로 영속을 건너뛴) 플레이스홀더는 [UX-3] 조기 분기에서 중립
    캡션("생성이 중단되었습니다")으로 처리한다.
    """
    # 스트리밍 중 오류가 실린 메시지는 즉시 표면화
    if msg.get("error"):
        st.error(str(msg.get("error")))
        return

    # [UX-3] 내용이 없는(■ 중지로 영속을 건너뛴) 스트리밍 플레이스홀더:
    # "Generating..." 거짓 진행 표시 금지. 중립 문구로 처리하고 진행 expander 생략.
    if not msg.get("content"):
        st.caption("생성이 중단되었습니다 · 표시할 답변이 없습니다")
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

    # 분석 상태 블록은 타임라인 흐름 안에 둔다(별도 배치 시 시작 말풍선과
    # 레이아웃 충돌). 폴링(1.5s)은 빌드/취소 진행 중에만 필요하다 — 진행 중에는
    # 전체 rerun이 없어 진행 바가 고착되기 때문. 그 외 상태는 정적 block이 렌더하므로
    # 유휴 폴링 타이머가 없다 (fix-001-polling-fragments).
    is_building = bool(SessionManager.get("is_building_rag", False, current_sid))
    is_cancelling = bool(SessionManager.get("rebuild_cancelled", False, current_sid))
    if is_building or is_cancelling:
        _render_build_progress_fragment(current_sid)
    else:
        _render_build_progress_block(current_sid)

    # 통합 타임라인 렌더. 스트리밍 메시지도 단일 pass로 직접 렌더하므로
    # 익스팬더 위치가 안정적으로 유지된다.
    _render_unified_timeline(current_sid)


def _resolve_chat_input_state(sid: str) -> tuple[str, bool]:
    """채팅 입력의 placeholder/disabled 상태를 결정하는 순수 함수입니다."""
    is_generating = bool(SessionManager.get("is_generating_answer", False, sid))
    is_ready = SessionManager.is_ready_for_chat(session_id=sid)
    is_swapping = bool(SessionManager.get("is_swapping_model", False, sid))

    if is_generating:
        return "AI가 답변을 생성 중입니다... · ■ 버튼으로 중지할 수 있습니다", False
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

    # [UX-4] 생성 진행 중 시각적 어포던스: 입력창 위 상태 캡션.
    # submit_mode="stop"의 ■ 중지 버튼과 함께 "생성 중"임을 명시한다.
    # (캡션 수명은 자리표시자와 동일 — 다음 rerun에서 갱신/소거)
    if SessionManager.get("is_generating_answer", False, current_sid):
        st.caption(input_placeholder)

    # [UX-3] 실사용 취소 = 네이티브 ■ 중지(submit_mode="stop") — ScriptRunner가
    # StopException(BaseException)을 발생시켜 영속화(아래 :628-645)를 건너뛴다.
    # generation_cancel 플래그는 프로그래매틱 전용(현재 src 호출자 없음)이며
    # 영속 메시지의 cancelled 마커로만 쓰인다.
    user_query = st.chat_input(
        input_placeholder,
        disabled=False,
        key=MAIN_CHAT_INPUT_KEY,
        submit_mode="stop",
    )

    if user_query:
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
            # [Phase2-Oracle] 이전 턴에 잔존한 스테일 취소 플래그 제거: 동기 단일-런
            # 동안 큐에 쌓인 Stop 클릭은 런 종료 후에만 반영되므로, 다음 제출 시
            # 플래그를 항상 초기화해야 새 질문이 오작동 취소되지 않는다.
            SessionManager.set("generation_cancel", False, current_sid)
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
    스트림 소비를 동일 script run에서 함께 수행해 깜빡임을 막는다. 실제
    누적/영속화는 순수 코어 ``consume_stream_into_message``(streaming.py)에
    위임하고, 이 함수는 ``on_chunk`` 콜백으로 받은 스냅샷으로 본문/익스팬더만
    그리는 렌더 전용 셸이다.
    """
    from ui.components.streaming import consume_stream_into_message

    msg_id = msg.get("msg_id", "")
    model_name = SessionManager.get("last_selected_model", session_id=current_sid) or ""

    accumulated = ""
    thought = ""
    documents: list[Any] = []
    metrics: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    process_steps: list[str] = []

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

        def _on_chunk(snapshot: dict[str, Any]) -> None:
            """코어가 청크마다 방출하는 스냅샷으로 라이브 본문/익스팬더를 갱신한다."""
            nonlocal accumulated, thought, documents, metrics, citations, process_steps
            new_accumulated = snapshot["accumulated"]
            new_thought = snapshot["thought"]
            new_documents = snapshot["documents"]
            new_metrics = snapshot["metrics"]
            new_citations = snapshot["citations"]
            new_process_steps = snapshot["process_steps"]

            aux_changed = (
                new_thought != thought
                or new_documents != documents
                or new_metrics != metrics
                or new_citations != citations
                or new_process_steps != process_steps
            )
            accumulated, thought, documents, metrics, citations, process_steps = (
                new_accumulated,
                new_thought,
                new_documents,
                new_metrics,
                new_citations,
                new_process_steps,
            )
            body_ph.markdown(accumulated + " ▌", unsafe_allow_html=False)
            if aux_changed:
                _render_aux()

        # 초기 프레임: 빈 본문이라도 익스팬더를 바로 붙여 순서를 고정.
        _render_aux()
        body_ph.markdown("", unsafe_allow_html=False)
        _persist()

        try:
            consume_stream_into_message(
                current_sid,
                query,
                model_name,
                msg_id=msg_id,
                on_chunk=_on_chunk,
            )
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
            # [FIX-STREAM-ERROR-VISIBLE] 실패를 저장만 하고 삼키면 사용자 화면에는
            # 빈 어시스턴트 버블만 남는다(Momus-APPROVE-WITH-CHANGES 재지향 P0).
            # 위 add_message는 동일 msg_id로 streaming 플레이스홀더를 대체하므로,
            # rerun 1회로 타임라인이 error가 담긴 general 브랜치를 렌더하게 한다.
            # is_generating_answer는 이미 False라 rerun 후 재진입(무한 루프)하지 않는다.
            st.rerun()

    # 스트림 정상 완료: 코어가 이미 최종 general 메시지 + 플래그 클리어를
    # 수행했으므로(중복 add_message 금지) 라이브 익스팬더를 최종 메타데이터로
    # 갱신한 뒤 rerun 1회로 전환을 확정한다. 이 rerun은 타임라인이 general
    # 브랜치를 타며 입력창도 is_generating_answer=False에 맞춰 정상 활성화된다.
    _render_aux()
    st.rerun()


def _render_streaming_with_write_stream(
    msg: dict[str, Any], current_sid: str, query: str
) -> None:
    """st.write_stream 기반 스트리밍 렌더 (비블로킹).

    텍스트는 ``st.write_stream``으로 표시하고, 부가 정보(thought/metrics/
    citations)는 완료 후 렌더링한다. ``submit_mode="stop"``과 호환되어
    사용자가 중지 시 부분 응답이 표시되고 영속화된다.
    """
    msg_id = msg.get("msg_id", "")
    model_name = SessionManager.get("last_selected_model", session_id=current_sid) or ""

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        # [UX-1/UX-2] 프리토큰/생성 중 라이브 상태 캡션.
        # write_stream과 동일 script run 안에서 _content_generator의 on_status
        # 콜백으로 갱신된다 (구 렌더러 aux_ph/body_ph 패턴의 후속 — 폴링 없음).
        status_ph = st.empty()
        status_ph.caption("AI가 답변을 생성 중입니다... ▍")

        def _on_status(status_text: str, elapsed_sec: float) -> None:
            status_ph.caption(f"{status_text} · {elapsed_sec:.0f}s ▍")

        # st.write_stream — 스트리밍 텍스트 표시
        response = None
        try:
            response = st.write_stream(
                _content_generator(
                    query,
                    model_name,
                    current_sid,
                    msg_id,
                    on_status=_on_status,
                ),
            )
        except Exception as exc:
            response = f"오류가 발생했습니다: {exc}"
        finally:
            # 완료/예외/■ 중지(StopException — BaseException) 모두에서 실행되어
            # 잘못된 "생성 중" 잔상이 남지 않는다. 플레이스홀더는 매 run의 일시 요소.
            status_ph.empty()

        # aux state 읽기 후 즉시 정리 (잔여 데이터 방지)
        aux_state = SessionManager.get(_AUX_STATE_KEY, {}, current_sid) or {}
        _clear_aux_state(current_sid)

        # expander는 chat_message 블록 안에서 렌더
        render_generation_expander(
            {
                "thought": aux_state.get("thought", ""),
                "documents": aux_state.get("documents", []),
                "citations": aux_state.get("citations", []),
                "metrics": aux_state.get("metrics", {}),
                "model": model_name,
                "process_steps": aux_state.get("process_steps", [])[-10:],
                "cancelled": bool(
                    SessionManager.get("generation_cancel", False, current_sid)
                ),
                "msg_id": msg_id,
            },
            expanded=False,
            generating=False,
        )

    # chat_message 블록 바깥 — 메시지 영속화
    cancelled = bool(SessionManager.get("generation_cancel", False, current_sid))
    error_text = aux_state.get("error")
    SessionManager.add_message(
        "assistant",
        str(response) if response else "",
        msg_type="general",
        msg_id=msg_id,
        thought=aux_state.get("thought", ""),
        documents=aux_state.get("documents", []),
        metrics=aux_state.get("metrics", {}),
        citations=aux_state.get("citations", []),
        process_steps=aux_state.get("process_steps", [])[-10:],
        processed_content=None,
        cancelled=cancelled,
        error=_friendly_stream_error(Exception(error_text)) if error_text else None,
        session_id=current_sid,
    )
    SessionManager.set("is_generating_answer", False, current_sid=current_sid)
    SessionManager.set("generation_cancel", False, current_sid=current_sid)
    _finalize_pdf_side_effects(current_sid, msg_id)


# isort: off
from ui.components.chat_references import (  # noqa: E402
    _extract_reference_pages,
    _handle_doc_jump,
    _handle_page_jump,
    _render_references_content,
    render_generation_expander,
)
from ui.components.chat_build import (  # noqa: E402
    _cancel_rebuild,
    _render_build_progress_block,
    _render_build_progress_fragment,
    _render_doc_context_inline,
    _render_guidance_panel,
)
# isort: on

# P3 분리된 모듈의 재-export 목록: 본 모듈의 공개 API로 유지한다
# (F401 의도적 무시 — 내부 호출자는 본문에서 재-export된 이름을 사용).
__all__ = [
    "_cancel_rebuild",
    "_extract_reference_pages",
    "_friendly_stream_error",
    "_handle_doc_jump",
    "_handle_page_jump",
    "_render_build_progress_block",
    "_render_build_progress_fragment",
    "_render_doc_context_inline",
    "_render_guidance_panel",
    "_render_references_content",
    "_render_unified_timeline",
    "_resolve_chat_input_state",
    "render_chat_input_area",
    "render_chat_messages_area",
    "render_generation_expander",
    "render_message",
]
