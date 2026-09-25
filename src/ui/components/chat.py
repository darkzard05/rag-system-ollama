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
from ui.strings import get_phase_labels, t
from ui.widget_keys import MAIN_CHAT_INPUT_KEY, SAMPLE_QUESTION_STATE_KEY

logger = logging.getLogger(__name__)


def _metrics_caption(metrics: dict | None, model: str | None = None) -> str:
    """완료 캡션(답변 아래)용 핵심 메트릭 문자열 생성 (중복 모델명 제외)."""
    metrics = metrics or {}
    parts: list[str] = []
    total_time = metrics.get("total_time", 0)
    if isinstance(total_time, (int, float)) and total_time > 0:
        parts.append(f"{total_time:.1f}s")
    input_tokens = metrics.get("input_token_count", 0)
    output_tokens = metrics.get("token_count", 0)
    if input_tokens or output_tokens:
        parts.append(f"{int(input_tokens or 0)}→{int(output_tokens or 0)} tok")
    return status_line(*parts)


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

        # 통합 익스팬더(사고·참조) — 질문↔답변 사이(답변 말풍선 상단) 고정.
        # [개선 5] thought나 documents, citations 중 하나라도 존재할 때만 렌더링 호출
        if role == "assistant" and (thought or documents or citations):
            should_expand = is_latest and bool(thought or documents)
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
                expanded=should_expand,
                generating=False,
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

        # 완료된 어시스턴트 메시지의 하단 상태줄 (중복 상태 문구 제거 및 핵심 지표만 노출)
        metric_txt = _metrics_caption(metrics, kwargs.get("model"))
        if (
            role == "assistant"
            and msg_type == "general"
            and (content or processed_content or "").strip()
        ):
            if cancelled:
                st.caption(
                    status_line("Stopped · Partial answer preserved", metric_txt)
                )
            elif metric_txt:
                st.caption(metric_txt)


def _render_unified_timeline(current_sid: str) -> None:
    _t_tl = time.perf_counter()
    messages = SessionManager.get_messages() or []
    n_msgs = len(messages)

    # 실제 대화 메시지(사용자/어시스턴트) 추출
    chat_messages = [m for m in messages if m.get("msg_type") != "build_progress"]

    # 실제 대화가 아직 시작되지 않은 경우:
    if not chat_messages:
        # 온보딩은 _render_guidance_panel 단일 소스에 위임한다 (파일 게이트는
        # 패널 내부에 일원화 — 여기서 중복 검사하지 않는다).
        _render_guidance_panel()
        # 문서 분석 완료 시 상단 status 블록("준비 완료")과 하단 입력창 placeholder가
        # 상태 안내 및 질문 유도를 전담하므로, 증발하는 중복 더미 말풍선은 렌더링하지 않고 종료
        return

    # 대화가 시작된 이후에는 messages 목록에 있는 내용들이 순서대로 출력됨
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        mtype = msg.get("msg_type", "general")
        is_latest = i == len(messages) - 1

        # 시스템/로그 메시지 처리
        if role == "system":
            if mtype == "build_progress":
                continue

            elif mtype == "build_error":
                with st.chat_message("system", avatar=AVATARS["error"]):
                    ui_error(msg.get("error", "Unknown error"))
                continue

            elif mtype == "log":
                st.caption(content)
                continue

            # [개선] _render_doc_context_inline 호출 제거, 일반 시스템 알림만 단일 렌더
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
                            "is_generating_answer", False, session_id=current_sid
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

# 프로세스 위상 라벨 (Phase 3: 답변 위 자막용). status 텍스트 키워드 기반으로
# 진행 위상을 추론해 "문서 검색 → 증거 수집 → 답변 생성" 흐름을 표시한다.
_PHASE_LABELS = get_phase_labels()


def _infer_proc_phase(status_text: str) -> int:
    """status 텍스트가 속한 프로세스 위상 인덱스를 반환한다 (미검출 -1)."""
    lowered = status_text.lower()
    for idx, (_label, _keywords) in enumerate(_PHASE_LABELS):
        if any(k.lower() in lowered for k in _keywords):
            return idx
    return -1


def _proc_phase_caption(status_text: str, elapsed_sec: float) -> str:
    """답변 위 자막용 프로세스 위상 캡션 텍스트를 만든다."""
    active_idx = _infer_proc_phase(status_text)
    parts: list[str] = []
    for idx, (label, _keywords) in enumerate(_PHASE_LABELS):
        if active_idx >= 0 and idx == active_idx:
            parts.append(f"▸ {label}")
        elif active_idx > idx:
            parts.append(f"✓ {label}")
        else:
            parts.append(label)
    line = "  ·  ".join(parts)
    if active_idx == -1 and status_text:
        line = f"{line}  ·  {status_text}"
    return f"{line} · {elapsed_sec:.0f}s ▍"


def get_stopped_answer_actions() -> dict[str, Any]:
    """중단된 답변용 구조화 액션 (chat_build.get_build_error_actions 미러링)."""
    message = t("status_stopped")
    return {
        "message": message,
        "cause": message,
        "retryable": True,
        "retry_kind": "answer",
        "cta": [t("action_retry_answer")],
    }


def _retry_stopped_answer(sid: str, query: str) -> None:
    """중단된 턴의 마지막 질문을 새 스트리밍 플레이스홀더로 재제출하는 콜백."""
    if not query.strip():
        return
    stream_msg_id = str(uuid.uuid4())
    SessionManager.add_message(
        "assistant",
        "",
        msg_type="streaming",
        msg_id=stream_msg_id,
        query=query,
        thought="",
        documents=[],
        metrics={},
        citations=[],
        processed_content=None,
        session_id=sid,
    )
    SessionManager.set("is_generating_answer", True, sid)
    SessionManager.set("generation_cancel", False, sid)
    SessionManager.set("active_stream_msg_id", stream_msg_id, sid)


def _draw_streaming_message(msg: dict[str, Any], current_sid: str) -> None:
    """스트리밍 메시지를 그립니다 (단일 pass 렌더).

    라이브 스트리밍은 ``_render_streaming_with_write_stream``이 ``st.write_stream``
    기반으로 동일 script run 안에서 처리한다. 이 함수는 그 비활성/폴백 브랜치로,
    렌더 시점의 msg 스냅샷을 한 번 읽어 익스팬더("Answer details")를 먼저 그리고
    그 아래에 본문을 그린다. 내용이 없는(■ 중지로 영속을 건너뛴) 플레이스홀더는
    [UX-3] 조기 분기에서 중립 캡션("생성이 중단되었습니다")으로 처리한다.
    """
    # 스트리밍 중 오류가 실린 메시지는 즉시 표면화
    if msg.get("error"):
        st.error(str(msg.get("error")))
        return

    # [UX-3] 내용이 없는(■ 중지로 영속을 건너뛴) 스트리밍 플레이스홀더:
    # "Generating..." 거짓 진행 표시 금지. 중립 문구로 처리하고 진행 expander 생략.
    if not msg.get("content"):
        actions = get_stopped_answer_actions()
        st.caption(t("status_stopped"))
        query = msg.get("query", "") or ""
        st.button(
            actions["cta"][0],
            key=f"retry_stopped_{msg.get('msg_id', '') or current_sid}",
            on_click=_retry_stopped_answer,
            args=(current_sid, query),
            disabled=not query.strip(),
            help=t("action_retry_answer_help"),
            use_container_width=True,
        )
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
    """사용자의 현재 컨텍스트에 부합하는 플레이스홀더와 활성화 여부를 결정합니다."""
    is_generating = bool(SessionManager.get("is_generating_answer", False, sid))
    is_swapping = bool(SessionManager.get("is_swapping_model", False, sid))
    is_building = bool(SessionManager.get("is_building_rag", False, sid))
    has_file = bool(SessionManager.get("last_uploaded_file_name", "", sid))
    is_ready = SessionManager.is_ready_for_chat(session_id=sid)
    messages = SessionManager.get_messages(session_id=sid) or []

    # 1. 답변 생성 중
    if is_generating:
        return "답변을 생성하고 있습니다... (중지는 우측 하단 ■ 버튼)", False

    # 2. 모델 교체 중
    if is_swapping:
        return "AI 모델을 전환하는 중입니다... 잠시만 기다려 주세요.", True

    # 3. 문서 분석/인덱싱 진행 중 (기존의 치명적 오류 해결)
    if is_building:
        return "문서 지식을 분석하고 있습니다... (완료 후 질문 가능)", True

    # 4. 문서가 아직 업로드되지 않음
    if not has_file:
        return "좌측 사이드바에서 PDF 문서를 먼저 업로드해 주세요.", True

    # 5. 문서는 올라왔으나 파이프라인 준비 미완료 (오류 등)
    if not is_ready:
        return "문서 인덱스를 준비 중입니다. 잠시만 기다려 주세요.", True

    # 6. 준비 완료: 첫 질문 vs 후속 질문 분기
    has_user_msg = any(m.get("role") == "user" for m in messages)
    if not has_user_msg:
        return "문서 내용에 대해 궁금한 점을 질문해 보세요.", False

    return "이어서 추가 질문을 입력하세요...", False


def _consume_sample_question() -> str | None:
    """Pop and return a pending sample question from session state, if any."""
    return st.session_state.pop(SAMPLE_QUESTION_STATE_KEY, None)


def render_chat_input_area() -> None:
    """Renders the native st.chat_input() at the bottom of the chat column.

    입력창 영역에는 폴링 fragment를 쓰지 않는다. 제출(submit_mode="stop")과
    스트리밍 소비는 **동일한 script run** 안에서 처리한다: 제출 시 st.rerun()을
    호출하지 않으면(분할 시 ■ 중지 버튼 미렌더 — Playwright 실측), ui.py가
    메시지 영역보다 먼저 렌더해 둔 입력창 뒤에서 타임라인의 `mtype=="streaming"`
    브랜치가 run 후반부에 st.write_stream으로 스트림을 소비한다. 생성 완료/예외/
    중지는 _render_streaming_with_write_stream의 finally가 is_generating_answer를
    즉시 False로 리셋한다(INT-입력동결 방지). 빌드 진행 바는 이 영역이 아닌 별도의
    ``_render_build_progress_fragment``(1.5초 폴링)가 담당한다.
    """
    current_sid = SessionManager.get_session_id()

    pending_sample = _consume_sample_question()
    if pending_sample and SessionManager.is_ready_for_chat(session_id=current_sid):
        SessionManager.add_message("user", pending_sample, session_id=current_sid)
        stream_msg_id = str(uuid.uuid4())
        SessionManager.add_message(
            "assistant",
            "",
            msg_type="streaming",
            msg_id=stream_msg_id,
            query=pending_sample,
            thought="",
            documents=[],
            metrics={},
            citations=[],
            processed_content=None,
            session_id=current_sid,
        )
        SessionManager.set("is_generating_answer", True, current_sid)
        SessionManager.set("generation_cancel", False, current_sid)
        SessionManager.set("active_stream_msg_id", stream_msg_id, current_sid)
        return

    # 생성 중에도 위젯을 disabled로 계속 렌더(입력창 소실 방지).
    input_placeholder, input_disabled = _resolve_chat_input_state(current_sid)

    # [UX-3] 실사용 취소 = 네이티브 ■ 중지(submit_mode="stop") — ScriptRunner가
    # StopException(BaseException)을 발생시켜 영속화(아래 :628-645)를 건너뛴다.
    # generation_cancel 플래그는 프로그래매틱 전용(현재 src 호출자 없음)이며
    # 영속 메시지의 cancelled 마커로만 쓰인다.
    user_query = st.chat_input(
        input_placeholder,
        disabled=input_disabled,
        key=MAIN_CHAT_INPUT_KEY,
        submit_mode="stop",
    )

    if user_query:
        if input_disabled:
            st.error(input_placeholder)
            return
        query_text = user_query.strip()
        if query_text:
            _t_submit = time.perf_counter()
            SessionManager.add_message("user", query_text, session_id=current_sid)

            # [FIX-ORDER] 스트리밍 버블을 입력창 아래(분리)가 아닌 대화 타임라인
            # 안(질문 바로 아래)에 그리려면, 스트리밍 루프를 입력 영역에서 직접
            # 돌리지 않는다. 대신 `streaming` 타입 플레이스홀더와 플래그를 세운 뒤
            # **이 동일한 script run의 후반부**(ui.py가 입력 영역을 메시지 영역보다
            # 먼저 렌더하므로 [FIX-STREAM-INPUT]) 타임라인의 `mtype=="streaming"`
            # 브랜치가 라이브 렌더 + 스트림 소비를 함께 수행한다.
            # ⚠️ 여기서 st.rerun()을 호출하지 않는다: rerun은 제출과 스트리밍을
            # 서로 다른 script run으로 분할해, 프론트엔드 submittedRunScope가
            # 초기화되어 submit_mode="stop"의 ■ 중지 버튼이 렌더되지 않는다
            # (Playwright 실측: split-rerun 패턴 stop 버튼 0개 vs 동일 실행 패턴 1개).
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
                "[PERF] submit handler: setup took %.3fs",
                time.perf_counter() - _t_submit,
            )
            # 스트림 소비는 이 run의 타임라인 브랜치가 담당하므로 여기서 종료한다.
            return


def _friendly_stream_error(exc: Exception) -> str:
    """원시 예외를 사용자 친화 메시지로 매핑합니다 (lazy import로 순환 의존 회피)."""
    from ui.components.streaming import friendly_error_message

    return str(friendly_error_message(exc))


def _render_streaming_with_write_stream(
    msg: dict[str, Any], current_sid: str, query: str
) -> None:
    """st.write_stream 기반 스트리밍 렌더 (비블로킹).

    부가 정보(익스팬더)는 텍스트보다 **위**에 표시된다: 진행 중 ``aux_ph``에
    spinner 포함 익스팬더가 먼저 렌더링되고, ``write_stream``이 그 아래에
    텍스트를 표시한다. 완료 시 ``finally``에서 최종 메타데이터로 갱신된다.
    ``submit_mode="stop"``과 호환되어 사용자가 중지 시 부분 응답이 표시되고
    영속화된다.
    """
    msg_id = msg.get("msg_id", "")
    model_name = SessionManager.get("last_selected_model", session_id=current_sid) or ""

    aux_state: dict[str, Any] = {}
    stop_hit = False
    response = None
    stream_error: str | None = None
    try:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            aux_ph = st.empty()  # 부가 정보(expander) 고정 슬롯 — 텍스트보다 위
            status_ph = st.empty()
            status_ph.caption("질문을 분석하고 관련 지식을 검색하는 중입니다... ▍")

            def _on_status(status_text: str, elapsed_sec: float) -> None:
                status_ph.caption(_proc_phase_caption(status_text, elapsed_sec))

            def _on_aux(aux: dict[str, Any]) -> None:
                # [개선] aux_state에 thought나 documents가 유입되었을 때만 익스팬더를 슬롯에 갱신
                aux_ph.empty()
                with aux_ph:
                    render_generation_expander(
                        {
                            "thought": aux.get("thought", ""),
                            "documents": aux.get("documents", []),
                            "citations": aux.get("citations", []),
                            "metrics": aux.get("metrics", {}),
                            "model": model_name,
                            "process_steps": (aux.get("process_steps") or [])[-10:],
                            "cancelled": False,
                            "msg_id": msg_id,
                        },
                        expanded=False,
                        generating=True,
                    )

            # [개선 4] 데이터가 없는 상태에서 불필요하게 빈 익스팬더를 그리던 기존 선행 렌더 블록 제거
            # (기존 lines 410-433 삭제: aux_ph.empty()로 슬롯만 확보하고 실제 렌더링은 _on_aux에 위임)
            aux_ph.empty()

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
                        on_aux=_on_aux,
                    ),
                )
            except Exception as exc:
                stream_error = _friendly_stream_error(exc)
                response = stream_error
            finally:
                # 완료/예외/■ 중지(StopException — BaseException) 모두에서 실행되어
                # 잘못된 "생성 중" 잔상이 남지 않는다. 플레이스홀더는 매 run의 일시 요소.
                status_ph.empty()
                aux_state = SessionManager.get(_AUX_STATE_KEY, {}, current_sid) or {}
                aux_ph.empty()
                if aux_state:
                    cancelled = bool(
                        SessionManager.get("generation_cancel", False, current_sid)
                    )
                    with aux_ph:
                        render_generation_expander(
                            {
                                "thought": aux_state.get("thought", ""),
                                "documents": aux_state.get("documents", []),
                                "citations": aux_state.get("citations", []),
                                "metrics": aux_state.get("metrics", {}),
                                "model": model_name,
                                "process_steps": aux_state.get("process_steps", [])[
                                    -10:
                                ],
                                "cancelled": cancelled,
                                "msg_id": msg_id,
                            },
                            expanded=False,
                            generating=False,
                        )
    except BaseException:
        stop_hit = True  # ■ 사용자 중지 — 부분 응답 영속화 후 rerun으로 확정 렌더

    # chat_message 블록 바깥 — 메시지 영속화
    cancelled = stop_hit or bool(
        SessionManager.get("generation_cancel", False, current_sid)
    )
    error_text = aux_state.get("error")
    if error_text:
        final_error: str | None = _friendly_stream_error(Exception(error_text))
    else:
        final_error = stream_error
    SessionManager.add_message(
        "assistant",
        str(response) if response else (aux_state.get("content") or ""),
        msg_type="general",
        msg_id=msg_id,
        thought=aux_state.get("thought", ""),
        documents=aux_state.get("documents", []),
        metrics=aux_state.get("metrics", {}),
        citations=aux_state.get("citations", []),
        process_steps=aux_state.get("process_steps", [])[-10:],
        processed_content=None,
        cancelled=cancelled,
        error=final_error,
        session_id=current_sid,
    )
    SessionManager.set("is_generating_answer", False, session_id=current_sid)
    SessionManager.set("generation_cancel", False, session_id=current_sid)
    _clear_aux_state(current_sid)
    _finalize_pdf_side_effects(current_sid, msg_id)

    # 정상 완료 및 ■ 중지 모두 부분 응답 확정 및 입력창 활성화를 위해 rerun 수행
    st.rerun()


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
    "_retry_stopped_answer",
    "get_stopped_answer_actions",
    "_resolve_chat_input_state",
    "render_chat_input_area",
    "render_chat_messages_area",
    "render_generation_expander",
    "render_message",
]
