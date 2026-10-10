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
from streamlit.runtime.scriptrunner_utils.exceptions import StopException

from common.utils import (
    normalize_latex_delimiters,
    strip_context_tokens,
)
from core.session import SessionManager
from ui.components.chat_autoscroll import render_autoscroll_sentinel
from ui.components.common import AVATARS, role_avatar, status_line, ui_error
from ui.components.streaming import (
    _AUX_STATE_KEY,
    _clear_aux_state,
    _content_generator,
    _finalize_pdf_side_effects,
)
from ui.strings import get_phase_labels, t
from ui.widget_keys import MAIN_CHAT_INPUT_KEY, SAMPLE_QUESTION_STATE_KEY

logger = logging.getLogger(__name__)


def _metrics_caption(
    metrics: dict | None,
    model: str | None = None,
    *,
    elapsed_sec: float | None = None,
    turn_start: float | None = None,
) -> str:
    """완료 캡션(답변 아래)용 핵심 메트릭 문자열 생성 (중복 모델명 제외).

    Incremental 계약: 최종 ``total_time``이 도착하기 전에는 단조(monotonic)
    경과 시드로 캡션을 먼저 보여주고, 최종 메트릭이 오면 업그레이드한다.
    시드는 ``turn_start`` (Task0 ``_record_turn_start``/``_t_submit`` perf
    clock) 또는 이미 계산된 ``elapsed_sec`` (streaming ``on_status`` 경과)
    중 하나면 충분하다. 렌더는 호출자의 ``st.caption`` 한 줄 전용이며
    절대 블로킹하지 않는다 (순수 문자열, I/O 없음).
    """
    _ = model
    metrics = metrics or {}
    parts: list[str] = []
    total_time = metrics.get("total_time", 0)
    if isinstance(total_time, (int, float)) and total_time > 0:
        parts.append(f"{total_time:.1f}s")
    else:
        seed = elapsed_sec
        if seed is None and turn_start is not None:
            try:
                seed = time.perf_counter() - float(turn_start)
            except (TypeError, ValueError):
                seed = None
        if isinstance(seed, (int, float)) and seed > 0:
            parts.append(f"{seed:.1f}s ▍")
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
    # role→avatar 매핑은 common.role_avatar에만 둔다 (인라인 분기 drift 방지).
    avatar_icon = role_avatar(role)
    msg_id = kwargs.get("msg_id", f"msg_{msg_index}")
    citations = citations or kwargs.get("citations")
    cancelled = bool(kwargs.get("cancelled", False))
    process_steps = process_steps or kwargs.get("process_steps") or []

    with (
        st.chat_message(role, avatar=avatar_icon)
        if wrap_in_container
        else st.container()
    ):
        # 오류 메시지: 증거(익스팬더)를 보존하면서 부분 답변 본문과 오류를 안내한다.
        # 기존 early return은 thought/documents/citations 접근을 통째로 버렸다.
        # 익스팬더는 이 분기 안에서만 렌더한다 (아래 일반 분기와 중복 금지 —
        # 동일 msg_id 키의 익스팬더가 2개면 StreamlitDuplicateElementKey).
        error = kwargs.get("error")
        if error:
            # 익스팬더 삭제: 오류 턴도 참조 목록만 남기고 확정 렌더한다.
            if role == "assistant" and (documents or citations):
                render_reference_list(msg_id, documents, citations)
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
            ui_error(f"{t('error_prefix')}{error}")
            return

        # 인용 익스팬더 삭제: 완료 메시지는 본문 + 점프 스트립만 렌더한다.
        # (thought/documents/citations 익스팬더, 툴팁 주입 모두 제거 — 하이라이트
        # 데이터(documents/citations 영속)는 유지하고 UI 계산만 끊는다.)
        _ = (thought, is_latest, process_steps, metrics, cancelled)

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
            st.markdown(display_text, unsafe_allow_html=(role == "assistant"))

        # 문서별 한 줄 요약 참조 목록 (기존 [1][2] 배지 + 지표 캡션 대체).
        if role == "assistant" and (documents or citations):
            render_reference_list(msg_id, documents, citations)

        # 중단 턴은 중립 캡션만 남긴다 (성능 지표 캡션 삭제).
        if role == "assistant" and cancelled and msg_type == "general":
            st.caption(t("caption_stopped_partial"))


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
    # is_latest는 가시 마지막 메시지 기준: 스킵되는 내부 메시지(build_progress,
    # READY_FOR_QUERY)가 말미에 있어도 실제 마지막 답변의 익스팬더가 펼쳐진다.
    last_visible = len(messages) - 1
    while last_visible >= 0:
        _m = messages[last_visible]
        if not (
            (_m.get("role") == "system" and _m.get("msg_type") == "build_progress")
            or _m.get("content") == "READY_FOR_QUERY"
        ):
            break
        last_visible -= 1
    for i, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")
        mtype = msg.get("msg_type", "general")
        is_latest = i == last_visible

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

    logger.info(
        "[PERF] _render_unified_timeline: rendered %d msg(s) in %.3fs",
        n_msgs,
        time.perf_counter() - _t_tl,
    )


# ---------------------------------------------------------------------------
# 스트리밍 전용 렌더 (단일 pass, 폴링 없음)
# ---------------------------------------------------------------------------

# 프로세스 위상 라벨 (Phase 3: 답변 위 자막용). status 텍스트 키워드 기반으로
# 진행 위상을 추론해 "문서 검색 → 증거 수집 → 답변 생성" 흐름을 표시한다.
# NOTE: 라벨은 호출 시점의 활성 로케일로 해석해야 스트리밍 중 언어 전환이
# 자막에 실시간 반영된다. import 시점에 바인딩하면 이전 언어로 고정되므로
# 아래 함수들은 모듈 상수가 아닌 get_phase_labels()를 직접 호출한다.
# _PHASE_LABELS는 하위 호환용 별칭으로만 유지한다.
_PHASE_LABELS = get_phase_labels()


def _infer_proc_phase(status_text: str) -> int:
    """status 텍스트가 속한 프로세스 위상 인덱스를 반환한다 (미검출 -1)."""
    lowered = status_text.lower()
    for idx, (_label, _keywords) in enumerate(get_phase_labels()):
        if any(k.lower() in lowered for k in _keywords):
            return idx
    return -1


def _proc_phase_caption(status_text: str, elapsed_sec: float) -> str:
    """답변 위 자막용 프로세스 위상 캡션 텍스트를 만든다."""
    active_idx = _infer_proc_phase(status_text)
    parts: list[str] = []
    for idx, (label, _keywords) in enumerate(get_phase_labels()):
        if active_idx >= 0 and idx == active_idx:
            parts.append(f"▸ {label}")
        elif active_idx > idx:
            parts.append(f"✓ {label}")
        else:
            parts.append(label)
    # 마지막 하드코딩 구분자를 status_line으로 일원화한다: 앱 전역 상태줄
    # 규격인 한 칸 미들닷(" · ")을 사용하고 빈 조각은 건너뛴다.
    line = status_line(*parts)
    if active_idx == -1 and status_text:
        line = status_line(line, status_text)
    return status_line(line, f"{elapsed_sec:.0f}s ▍")


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
        ui_error(str(msg.get("error")))
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
            width="stretch",
        )
        return

    status_text = msg.get("status", t("status_fallback_generating"))
    cancel_requested = bool(SessionManager.get("generation_cancel", False, current_sid))
    if cancel_requested:
        status_text = t("status_fallback_stopping")

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


def _record_turn_start(sid: str, msg_id: str, t_start: float) -> None:
    """Persist the turn start clock for post-stream [PERF] elapsed baselines."""
    SessionManager.set("perf_turn_start", t_start, session_id=sid)
    SessionManager.set("perf_turn_msg_id", msg_id, session_id=sid)


def report_rerun_wall_if_pending(sid: str) -> None:
    """Log the true post-stream rerun wall on the script run after st.rerun().

    The streaming tail persists ``perf_rerun_start`` before raising through
    ``st.rerun()`` (which discards locals via RerunException), so the next
    script run landing here measures the real wall (queue + boot + first
    render) instead of the ~0ms ``st.rerun()`` call time. Logging only —
    pending keys are consumed exactly once.
    """
    start = SessionManager.get("perf_rerun_start", None, sid)
    if start is None:
        return
    try:
        wall_sec = time.perf_counter() - float(start)
    except (TypeError, ValueError):
        wall_sec = -1.0
    msg_id = SessionManager.get("perf_rerun_msg_id", "", sid) or ""
    SessionManager.set("perf_rerun_start", None, session_id=sid)
    SessionManager.set("perf_rerun_msg_id", None, session_id=sid)
    if wall_sec < 0:
        return
    turn_start = SessionManager.get("perf_turn_start", None, sid)
    try:
        turn_sec = (
            time.perf_counter() - float(turn_start) if turn_start is not None else -1.0
        )
    except (TypeError, ValueError):
        turn_sec = -1.0
    if turn_sec >= 0:
        logger.info(
            "[PERF] rerun-wall msg_id=%s took %.3fs turn_elapsed=%.3fs",
            msg_id,
            wall_sec,
            turn_sec,
        )
    else:
        logger.info("[PERF] rerun-wall msg_id=%s took %.3fs", msg_id, wall_sec)


def render_chat_messages_area() -> None:
    """Renders the chat column: unified timeline (streaming included)."""
    current_sid = SessionManager.get_session_id()
    report_rerun_wall_if_pending(current_sid)

    # 분석 상태 블록은 타임라인 흐름 안에 둔다(별도 배치 시 시작 말풍선과
    # 레이아웃 충돌). 마운트 판단은 BuildState 단일 predicate에 위임한다
    # (should_mount_fragment: 비터미널일 때만 fragment, 그 외 정적 block).
    # 평탄키(is_building_rag/rebuild_cancelled) 직접 분기는 폐지(3B) —
    # 입력 placeholder 가드(_resolve_chat_input_state)용으로만 유지한다.
    # 유휴 폴링 금지, run_every 없음(4A). 진행 중 갱신은 빌드 워커의
    # 상태 전파 rerun으로 전달된다.
    from ui.components.doc_analysis.panel import should_mount_fragment

    if should_mount_fragment(current_sid):
        _render_build_progress_fragment(current_sid)
    else:
        _render_build_progress_block(current_sid)

    # 통합 타임라인 렌더. 스트리밍 메시지도 단일 pass로 직접 렌더하므로
    # 익스팬더 위치가 안정적으로 유지된다.
    _render_unified_timeline(current_sid)

    is_active = bool(
        SessionManager.get("is_generating_answer", False, current_sid)
        and SessionManager.get("active_stream_msg_id", "", current_sid)
    )
    render_autoscroll_sentinel(current_sid, force=is_active)


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
        return t("chat_placeholder_generating"), False

    # 2. 모델 교체 중
    if is_swapping:
        return t("chat_placeholder_swapping"), True

    # 3. 문서 분석/인덱싱 진행 중 (기존의 치명적 오류 해결)
    if is_building:
        return t("chat_placeholder_building"), True

    # 4. 문서가 아직 업로드되지 않음
    if not has_file:
        return t("chat_placeholder_no_pdf"), True

    # 5. 문서는 올라왔으나 파이프라인 준비 미완료 (오류 등)
    if not is_ready:
        return t("chat_placeholder_not_ready"), True

    # 6. 준비 완료: 첫 질문 vs 후속 질문 분기
    has_user_msg = any(m.get("role") == "user" for m in messages)
    if not has_user_msg:
        return t("chat_placeholder_ready"), False

    return t("chat_placeholder_next"), False


def _consume_sample_question() -> str | None:
    """Pop and return a pending sample question from session state, if any."""
    return st.session_state.pop(SAMPLE_QUESTION_STATE_KEY, None)


def _peek_sample_question() -> str | None:
    """Return the pending sample question without consuming it."""
    return st.session_state.get(SAMPLE_QUESTION_STATE_KEY)


def render_chat_input_area() -> None:
    """Renders the native st.chat_input() at the bottom of the chat column.

    입력창 영역에는 폴링 fragment를 쓰지 않는다. 제출(submit_mode="stop")과
    스트리밍 소비는 **동일한 script run** 안에서 처리한다: 제출 시 st.rerun()을
    호출하지 않으면(분할 시 ■ 중지 버튼 미렌더 — Playwright 실측), ui.py가
    메시지 영역보다 먼저 렌더해 둔 입력창 뒤에서 타임라인의 `mtype=="streaming"`
    브랜치가 run 후반부에 st.write_stream으로 스트림을 소비한다. 생성 완료/예외/
    중지는 _render_streaming_with_write_stream의 finally가 is_generating_answer를
    즉시 False로 리셋한다(INT-입력동결 방지). 빌드 진행 바는 이 영역이 아닌 별도의
    ``_render_build_progress_fragment``(빌드 중에만 마운트, run_every 없음)가 담당한다.
    """
    current_sid = SessionManager.get_session_id()

    # 샘플 질문은 검증 후에만 소비한다: 준비 미완료면 큐에 남겨 유실을 막고,
    # 빈 문자열이면 버린다. 유효하면 스트리밍 플레이스홀더를 세운 뒤에도
    # 입력창을 항상 렌더해 DOM 순서를 유지한다 (ui.render_main_content 계약).
    pending_raw = _peek_sample_question()
    if isinstance(pending_raw, str):
        if not pending_raw.strip():
            # 빈 샘플 질문은 버리고 입력창 렌더로 진행한다.
            _consume_sample_question()
        elif SessionManager.is_ready_for_chat(session_id=current_sid):
            pending_sample = (_consume_sample_question() or "").strip()
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
            _record_turn_start(current_sid, stream_msg_id, time.perf_counter())
            # 조기 return 금지: 입력창을 항상 렌더해 DOM 순서를 유지한다.
            # 타임라인 브랜치가 이 run 후반부에서 스트림을 소비한다.
        # 준비 미완료면 큐에 남겨 유실을 막고 입력창 렌더로 진행한다.

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
            ui_error(input_placeholder)
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
            _record_turn_start(current_sid, stream_msg_id, _t_submit)
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


def _is_stale_ghost(aux: dict[str, Any], msg_id: str) -> bool:
    """다른 턴의 고스트 스냅샷이면 True (msg_id 불일치 시 폐기)."""
    owner = aux.get("ghost_msg_id")
    return owner is not None and owner != msg_id


def _paint_midstream_badges(aux: dict[str, Any]) -> bool:
    """Mid-stream 인용 배지를 그린다 (단일 markdown, 위젯 없음).

    ``_on_aux`` 스냅샷 게이트 안에서 호출된다: 인용이 도착한 aux에만 반응하고,
    전체 참조 본문(``_render_references_content`` 버튼 경로)은 finalize에만
    렌더한다. 스트리밍을 깨뜨리지 않도록 실패는 흡수하고 False를 반환한다.
    """
    citations = aux.get("citations") or []
    if not citations:
        return False
    try:
        return bool(render_inline_citation_badges(citations, aux.get("documents")))
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        logger.debug("[CHAT] mid-stream badges 생략: %s", exc)
        return False


def _seeded_metrics_text(aux: dict[str, Any], sid: str) -> str:
    """Pre-final 캡션 텍스트: 최종 total_time 전에는 turn clock 시드로 표시.

    최종 메트릭(``total_time``)이 있으면 ``_metrics_caption``이 최종값으로
    업그레이드하고, 없으면 Task0 ``perf_turn_start`` 단조 clock으로 시드한다.
    빈 문자열이면 호출자가 캡션을 그리지 않는다 (팬텀 캡션 금지).
    """
    metrics = aux.get("metrics") or {}
    if not isinstance(metrics, dict):
        return ""
    if metrics.get("total_time"):
        return _metrics_caption(metrics)
    turn_start = SessionManager.get("perf_turn_start", None, sid)
    if turn_start is None:
        return ""
    try:
        t0 = float(turn_start)
    except (TypeError, ValueError):
        return ""
    return _metrics_caption(metrics, turn_start=t0)


def _should_skip_final_rerun(*, cancelled: bool, error_text: str | None) -> bool:
    """Clean finalize needs no st.rerun (Task3 terminalize, safe subset).

    Badges/metrics already painted in-run via the Task2 helpers
    (``_paint_midstream_badges``/``_seeded_metrics_text``) plus the
    ``finally`` final expander, so the only pending change is the
    ``msg_type`` streaming->general conversion for the current ``msg_id``.
    ■ stop (``cancelled``) and error turns still rerun for a confirmed
    render. No fragment infrastructure — single boolean guard only.
    """
    if cancelled:
        return False
    return not error_text


def _render_streaming_with_write_stream(
    msg: dict[str, Any], current_sid: str, query: str
) -> None:
    """st.write_stream 기반 스트리밍 렌더 (동기 조인).

    부가 정보(익스팬더)는 텍스트보다 **위**에 표시된다: 진행 중 ``aux_ph``에
    spinner 포함 익스팬더가 먼저 렌더링되고, ``write_stream``이 그 아래에
    텍스트를 표시한다. 완료 시 ``finally``에서 최종 메타데이터로 갱신된다.
    ``submit_mode="stop"``과 호환되어 사용자가 중지 시 부분 응답이 표시되고
    영속화된다. tail 확정은 점프 스트립 1회만 렌더한다 (익스팬더/배지/지표 없음).
    """
    msg_id = msg.get("msg_id", "")
    model_name = SessionManager.get("last_selected_model", session_id=current_sid) or ""

    aux_state: dict[str, Any] = {}
    stop_hit = False
    response = None
    stream_error: str | None = None
    try:
        with st.chat_message("assistant", avatar=AVATARS["assistant"]):
            aux_ph = st.empty()  # 점프 스트립 확정 슬롯 — 텍스트보다 위
            status_ph = st.empty()
            status_ph.caption(t("status_initial_search"))

            def _on_status(status_text: str, elapsed_sec: float) -> None:
                status_ph.caption(_proc_phase_caption(status_text, elapsed_sec))

            def _on_aux(aux: dict[str, Any]) -> None:
                # 인용/지표 중간 렌더 삭제: 스트리밍 중에는 aux 슬롯을 비운 채
                # 두고 상태 캡션만 유지한다. 점프 스트립은 tail에서 1회 확정
                # 렌더하므로 mid-stream 위젯/마크다운 비용이 사라진다.
                if _is_stale_ghost(aux, msg_id):
                    return
                aux_ph.empty()

            # [개선 4] 데이터가 없는 상태에서 불필요하게 빈 익스팬더를 그리던 기존 선행 렌더 블록 제거
            # (기존 lines 410-433 삭제: aux_ph.empty()로 슬롯만 확보하고 실제 렌더링은 _on_aux에 위임)
            aux_ph.empty()

            # st.write_stream — 스트리밍 텍스트 표시
            response = None
            _t_join = time.perf_counter()
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
                # 삼킴 금지: 사용자에게는 제네릭 메시지만 보이지만, 로그에는
                # 원본 traceback을 남긴다 (무로그 실패 방지).
                logger.exception("[CHAT] write_stream 소비 오류: %s", exc)
                stream_error = _friendly_stream_error(exc)
                response = stream_error
            finally:
                logger.info(
                    "[PERF] join-block msg_id=%s took %.3fs (sync write_stream join)",
                    msg_id,
                    time.perf_counter() - _t_join,
                )
                # 완료/예외/■ 중지(StopException — BaseException) 모두에서 실행되어
                # 잘못된 "생성 중" 잔상이 남지 않는다. 플레이스홀더는 매 run의 일시 요소.
                status_ph.empty()
                aux_state = SessionManager.get(_AUX_STATE_KEY, {}, current_sid) or {}
                aux_ph.empty()
                # Tail 확정 렌더 (단순안): 익스팬더/배지/지표 없이 참조 목록만.
                # 본문은 write_stream이 이미 그렸고, 완료 메시지와 동일 슬롯
                # 순서(본문 아래)에 목록을 확정한다.
                if aux_state and (
                    aux_state.get("documents") or aux_state.get("citations")
                ):
                    render_reference_list(
                        msg_id,
                        aux_state.get("documents"),
                        aux_state.get("citations"),
                    )
    except StopException:
        stop_hit = True  # ■ 사용자 중지 — 부분 응답 영속화 후 rerun으로 확정 렌더
    except BaseException as exc:
        # Stop 외 BaseException은 취소가 아닌 에러로 보존한다.
        logger.exception("[CHAT] write_stream 소비 중 BaseException: %s", exc)
        stream_error = _friendly_stream_error(Exception(f"{type(exc).__name__}: {exc}"))
        response = stream_error

    # chat_message 블록 바깥 — 메시지 영속화
    cancelled = stop_hit or bool(
        SessionManager.get("generation_cancel", False, current_sid)
    )
    error_text = aux_state.get("error")
    if error_text:
        final_error: str | None = _friendly_stream_error(Exception(error_text))
    else:
        final_error = stream_error
    _t_persist = time.perf_counter()
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
    logger.info(
        "[PERF] live persist took %.3fs (%d docs, %d citations)",
        time.perf_counter() - _t_persist,
        len(aux_state.get("documents", []) or []),
        len(aux_state.get("citations", []) or []),
    )
    SessionManager.set("is_generating_answer", False, session_id=current_sid)
    SessionManager.set("generation_cancel", False, session_id=current_sid)
    _clear_aux_state(current_sid)
    _t_fin = time.perf_counter()
    _finalize_pdf_side_effects(current_sid, msg_id)
    logger.info("[PERF] finalize took %.3fs", time.perf_counter() - _t_fin)

    # Terminalize (Task3 safe subset): clean finalize already painted the
    # final expander/badges/metrics in-run, so skip st.rerun() when the only
    # pending change is msg_type streaming->general for this msg_id. The
    # finalized flag lets the next run know no rerun wall is pending.
    if _should_skip_final_rerun(cancelled=cancelled, error_text=final_error):
        SessionManager.set("stream_finalized_no_rerun", msg_id, session_id=current_sid)
        logger.info(
            "[PERF] rerun skipped msg_id=%s (final already painted in-run)",
            msg_id,
        )
        return

    # ■ 중지 및 오류 턴은 부분 응답 확정 및 입력창 활성화를 위해 rerun 수행
    # Task 0 instrumentation: st.rerun()은 RerunException을 발생시켜 호출 시간
    # 측정이 항상 ~0ms가 되므로, wall 측정은 다음 script run으로 이월한다.
    _turn_start = SessionManager.get("perf_turn_start", None, current_sid)
    try:
        _turn_sec = (
            time.perf_counter() - float(_turn_start)
            if _turn_start is not None
            else -1.0
        )
    except (TypeError, ValueError):
        _turn_sec = -1.0
    if _turn_sec >= 0:
        logger.info("[PERF] turn msg_id=%s elapsed_sec=%.3fs", msg_id, _turn_sec)
    SessionManager.set("perf_rerun_start", time.perf_counter(), session_id=current_sid)
    SessionManager.set("perf_rerun_msg_id", msg_id, session_id=current_sid)
    logger.info("[PERF] rerun scheduled msg_id=%s", msg_id)
    st.rerun()


# isort: off
from ui.components.chat_references import (  # noqa: E402
    _extract_reference_pages,
    _handle_doc_jump,
    _handle_page_jump,
    _render_references_content,
    normalize_excerpt,
    render_generation_expander,
    render_inline_citation_badges,
    render_reference_list,
)
from ui.components.chat_build import (  # noqa: E402
    _cancel_rebuild,
    _render_build_progress_block,
    _render_build_progress_fragment,
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
    "_render_guidance_panel",
    "_render_references_content",
    "_render_unified_timeline",
    "_retry_stopped_answer",
    "get_stopped_answer_actions",
    "_resolve_chat_input_state",
    "normalize_excerpt",
    "render_chat_input_area",
    "render_chat_messages_area",
    "render_generation_expander",
    "render_inline_citation_badges",
    "render_reference_list",
    "render_message",
    "report_rerun_wall_if_pending",
]
