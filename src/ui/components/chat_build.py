"""문서 분석 (빌드) 진행 렌더 컴포넌트.

``chat.py`` (P3 분리)에서 이동한 빌드 진행/문서 컨텍스트 렌더 함수들:
- ``_cancel_rebuild`` — 분석 재구축 취소 요청 콜백
- ``_render_build_progress_block`` — 문서 분석 카드 단독 렌더 (BuildState 우선)
- ``_render_build_card`` — 타입 상태 카드 본문 (신규)
- ``_render_build_legacy_block`` — 평탄키 폴백 렌더 (호환)
- ``_render_build_progress_fragment`` — 빌드 상태 fragment (빌드/취소 중에만 마운트)
- ``_render_guidance_panel`` — 빈 대화 가이드 패널

본 모듈은 ``chat.py``를 import하지 않는다 (순환 의존 방지).
"""

import time
from typing import Any

import streamlit as st

from core.build_state import (
    PHASE_DETAIL_KEY,
    BuildPhase,
    BuildState,
    elapsed_sec,
)
from core.session import SessionManager
from ui.components.common import AVATARS, status_line, ui_error
from ui.strings import t
from ui.widget_keys import (
    SAMPLE_QUESTION_STATE_KEY,
    cancel_rebuild_key,
    retry_build_key,
)

__all__ = [
    "_cancel_rebuild",
    "_render_build_card",
    "_render_build_legacy_block",
    "_render_build_progress_block",
    "_render_build_progress_fragment",
    "_render_guidance_panel",
    "get_build_error_actions",
]

# 스테퍼 단계 순서 (PREPARING은 시작 전이므로 전부 ○로 표시).
_BUILD_STEP_ORDER: tuple[BuildPhase, ...] = (
    BuildPhase.EXTRACTING,
    BuildPhase.CHUNKING,
    BuildPhase.INDEXING,
    BuildPhase.REGISTERING,
)

_RUNNING_PHASES: frozenset[BuildPhase] = frozenset(
    {
        BuildPhase.PREPARING,
        BuildPhase.EXTRACTING,
        BuildPhase.CHUNKING,
        BuildPhase.INDEXING,
        BuildPhase.REGISTERING,
    }
)


def _stepper_caption(phase: BuildPhase) -> str:
    """단계 스테퍼 1줄 캡션: ✓ 완료 · ▸ 진행중 · ○ 대기 (결정 3).

    단계 라벨은 리터럴 ``t()`` 호출로 읽는다 (i18n AST 게이트).
    """
    steps: list[tuple[BuildPhase, str]] = [
        (BuildPhase.EXTRACTING, t("build_step_extract")),
        (BuildPhase.CHUNKING, t("build_step_chunk")),
        (BuildPhase.INDEXING, t("build_step_index")),
        (BuildPhase.REGISTERING, t("build_step_finalize")),
    ]
    if phase == BuildPhase.DONE:
        current = len(steps)
    else:
        current = 0
        for idx, (step_phase, _label) in enumerate(steps):
            if step_phase == phase:
                current = idx
                break
    parts: list[str] = []
    for idx, (_step_phase, label) in enumerate(steps):
        if phase == BuildPhase.DONE or idx < current:
            marker = "✓"
        elif idx == current:
            marker = "▸"
        else:
            marker = "○"
        parts.append(f"{marker} {label}")
    return status_line(*parts)


def _build_cache_tag(sid: str, build: BuildState | None) -> str:
    """캐시 재사용 태그 (결정 5: 유지). 임베딩 모델명은 표기하지 않는다."""
    cache_used: bool | None = build.cache_used if build is not None else None
    if cache_used is None:
        doc_stats = SessionManager.get("doc_stats", {}, sid)
        if isinstance(doc_stats, dict):
            cache_used = doc_stats.get("cache_used")
    if cache_used:
        return t("cache_tag_reused")
    if cache_used is False:
        return t("cache_tag_new")
    return ""


def _cancel_rebuild(sid: str) -> None:
    # on_click 콜백 종료 후 Streamlit 자동 post-callback 리런에 의존 (명시 rerun 금지).
    """문서 분석 재구축 취소 요청 콜백입니다."""
    SessionManager.set("rebuild_cancelled", True, session_id=sid)


def get_build_error_actions(error_msg: str) -> dict[str, Any]:
    """Return structured build error info with retry CTA.

    Called by test_error_recovery.py and render code.
    """
    return {
        "message": error_msg,
        "cause": error_msg,
        "retryable": True,
        "retry_kind": "rebuild",
        "cta": [t("action_retry_build")],
    }


def _retry_build(sid: str) -> None:
    """문서 분석 실패/취소 시 파이프라인 재구축을 트리거하는 콜백."""
    SessionManager.reset_build_state(session_id=sid)
    SessionManager.set("pdf_processing_error", "", session_id=sid)
    SessionManager.set("rebuild_error", None, session_id=sid)
    SessionManager.set("rebuild_done", False, session_id=sid)
    SessionManager.set("rebuild_progress", 0, session_id=sid)
    SessionManager.set("rebuild_status", "Restarting analysis...", session_id=sid)
    SessionManager.set(
        "is_building_rag", False, session_id=sid
    )  # main.py의 not is_building 가드 통과 보장
    SessionManager.set("needs_rag_rebuild", True, session_id=sid)
    # 재시도 시 경과 시간 기준점을 리셋하고 취소 플래그를 해제한다.
    # 같은 버튼 키 재사용으로 인한 stale 클릭 상태도 세대 카운터로 분리한다.
    SessionManager.set("rebuild_start_time", time.time(), session_id=sid)
    SessionManager.set("rebuild_cancelled", False, session_id=sid)
    try:
        attempt = int(SessionManager.get("rebuild_attempt", 0, sid) or 0)
    except (TypeError, ValueError):
        attempt = 0
    SessionManager.set("rebuild_attempt", attempt + 1, session_id=sid)


def _render_build_progress_block(sid: str) -> None:
    """문서 분석 카드 렌더 (BuildState 우선, 레거시 키 폴백).

    - 실행 중: 상태 pill 라벨 + 스테퍼 1줄 + 게이지바 + 경과 캡션 + 취소 버튼.
    - 완료: 접힌 상태 + 소요 시간 1줄 (게이지바 없음, 결정 1).
    - 실패/취소: 에러 카드 + 재시도 버튼.
    ``build_state``가 없는 세션(구 흐름/테스트)은 기존 렌더로 폴백한다.
    """
    build = SessionManager.get_build_state(session_id=sid)
    if build is not None:
        _render_build_card(sid, build)
    else:
        _render_build_legacy_block(sid)


def _render_build_card(sid: str, build: BuildState) -> None:
    file_name = build.file_name or t("doc_default_name")
    if build.phase == BuildPhase.IDLE and not build.file_name:
        return
    elapsed = elapsed_sec(build)
    cache_tag = _build_cache_tag(sid, build)

    if build.phase in _RUNNING_PHASES:
        is_cancelling = bool(SessionManager.get("rebuild_cancelled", False, sid))
        if is_cancelling:
            label = f"{file_name} · {t('build_cancelled')}"
        else:
            phase_text = t(PHASE_DETAIL_KEY.get(build.phase, "build_in_progress"))
            label = f"{file_name} · {phase_text}"
        with st.status(label, expanded=True, state="running"):
            st.progress(max(0, min(100, build.pct)) / 100)
            st.caption(_stepper_caption(build.phase))
            st.caption(
                t("build_progress_caption").format(progress=build.pct, elapsed=elapsed)
            )
            if is_cancelling:
                return
            try:
                attempt = int(build.attempt)
            except (TypeError, ValueError):
                attempt = 0
            st.button(
                t("action_cancel_build"),
                key=cancel_rebuild_key(sid, attempt),
                on_click=_cancel_rebuild,
                args=(sid,),
                width="stretch",
            )
        return

    if build.phase == BuildPhase.DONE:
        label = f"{file_name} · 준비 완료{cache_tag}"
        with st.status(label, expanded=False, state="complete"):
            st.caption(
                t("build_complete_caption").format(elapsed=elapsed)
                if elapsed > 0
                else t("build_complete_default")
            )
        return

    if build.phase == BuildPhase.CANCELLED:
        label = f"{file_name} · {t('build_cancelled')}"
        with st.status(label, expanded=True, state="running"):
            st.progress(max(0, min(100, build.pct)) / 100)
            st.caption(_stepper_caption(build.phase))
        return

    # ERROR
    error = build.error or SessionManager.get("pdf_processing_error", "", sid) or ""
    label = f"{file_name} · 분석 실패"
    with st.status(label, expanded=True, state="error"):
        if error:
            actions = get_build_error_actions(str(error))
            ui_error(actions["message"])
            st.caption(
                t("build_progress_caption").format(progress=build.pct, elapsed=elapsed)
            )
            try:
                attempt = int(build.attempt)
            except (TypeError, ValueError):
                attempt = 0
            st.button(
                actions["cta"][0],
                key=retry_build_key(sid, attempt),
                on_click=_retry_build,
                args=(sid,),
                disabled=not actions["retryable"],
                help=t("action_retry_build_help"),
                width="stretch",
            )


def _render_build_legacy_block(sid: str) -> None:
    is_building = bool(SessionManager.get("is_building_rag", False, sid))
    is_cancelling = bool(SessionManager.get("rebuild_cancelled", False, sid))
    is_done = bool(SessionManager.get("rebuild_done", False, sid))
    has_doc = bool(SessionManager.get("last_uploaded_file_name", "", sid))

    # 빌드가 한 번도 시작되지 않은 초기 상태(업로드 전)에서는 노출하지 않는다.
    if not (is_building or is_cancelling or is_done or has_doc):
        return

    try:
        progress = max(0, min(100, int(SessionManager.get("rebuild_progress", 0, sid))))
    except (TypeError, ValueError):
        progress = 0
    status_text = str(SessionManager.get("rebuild_status", "", sid) or "")
    error = SessionManager.get("pdf_processing_error", "", sid) or ""
    is_processed = bool(SessionManager.get("pdf_processed", False, sid))

    # [개선] 파일명 및 캐시 정보 추출 (문서 컨텍스트 통합)
    file_name = str(
        SessionManager.get("last_uploaded_file_name", "", sid) or t("doc_default_name")
    )
    doc_stats = SessionManager.get("doc_stats", {}, sid)
    if not isinstance(doc_stats, dict):
        doc_stats = {}
    cache_tag = (
        t("cache_tag_reused")
        if doc_stats.get("cache_used")
        else t("cache_tag_new")
        if is_processed
        else ""
    )

    # 1. 진행 중 상태 텍스트 정제 (중복 문구 제거)
    clean_status = status_text.strip()
    if clean_status.startswith("Analyzing"):
        # "Analyzing 'file'..." 구문이 들어온 경우 중복을 제거하고 핵심 상태만 추출
        clean_status = (
            t("build_analyzing") if clean_status.endswith("...") else clean_status
        )

    if error:
        label, state, expanded = f"{file_name} · 분석 실패", "error", True
    elif is_cancelling:
        label, state, expanded = (
            f"{file_name} · {t('build_cancelled')}",
            "running",
            True,
        )
    elif (progress >= 100 or is_done) and is_processed:
        label, state, expanded = (
            f"{file_name} · 준비 완료{cache_tag}",
            "complete",
            False,
        )
    else:
        # 일관된 "{파일명} · {상태}" 포맷으로 정돈
        label, state, expanded = (
            f"{file_name} · {clean_status or t('build_in_progress')}",
            "running",
            True,
        )

    # 2. st.status 단독 사용으로 이중 아이콘/이중 프레임 노이즈 제거
    with st.status(label, expanded=expanded, state=state):
        start_time = SessionManager.get("rebuild_start_time", None, sid)
        elapsed = 0
        if start_time:
            try:
                elapsed = max(0, int(time.time() - float(start_time)))
            except (TypeError, ValueError):
                elapsed = 0

        # 3. 진행 중일 때만 게이지 바 및 진행률 표시, 완료 시에는 핵심 요약만 표시
        if state == "running":
            st.progress(progress / 100)
            st.caption(
                t("build_progress_caption").format(progress=progress, elapsed=elapsed)
            )
        elif state == "complete":
            # 100% 게이지바와 중복 캡션("100% complete")을 제거하고 유의미한 소요 시간만 간결하게 표시
            st.caption(
                t("build_complete_caption").format(elapsed=elapsed)
                if elapsed > 0
                else t("build_complete_default")
            )

        if error:
            actions = get_build_error_actions(error)
            ui_error(actions["message"])
            # 실패 직전 진행률 맥락을 유지한다 (게이지 없이 캡션 1줄).
            st.caption(
                t("build_progress_caption").format(progress=progress, elapsed=elapsed)
            )
            # 라벨/도움말은 actions 메타데이터 단일 소스에서 읽는다: 중단-답변
            # dead-end(chat.py)와 하드코딩이 다시 drift 되는 것을 방지한다.
            try:
                attempt = int(SessionManager.get("rebuild_attempt", 0, sid) or 0)
            except (TypeError, ValueError):
                attempt = 0
            st.button(
                actions["cta"][0],
                key=retry_build_key(sid, attempt),
                on_click=_retry_build,
                args=(sid,),
                disabled=not actions["retryable"],
                help=t("action_retry_build_help"),
                width="stretch",
            )
        if state == "running" and not is_cancelling:
            try:
                attempt = int(SessionManager.get("rebuild_attempt", 0, sid) or 0)
            except (TypeError, ValueError):
                attempt = 0
            st.button(
                t("action_cancel_build"),
                key=cancel_rebuild_key(sid, attempt),
                on_click=_cancel_rebuild,
                args=(sid,),
                width="stretch",
            )


@st.fragment()
def _render_build_progress_fragment(sid: str) -> None:
    """빌드 상태 카드 전용 fragment (run_every 없음).

    전체 rerun 없이 ``build_state`` 를 다시 읽어 카드를 갱신한다.
    ``run_every`` 타이머는 두지 않는다: 빌드 완료 후 전체 rerun 시 스테일
    타이머가 경고를 유발한다(test_progress_poller_mounting). 진행 중 갱신은
    빌드 워커의 상태 전파 rerun + fragment 내 인터랙션(취소 버튼) rerun으로
    전달된다. 완료/취소/에러 상태에서도 카드는 그대로 남아 기록으로 잔존한다.
    """
    _render_build_progress_block(sid)

    # on_click 콜백 종료 후 Streamlit 자동 post-callback 리런에 의존 (명시 rerun 금지).


def _on_sample_question_click(question: str) -> None:
    st.session_state[SAMPLE_QUESTION_STATE_KEY] = question


def _render_guidance_panel() -> None:
    """온보딩 단일 소스: 파일 미업로드 시에만 안내 카드를 렌더한다.

    렌더 판단(파일 게이트)과 본문(strings onboarding_guide)이 여기 일원화되어
    있으므로 호출자는 조건 없이 호출만 한다 (chat.py 타임라인 빈 분기).
    """
    sid = SessionManager.get_session_id()
    if SessionManager.get("last_uploaded_file_name", "", sid):
        return

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        st.markdown(t("onboarding_guide"))
