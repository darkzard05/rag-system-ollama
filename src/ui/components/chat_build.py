"""문서 분석 (빌드) 진행 렌더 컴포넌트.

``chat.py`` (P3 분리)에서 이동한 빌드 진행/문서 컨텍스트 렌더 함수들:
- ``_cancel_rebuild`` — 분석 재구축 취소 요청 콜백
- ``_render_build_progress_block`` — 빌드 상태 블록 단독 렌더
- ``_render_build_progress_fragment`` — 1.5초 폴링 빌드 상태 fragment
- ``_render_guidance_panel`` — 빈 대화 가이드 패널
- ``_render_doc_context_inline`` — 타임라인 첫 메시지 문서 컨텍스트

본 모듈은 ``chat.py``를 import하지 않는다 (순환 의존 방지).
"""

import time
from typing import Any

import streamlit as st

from core.session import SessionManager
from ui.components.common import AVATARS
from ui.strings import t
from ui.widget_keys import (
    SAMPLE_QUESTION_STATE_KEY,
    cancel_rebuild_key,
)

__all__ = [
    "_cancel_rebuild",
    "_render_build_progress_block",
    "_render_build_progress_fragment",
    "_render_doc_context_inline",
    "_render_guidance_panel",
    "get_build_error_actions",
]


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
        "cta": ["Retry Analysis"],
    }


def _retry_build(sid: str) -> None:
    """문서 분석 실패/취소 시 파이프라인 재구축을 트리거하는 콜백."""
    SessionManager.set("pdf_processing_error", "", session_id=sid)
    SessionManager.set("rebuild_error", None, session_id=sid)
    SessionManager.set("rebuild_done", False, session_id=sid)
    SessionManager.set("rebuild_progress", 0, session_id=sid)
    SessionManager.set("rebuild_status", "Restarting analysis...", session_id=sid)
    SessionManager.set(
        "is_building_rag", False, session_id=sid
    )  # main.py의 not is_building 가드 통과 보장
    SessionManager.set("needs_rag_rebuild", True, session_id=sid)


def _render_build_progress_block(sid: str) -> None:
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
    is_processed = bool(SessionManager.get("pdf_processed", False, sid))

    # [개선] 파일명 및 캐시 정보 추출 (문서 컨텍스트 통합)
    file_name = str(
        SessionManager.get("last_uploaded_file_name", "", sid) or "Document"
    )
    doc_stats = SessionManager.get("doc_stats", {}, sid) or {}
    cache_tag = (
        " (기존 분석 재사용)"
        if doc_stats.get("cache_used")
        else " [new]"
        if is_processed
        else ""
    )

    # 1. 진행 중 상태 텍스트 정제 (중복 문구 제거)
    clean_status = status_text.strip()
    if clean_status.startswith("Analyzing"):
        # "Analyzing 'file'..." 구문이 들어온 경우 중복을 제거하고 핵심 상태만 추출
        clean_status = (
            "지식 베이스 구축 중..." if clean_status.endswith("...") else clean_status
        )

    if error:
        label, state, expanded = f"{file_name} · 분석 실패", "error", True
    elif (progress >= 100 or is_done) and is_processed:
        label, state, expanded = (
            f"{file_name} · 준비 완료{cache_tag}",
            "complete",
            False,
        )
    elif is_cancelling:
        label, state, expanded = f"{file_name} · 분석 취소 중...", "running", True
    else:
        # 일관된 "{파일명} · {상태}" 포맷으로 정돈
        label, state, expanded = (
            f"{file_name} · {clean_status or '문서 분석 중'}",
            "running",
            True,
        )

    # 2. st.status 단독 사용으로 이중 아이콘/이중 프레임 노이즈 제거
    with st.status(label, expanded=expanded, state=state):
        start_time = SessionManager.get("rebuild_start_time", None, sid)
        elapsed = 0
        if start_time:
            try:
                elapsed = int(time.time() - float(start_time))
            except (TypeError, ValueError):
                elapsed = 0

        # 3. 진행 중일 때만 게이지 바 및 진행률 표시, 완료 시에는 핵심 요약만 표시
        if state == "running":
            st.progress(progress / 100)
            st.caption(f"{progress}% 완료 · {elapsed}초 경과")
        elif state == "complete":
            # 100% 게이지바와 중복 캡션("100% complete")을 제거하고 유의미한 소요 시간만 간결하게 표시
            st.caption(f"분석 소요 시간: {elapsed}초" if elapsed > 0 else "분석 완료")

        if error:
            actions = get_build_error_actions(error)
            st.error(actions["message"])
            st.button(
                "다시 시도",
                key=f"retry_build_{sid}",
                on_click=_retry_build,
                args=(sid,),
                use_container_width=True,
            )
        if state == "running" and not is_cancelling:
            st.button(
                "분석 취소",
                key=cancel_rebuild_key(sid),
                on_click=_cancel_rebuild,
                args=(sid,),
                use_container_width=True,
            )


@st.fragment()
def _render_build_progress_fragment(sid: str) -> None:
    """빌드 상태 블록 전용 폴링 fragment.

    전체 rerun 없이 ``rebuild_progress`` 를 다시 읽어 진행 바를
    갱신한다(타임라인 폴링이 제거된 빈틈을 메움). 빌드 완료 후에는
    ``run_in_background_worker._on_complete`` 의 rerun이 최종 100%를 확정한다.
    완료/취소/에러 상태에서도 블록은 그대로 남아 대화 기록으로 잔존한다.
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


def _render_doc_context_inline(sid: str) -> None:
    """[DEPRECATED] _render_build_progress_block으로 통합되어 no-op 처리 (하위 호환성 유지)"""
    pass
