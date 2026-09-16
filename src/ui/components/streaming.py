"""
스트리밍 응답 소비·업데이트 컴포넌트.

- stream_chunks: 비동기 RAG 스트림을 동기 Streamlit 환경에서 소비하는
  전용-루프 스레드+큐 브릿지 (3회 연속 타임아웃 가드 포함). 공용
  AsyncWorker 루프를 공유하지 않아 스트림이 매달려도 빌드 등이 정지되지 않는다.
- consume_stream_into_message: ``stream_chunks``를 단일 script run 안에서
  동기 소비해 어시스턴트 메시지를 영속화한다 (백그라운드 스레드 없음).
  ``api.stream_events.chunk_to_stream_events`` 공유 매핑을 통해 여섯 가지
  표준 이벤트 종류(status/message/thought/sources/citations/metrics)를
  통일적으로 처리한다.
- _finalize_pdf_side_effects: 완료 턴의 PDF 주석 반영을 담당한다.
"""

import logging
import uuid
from collections.abc import Callable
from typing import Any

from api.stream_events import chunk_to_stream_events
from common.config import MSG_ERROR_OLLAMA_NOT_RUNNING
from common.utils import extract_annotations_from_docs
from core.session import SessionManager

# --- B5.2: extracted units live in sibling modules; re-exported here for
# backward compat. Every name below remains importable from this module.
from ui.components.streaming_extractors import (
    _extract_final_answer_delta,
    _FinalAnswerExtractor,
    _recover_final_answer,
)
from ui.components.streaming_runtime import stream_chunks, stream_content
from ui.components.streaming_state import (
    _build_process,
    _clear_aux_state,
    _content_generator,
    _target_message_dict,
    _write_aux_state,
)

__all__ = [
    "_AUX_STATE_KEY",
    "_build_process",
    "_clear_aux_state",
    "_content_generator",
    "_extract_final_answer_delta",
    "_FinalAnswerExtractor",
    "_finalize_pdf_side_effects",
    "_recover_final_answer",
    "_target_message_dict",
    "_write_aux_state",
    "consume_stream_into_message",
    "friendly_error_message",
    "stream_chunks",
    "stream_content",
]

logger = logging.getLogger(__name__)

_GENERIC_STREAMING_MSG = "An error occurred while generating the answer."

# 원시 예외 서명 → 사용자 친화 메시지 매핑 (config.yml errors 영역 상수 활용)
_ERROR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("connection refused", MSG_ERROR_OLLAMA_NOT_RUNNING),
    ("connection reset", MSG_ERROR_OLLAMA_NOT_RUNNING),
    ("cannot connect", MSG_ERROR_OLLAMA_NOT_RUNNING),
    ("failed to connect", MSG_ERROR_OLLAMA_NOT_RUNNING),
    ("max retries exceeded", MSG_ERROR_OLLAMA_NOT_RUNNING),
    ("연결할 수 없", MSG_ERROR_OLLAMA_NOT_RUNNING),
)


def friendly_error_message(exc: Exception) -> str:
    """원시 예외를 설정 기반 친화적 메시지로 매핑합니다. 스택/원문은 노출하지 않습니다."""
    text = str(exc).lower()
    for signature, friendly in _ERROR_SIGNATURES:
        if signature in text:
            return friendly
    return _GENERIC_STREAMING_MSG


def _finalize_pdf_side_effects(sid: str, msg_id: str) -> None:
    """완료된 스트리밍 턴의 PDF 주석을 반영합니다.

    fitz 기반 좌표 추출(느린 작업)은 세션 락 밖에서 수행하며, 실패해도 턴
    완료와 is_generating_answer 해제에는 영향을 주지 않습니다.
    """
    # 문서가 로드되지 않은 세션은 스킵 (저비용 가드)
    if not SessionManager.get("pdf_file_path", "", sid):
        return

    documents: list[Any] = []
    has_error = False
    with SessionManager._acquire_lock(sid):
        state = SessionManager._get_state(sid)
        for msg in state["messages"]:
            if msg.get("msg_id") == msg_id:
                has_error = bool(msg.get("error"))
                documents = msg.get("documents") or []
                break

    # 오류 턴 또는 문서 없음 → 주석 생략
    if has_error or not documents:
        return

    try:
        annotations = extract_annotations_from_docs(documents)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.exception(f"[STREAMING] PDF 주석 추출 실패: {exc}")
        return
    SessionManager.set(
        "pdf_annotations",
        {
            "file_hash": SessionManager.get("file_hash", None, sid),
            "annotations": annotations,
        },
        sid,
    )

    # 자동 페이지 점프 제거 (uiux-fix-p1 INT-1) — 사용자 발의 없는 화면 이동 금지.
    # 답변 완료 시 pdf_target_page/current_page를 자동 세팅하지 않는다.
    # 수동 점프는 chat.py 참조 버튼이 pdf_target_page(source="manual")로 처리한다.


def consume_stream_into_message(
    sid: str,
    query: str,
    model_name: str,
    *,
    msg_id: str | None = None,
    on_chunk: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any] | None:
    """위젯 없이 ``stream_chunks``를 동기 소비해 메시지를 영속화한다.

    순수 로직 코어: 백그라운드 스레드 없이 단일 호출 안에서 청크를 순회하며
    본문(raw_json 시 final_answer만 추출), thought/documents/metrics/citations
    를 누적하고, 완료 시 어시스턴트 메시지(content)를 SessionManager에
    영속화한다. UI 렌더는 ``on_chunk`` 콜백(라이브 렌더 셸: ``_run_active_
    stream_in_timeline``)과 단위 테스트가 동일 코어를 재사용한다.

    ``msg_id``가 주어지면 새 uuid를 만들지 않고 해당 메시지를 플레이스홀더/
    최종 영속화에 재사용한다. 플레이스홀더는 기존 메시지가 ``streaming``
    타입이면 ``streaming``을 유지해 타임라인의 스트리밍 브랜치와 일관되게
    하며, 최종 영속화 시 ``general``로 전환한다. ``msg_id``가 없으면 기존
    동작(자체 uuid + general 플레이스홀더)을 유지한다.

    ``on_chunk``: 처리된 각 청크 직후 및 스트림 종료 시(최종 스냅샷) 1회
    추가 호출되며, ``{"accumulated", "thought", "documents", "metrics",
    "citations", "process_steps", "cancelled"}`` 형식의 스냅샷 dict를 받는다.
    렌더/위젯 코드는 코어에 두지 않는다(on_chunk가 유일한 접합부).

    반환: 영속화된 어시스턴트 메시지 dict (실패 시 error 필드 포함).
    """
    resolved_id: str = msg_id or str(uuid.uuid4())

    # 플레이스홀더 타입 결정: 호출자가 전달한 기존 메시지가 streaming이면
    # streaming을 유지한다(타임라인 스트리밍 브랜치 일관성). 그 외에는
    # 기존 동작과 동일하게 general.
    placeholder_type = "general"
    if msg_id is not None:
        existing = _target_message_dict(sid, resolved_id)
        if existing is not None and existing.get("msg_type") == "streaming":
            placeholder_type = "streaming"

    SessionManager.add_message(
        role="assistant",
        content="",
        msg_type=placeholder_type,
        msg_id=resolved_id,
        thought="",
        documents=[],
        metrics={},
        citations=[],
        processed_content=None,
        session_id=sid,
    )

    accumulated = ""
    thought = ""
    documents: list[Any] = []
    metrics: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    process_steps: list[str] = []
    raw_json_extractor = _FinalAnswerExtractor()

    def _emit(cancelled: bool) -> None:
        if on_chunk is not None:
            on_chunk(
                {
                    "accumulated": accumulated,
                    "thought": thought,
                    "documents": documents,
                    "metrics": metrics,
                    "citations": citations,
                    "process_steps": list(process_steps),
                    "cancelled": cancelled,
                }
            )

    try:
        for chunk in stream_chunks(query, model_name, sid):
            # 사용자 중단 요청 감지 → 누적된 부분 콘텐츠를 그대로 확정.
            if SessionManager.get("generation_cancel", False, session_id=sid):
                logger.info("[CHAT] 사용자가 답변 생성을 중단했습니다.")
                break
            if chunk.content:
                # content는 raw_json/final_answer 처리 때문에
                # chunk.content로 직접 누적한다 (message 이벤트는 SSE 경로
                # 전용; 여기선 누적 목적으로만 사용).
                if getattr(chunk, "raw_json", False):
                    delta = raw_json_extractor.feed(chunk.content)
                    accumulated += delta
                else:
                    accumulated += chunk.content
            for ev in chunk_to_stream_events(chunk):
                if ev.type == "status":
                    step = ev.payload["message"]
                    if not process_steps or process_steps[-1] != step:
                        process_steps.append(step)
                elif ev.type == "thought":
                    thought += ev.payload["content"]
                elif ev.type == "sources":
                    if ev.payload["documents"]:
                        documents = ev.payload["documents"]
                elif ev.type == "metrics":
                    metrics = ev.payload["metrics"]
                elif ev.type == "citations":
                    citations = ev.payload["citations"]
                # "message"는 위 content 분기에서 이미 처리됨 (payload 미사용)
            _emit(False)
    except Exception as exc:  # noqa: BLE001 - 스트림 레벨 오류를 메시지에 보존
        logger.exception("[CHAT] 스트리밍 소비 오류: %s", exc)
        SessionManager.set("is_generating_answer", False, current_sid=sid)
        SessionManager.add_message(
            "assistant",
            accumulated,
            msg_type="general",
            msg_id=resolved_id,
            thought=thought,
            documents=documents,
            metrics=metrics,
            citations=citations,
            process_steps=process_steps[-10:],
            error=friendly_error_message(exc),
            session_id=sid,
        )
        _emit(False)
        return _target_message_dict(sid, resolved_id)

    cancelled = bool(SessionManager.get("generation_cancel", False, session_id=sid))
    SessionManager.set("is_generating_answer", False, current_sid=sid)
    SessionManager.add_message(
        "assistant",
        accumulated,
        msg_type="general",
        msg_id=resolved_id,
        thought=thought,
        documents=documents,
        metrics=metrics,
        citations=citations,
        process_steps=process_steps[-10:],
        processed_content=None,
        cancelled=cancelled,
        session_id=sid,
    )
    # 확정 상태 저장이 클리어보다 먼저 수행되어야 한다 (G4 순서 함정 회귀 방지).
    SessionManager.set("generation_cancel", False, current_sid=sid)
    # 스트림 종료 스냅샷을 1회 더 전달 (최종 누적 메타데이터를 라이브 렌더에 반영).
    _emit(cancelled)
    # 완료 턴의 PDF 주석 반영 (기존 백그라운드 스레드 finally 역할을 동기 수행).
    _finalize_pdf_side_effects(sid, resolved_id)
    return _target_message_dict(sid, resolved_id)


# ---------------------------------------------------------------------------
# st.write_stream 호환 content 제너레이터 + 부가 정보 누적
# ---------------------------------------------------------------------------

_AUX_STATE_KEY = "stream_active_aux"
