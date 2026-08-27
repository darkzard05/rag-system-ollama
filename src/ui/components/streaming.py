"""
스트리밍 응답 소비·업데이트 컴포넌트.

- stream_chunks: 비동기 RAG 스트림을 동기 Streamlit 환경에서 소비하는
  스레드+큐 브릿지 (3회 연속 타임아웃 가드 포함).
- consume_stream_into_message: ``stream_chunks``를 단일 script run 안에서
  동기 소비해 어시스턴트 메시지를 영속화한다 (백그라운드 스레드 없음).
  content/thought/documents/metrics/citations 누적, raw_json의 final_answer
  추출, 사용자 중단(cancelled) 감지, PDF 주석 반영을 포함한다.
- _finalize_pdf_side_effects: 완료 턴의 PDF 주석 반영을 담당한다.
"""

import asyncio
import logging
import queue
import re
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future
from typing import Any

from streamlit.runtime.scriptrunner import add_script_run_ctx

from api.streaming_handler import StreamChunk, get_streaming_handler
from common.async_worker import AsyncWorker
from common.config import (
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    UI_STREAMING_TIMEOUT,
)
from common.utils import (
    extract_annotations_from_docs,
)
from core.session import SessionManager
from ui.components.common import get_doc_metadata

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


def _build_process(msg: dict[str, Any]) -> dict[str, Any]:
    """스트리밍 중 누적한 단계·문서·지표를 요약 사전으로 변환합니다.

    완료 시점(finally)에서 소비 스레드가 세션 락을 보유한 채 호출되므로 순수
    함수여야 하며, msg를 변경하거나 I/O/SessionManager를 호출하지 않습니다.
    """
    steps: list[str] = []
    for step in msg.get("process_steps") or []:
        if not steps or steps[-1] != step:
            steps.append(step)
    steps = steps[-10:]

    documents = msg.get("documents") or []

    sections: list[str] = []
    seen_sections: set[str] = set()
    for d in documents:
        meta = get_doc_metadata(d)
        section = meta.get("current_section")
        if section and section not in seen_sections:
            seen_sections.add(section)
            sections.append(section)
            if len(sections) == 5:
                break

    top_scores: list[dict[str, Any]] = []
    for d in documents:
        meta = get_doc_metadata(d)
        if (score := meta.get("rerank_score")) is None:
            continue
        try:
            top_scores.append(
                {
                    "section": meta.get("current_section", ""),
                    "score": round(float(score), 3),
                }
            )
        except (TypeError, ValueError):
            continue
    top_scores.sort(key=lambda s: s["score"], reverse=True)
    top_scores = top_scores[:3]

    metrics = msg.get("metrics") or {}
    _ALLOWED_METRIC_KEYS = {
        "total_time",
        "tps",
        "input_token_count",
        "token_count",
        "relevant_docs_count",
    }
    perf = {k: v for k, v in metrics.items() if k in _ALLOWED_METRIC_KEYS}

    return {
        "steps": steps,
        "retrieved_count": len(documents),
        "sections": sections,
        "top_scores": top_scores,
        "perf": perf,
    }


def _extract_final_answer_delta(buffer: str, start: int) -> tuple[str, int]:
    """Incrementally pull the growing `final_answer` string value out of a
    partial JSON buffer. Returns (new_delta_text, new_scan_pos).

    Tracks the `final_answer` key's string value char-by-char, honoring
    escapes (\\\", \\\\, \\n, \\/) and nested braces. If the value is not yet
    inside a complete-enough structure, returns ("", start). `start` lets us
    resume scanning so only newly-arrived characters are emitted.
    """
    n = len(buffer)
    key = '"final_answer"'

    # 1) Locate the "final_answer" key. ALWAYS search from the start of the
    #    buffer: `start` tracks the value-scan offset, not the key position,
    #    so incremental calls must re-find the key from 0.
    idx = buffer.find(key, 0)
    if idx == -1:
        return "", 0
    # Ensure it is a real key (followed by optional whitespace + colon).
    after_key = idx + len(key)
    lead = buffer[after_key:].lstrip()
    if not lead.startswith(":"):
        return "", 0

    # 2) The first '"' after the key is the value's opening quote. Skip past
    #    the colon to locate it (the key's own closing quote is before it).
    quote_idx = buffer.find('"', after_key)
    if quote_idx == -1:
        return "", 0

    # How many value chars were already emitted before this call.
    emit_from = max(0, start - (quote_idx + 1))

    # 3) Copy characters until the matching unescaped closing quote, emitting
    #    only the slice that is NEW since `start` (no duplication).
    i = quote_idx + 1
    delta_chars: list[str] = []
    while i < n:
        ch = buffer[i]
        if ch == "\\":
            # Emit the escaped pair verbatim; skip the escape + next char.
            if i + 1 < n:
                delta_chars.append(ch)
                delta_chars.append(buffer[i + 1])
                i += 2
                continue
            # Trailing backslash: incomplete escape, stop scanning.
            break
        if ch == '"':
            # Found the closing quote → value complete.
            new_delta = "".join(delta_chars)[emit_from:]
            return new_delta, i + 1
        delta_chars.append(ch)
        i += 1

    # Value still open; emit only chars we have not emitted yet.
    new_delta = "".join(delta_chars)[emit_from:]
    return new_delta, i


_FA_RE = re.compile(r'"final_answer"\s*:\s*"(.*)', re.DOTALL)


def _recover_final_answer(blob: str) -> str | None:
    """깨진 JSON에서 final_answer 값을 정규식으로 복구한다.

    스트리밍 중 누적된 raw_json 이 닫히지 않은 따옴표/이스케이프로 인해
    json.loads 에 실패하더라도, 버블에는 원시 JSON 이 아닌 복구된 정답 텍스트만
    남도록 한다. 복구 불가능하면 None 반환.
    """
    if not blob:
        return None
    m = _FA_RE.search(blob)
    if not m:
        return None
    value = m.group(1)
    # 닫는 따옴표가 있으면 그 전까지, 없으면 그대로(열린 값) 사용
    end = value.find('"')
    if end != -1:
        value = value[:end]
    return value.strip()


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


def stream_chunks(
    query: str, model_name: str, session_id: str
) -> Iterator[StreamChunk]:
    """비동기 스트림을 동기 Streamlit 환경에서 소비하기 위한 브릿지 제너레이터

    취소 갭: 타임아웃 시 _future.cancel()로 AsyncWorker 태스크를 취소하지만,
    LangGraph/LLM 내부의 동기 호출 구간에서는 다음 await 지점에서만 취소가
    반영됩니다. join(2s)이 타임아웃되면 daemon 스레드는 LLM 호출이 끝날 때까지
    RAG 작업을 계속 실행합니다 (리소스는 rag_core finally에서 해제됨).
    """
    q: queue.Queue = queue.Queue()
    _stop_event = threading.Event()
    _future: Future | None = None

    def bg_task() -> None:
        nonlocal _future
        SessionManager.set_session_id(session_id or "default")

        async def run() -> None:
            try:
                from core.rag_core import RAGSystem

                sid = session_id or "default"
                rag_sys = RAGSystem(session_id=sid)
                event_generator = await rag_sys.astream(query, model_name=model_name)
                handler = get_streaming_handler()
                event_stream = handler.stream_graph_events(event_generator)

                async for chunk in event_stream:
                    if _stop_event.is_set():
                        break
                    q.put(("chunk", chunk))
            except asyncio.CancelledError:
                logger.info("[CHAT] 스트리밍 작업이 취소되었습니다")
            except Exception as e:
                logger.error(f"[CHAT] RAG 스트림 처리 오류: {e}", exc_info=True)
                q.put(("error", e))
            finally:
                q.put(("done", None))

        try:
            _future = AsyncWorker().submit(run())
            _future.result()
        except CancelledError:
            logger.info("[CHAT] 스트리밍 작업이 취소되었습니다")
        except Exception as e:
            logger.error(f"[CHAT] 백그라운드 작업 오류: {e}", exc_info=True)

    t = threading.Thread(
        target=bg_task, daemon=True, name=f"chat-stream-{session_id[-12:]}"
    )
    add_script_run_ctx(t)
    t.start()

    # 연속 타임아웃 카운터: 짧은 지연으로 인한 오탐지 방지
    _timeout_count = 0
    _max_timeouts = 3

    try:
        while True:
            try:
                msg_type, data = q.get(timeout=UI_STREAMING_TIMEOUT)
                _timeout_count = 0  # 성공 시 카운터 리셋
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    raise data
                else:
                    yield data
            except queue.Empty:
                _timeout_count += 1
                if _timeout_count >= _max_timeouts:
                    logger.error(
                        f"[CHAT] 스트리밍 타임아웃 ({_max_timeouts}회 연속): "
                        "백그라운드 작업이 응답하지 않음"
                    )
                    raise TimeoutError(
                        "스트리밍 응답을 기다리는 동안 시간이 초과되었습니다. "
                        "네트워크 및 모델 상태를 확인해주세요."
                    ) from None
                logger.debug(
                    "[CHAT] 스트리밍 타임아웃 경고 (%d/%d)",
                    _timeout_count,
                    _max_timeouts,
                )
            except Exception as e:
                logger.error(f"[CHAT] 스트리밍 오류: {e}")
                raise
    finally:
        _stop_event.set()
        if _future is not None and not _future.done():
            _future.cancel()
        if t.is_alive():
            t.join(timeout=2)
            if t.is_alive():
                logger.warning(
                    f"[CHAT] 스트림 스레드가 제한 시간 내 종료되지 않음 "
                    f"(취소는 다음 await에서만 반영됨): {t.name}"
                )
            else:
                logger.debug(f"[CHAT] Stream thread cleanup: {t.name}")


def stream_content(query: str, model_name: str, session_id: str) -> Iterator[str]:
    """``st.write_stream`` 호환 content-only 브릿지 제너레이터.

    표준 스트리밍 패턴(``st.chat_message`` 안에서 ``st.write_stream``)에서
    토큰 본문만 점진적으로 흘리기 위해 ``stream_chunks``의 ``content``
    델타만 yield 한다. thought/documents/metrics/citations 같은 부가 정보는
    호출자가 별도로 소비할 수 있도록 ``stream_chunks``를 직접 사용한다.

    취소/타임아웃 가드는 ``stream_chunks`` 내부 큐 브릿지가 그대로 보장한다.
    """
    for chunk in stream_chunks(query, model_name, session_id):
        if chunk.content:
            yield chunk.content


def consume_stream_into_message(
    sid: str, query: str, model_name: str
) -> dict[str, Any] | None:
    """위젯 없이 ``stream_chunks``를 동기 소비해 메시지를 영속화한다.

    표준 스트리밍 리팩터의 순수 로직 코어: 백그라운드 스레드 없이 단일
    호출 안에서 청크를 순회하며 본문(raw_json 시 final_answer만 추출),
    thought/documents/metrics/citations를 누적하고, 완료 시 어시스턴트
    메시지(content)를 SessionManager에 영속화한다. UI 렌더(``_run_standard_
    streaming_turn``)와 단위 테스트가 동일 코어를 재사용한다.

    반환: 영속화된 어시스턴트 메시지 dict (실패 시 error 필드 포함).
    """
    msg_id = str(uuid.uuid4())
    SessionManager.add_message(
        role="assistant",
        content="",
        msg_type="general",
        msg_id=msg_id,
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
    raw_json_parts: list[str] = []
    fa_scan_pos = 0

    try:
        for chunk in stream_chunks(query, model_name, sid):
            # 사용자 중단 요청 감지 → 누적된 부분 콘텐츠를 그대로 확정.
            if SessionManager.get("generation_cancel", False, session_id=sid):
                logger.info("[CHAT] 사용자가 답변 생성을 중단했습니다.")
                break
            if not chunk.content:
                if chunk.thought:
                    thought += chunk.thought
                # 상태(파이프라인 단계) 청크 누적 → 완료 시 process_steps에 보존.
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
                continue
            if getattr(chunk, "raw_json", False):
                raw_json_parts.append(chunk.content)
                blob = "".join(raw_json_parts)
                delta, fa_scan_pos = _extract_final_answer_delta(blob, fa_scan_pos)
                accumulated += delta
            else:
                accumulated += chunk.content
            if chunk.thought:
                thought += chunk.thought
            _meta = chunk.metadata or {}
            if _meta.get("documents"):
                documents = _meta["documents"]
            if chunk.performance:
                metrics = chunk.performance
            if getattr(chunk, "citations", None):
                citations = chunk.citations or []
    except Exception as exc:  # noqa: BLE001 - 스트림 레벨 오류를 메시지에 보존
        logger.exception("[CHAT] 스트리밍 소비 오류: %s", exc)
        SessionManager.set("is_generating_answer", False, current_sid=sid)
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
            error=friendly_error_message(exc),
            session_id=sid,
        )
        return _target_message_dict(sid, msg_id)

    cancelled = bool(SessionManager.get("generation_cancel", False, session_id=sid))
    SessionManager.set("is_generating_answer", False, current_sid=sid)
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
        cancelled=cancelled,
        session_id=sid,
    )
    # 확정 상태 저장이 클리어보다 먼저 수행되어야 한다 (G4 순서 함정 회귀 방지).
    SessionManager.set("generation_cancel", False, current_sid=sid)
    # 완료 턴의 PDF 주석 반영 (기존 백그라운드 스레드 finally 역할을 동기 수행).
    _finalize_pdf_side_effects(sid, msg_id)
    return _target_message_dict(sid, msg_id)


def _target_message_dict(sid: str, msg_id: str) -> dict[str, Any] | None:
    """세션에서 msg_id에 해당하는 메시지 dict를 반환한다."""
    messages = SessionManager.get_messages(session_id=sid)
    return next((m for m in messages if m.get("msg_id") == msg_id), None)
