"""
스트리밍 응답 소비·업데이트 컴포넌트.

- stream_chunks: 비동기 RAG 스트림을 동기 Streamlit 환경에서 소비하는
  스레드+큐 브릿지 (3회 연속 타임아웃 가드 포함).
- start_streaming_turn: streaming 메시지를 타임라인에 추가하고 백그라운드
  스레드에서 청크를 소비·업데이트한다.
- _finalize_pdf_side_effects: 완료 턴의 PDF 주석·자동 페이지 점프를 반영한다.
"""

import asyncio
import html
import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future
from typing import Any

from streamlit.runtime.scriptrunner import add_script_run_ctx

from api.streaming_handler import StreamChunk, get_streaming_handler
from common.async_worker import AsyncWorker
from common.config import MSG_ERROR_OLLAMA_NOT_RUNNING, UI_STREAMING_TIMEOUT
from common.utils import apply_tooltips_to_response, extract_annotations_from_docs
from core.session import SessionManager

logger = logging.getLogger(__name__)

_GENERIC_STREAMING_MSG = "답변 생성 중 오류가 발생했습니다."

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


def start_streaming_turn(sid: str, query: str, model_name: str) -> str:
    """스트리밍 시작: 빈 streaming 메시지 생성 후 msg_id 반환"""
    msg_id = str(uuid.uuid4())
    SessionManager.add_message(
        role="assistant",
        content="",
        msg_type="streaming",
        msg_id=msg_id,
        thought="",
        documents=[],
        metrics={},
        processed_content=None,
        session_id=sid,
    )
    # 매 턴 시작 시 취소 플래그 초기화
    SessionManager.set("generation_cancel", False, session_id=sid)
    # 백그라운드 스레드에서 스트리밍 소비 시작
    _spawn_stream_consumer(sid, msg_id, query, model_name)
    return msg_id


def _spawn_stream_consumer(sid: str, msg_id: str, query: str, model_name: str):
    """별도 스레드에서 스트림 소비 → SessionManager 메시지 직접 업데이트"""

    def bg_task():
        SessionManager.set_session_id(sid)
        try:
            for chunk in stream_chunks(query, model_name, sid):
                # 중단 요청 감지 → 누적된 부분 콘텐츠를 그대로 확정 (finally에서 처리)
                if SessionManager.get("generation_cancel", False, session_id=sid):
                    logger.info("[CHAT] 사용자가 답변 생성을 중단했습니다.")
                    break
                # 세션 락 획득 후 메시지 부분 업데이트
                with SessionManager._acquire_lock(sid):
                    state = SessionManager._get_state(sid)
                    messages = state["messages"]
                    updated = False
                    for msg in messages:
                        if msg.get("msg_id") == msg_id:
                            if chunk.status:
                                msg["status"] = chunk.status
                            if chunk.thought:
                                msg["thought"] = msg.get("thought", "") + chunk.thought
                            if chunk.content:
                                msg["content"] = msg.get("content", "") + chunk.content
                            if chunk.metadata and "documents" in chunk.metadata:
                                msg["documents"] = chunk.metadata["documents"]
                            if chunk.performance:
                                msg["metrics"] = chunk.performance
                            updated = True
                            break
                    if updated:
                        state["_dirty_keys"].add("messages")
        except Exception as e:
            logger.error(f"[STREAMING] 백그라운드 소비 오류: {e}", exc_info=True)
            with SessionManager._acquire_lock(sid):
                state = SessionManager._get_state(sid)
                for msg in state["messages"]:
                    if msg.get("msg_id") == msg_id:
                        msg["error"] = friendly_error_message(e)
                        msg["msg_type"] = "general"
                        break
                # 실패 시에도 is_generating_answer 해제 (락 재진입 금지: set() 대신 직접 수정)
                state["is_generating_answer"] = False
                state["_dirty_keys"].add("is_generating_answer")
                state["_dirty_keys"].add("messages")
        finally:
            # 완료 시 msg_type 전환
            # 주의: 세션 락은 비재진입(threading.Lock)이므로 락 보유 중
            # SessionManager.set()을 호출하면 데드락이 발생한다. 상태는 직접 수정한다.
            with SessionManager._acquire_lock(sid):
                state = SessionManager._get_state(sid)
                # is_generating_answer를 먼저 해제하여 후속 처리 오류에도 플래그가 남지 않게 함
                state["is_generating_answer"] = False
                state["generation_cancel"] = False
                state["_dirty_keys"].update(
                    {"is_generating_answer", "generation_cancel"}
                )
                for msg in state["messages"]:
                    if msg.get("msg_id") == msg_id:
                        msg["msg_type"] = "general"
                        msg["processed_content"] = apply_tooltips_to_response(
                            html.escape(msg.get("content", "")),
                            msg.get("documents", []),
                        )
                        break
                state["_dirty_keys"].add("messages")
            # PDF 주석·자동 점프는 세션 락 밖에서 수행한다 (느린 fitz 파싱 방지)
            _finalize_pdf_side_effects(sid, msg_id)

    t = threading.Thread(
        target=bg_task, daemon=True, name=f"stream-consumer-{msg_id[:8]}"
    )
    add_script_run_ctx(t)
    t.start()


def _finalize_pdf_side_effects(sid: str, msg_id: str) -> None:
    """완료된 스트리밍 턴의 PDF 주석·자동 페이지 이동을 반영합니다.

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

    # 오류 턴 또는 문서 없음 → 주석/점프 생략
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

    try:
        target_p = getattr(documents[0], "metadata", {}).get("page")
        if target_p:
            SessionManager.set(
                "pdf_target_page",
                {"page": int(target_p), "source": "auto", "ts": time.time()},
                sid,
            )
            SessionManager.set("current_page", int(target_p), sid)
    except (ValueError, TypeError, IndexError, AttributeError) as exc:
        logger.exception(f"[CHAT] 자동 페이지 이동 실패: {exc}")


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
