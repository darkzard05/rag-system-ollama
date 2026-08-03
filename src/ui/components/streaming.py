"""
스트리밍 응답 소비·업데이트 컴포넌트.

- stream_chunks: 비동기 RAG 스트림을 동기 Streamlit 환경에서 소비하는
  스레드+큐 브릿지 (3회 연속 타임아웃 가드 포함).
- start_streaming_turn: streaming 메시지를 타임라인에 추가하고 백그라운드
  스레드에서 청크를 소비·업데이트한다.
- persist_completed_turn: 완료/오류 결과를 세션에 저장하고 정리.
"""

import asyncio
import html
import logging
import queue
import threading
import uuid
from collections.abc import Iterator
from concurrent.futures import CancelledError, Future
from typing import Any, TypedDict

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from api.streaming_handler import StreamChunk, get_streaming_handler
from common.async_worker import AsyncWorker
from common.config import UI_STREAMING_TIMEOUT
from common.utils import apply_tooltips_to_response, extract_annotations_from_docs
from core.session import SessionManager

logger = logging.getLogger(__name__)


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
    # 백그라운드 스레드에서 스트리밍 소비 시작
    _spawn_stream_consumer(sid, msg_id, query, model_name)
    return msg_id


def _spawn_stream_consumer(sid: str, msg_id: str, query: str, model_name: str):
    """별도 스레드에서 스트림 소비 → SessionManager 메시지 직접 업데이트"""

    def bg_task():
        SessionManager.set_session_id(sid)
        try:
            for chunk in stream_chunks(query, model_name, sid):
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
                        msg["error"] = str(e)
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
                state["_dirty_keys"].add("is_generating_answer")
                for msg in state["messages"]:
                    if msg.get("msg_id") == msg_id:
                        msg["msg_type"] = "general"
                        msg["processed_content"] = apply_tooltips_to_response(
                            html.escape(msg.get("content", "")),
                            msg.get("documents", []),
                        )
                        break
                state["_dirty_keys"].add("messages")

    t = threading.Thread(
        target=bg_task, daemon=True, name=f"stream-consumer-{msg_id[:8]}"
    )
    add_script_run_ctx(t)
    t.start()


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


class StreamingResult(TypedDict):
    """한 번의 스트리밍 라운드가 끝난 뒤의 누적/처리 결과입니다."""

    content: str
    thought: str
    documents: list[Any]
    performance: dict[str, Any]
    processed_content: str | None
    error: str | None


def persist_completed_turn(sid: str, result: StreamingResult) -> None:
    """스트리밍 완료 후 대화·PDF 상태를 저장하고 정리합니다 (한 번만 호출)."""
    if result["error"] is not None:
        SessionManager.add_message("assistant", result["error"], session_id=sid)
    else:
        documents = result["documents"]
        SessionManager.add_message(
            role="assistant",
            content=result["content"],
            thought=result["thought"],
            documents=documents,
            metrics=result["performance"],
            processed_content=result["processed_content"],
            session_id=sid,
        )

        if documents:
            annotations = extract_annotations_from_docs(documents)
            SessionManager.set("pdf_annotations", annotations, sid)
            try:
                target_p = getattr(documents[0], "metadata", {}).get("page")
                if target_p:
                    SessionManager.set("pdf_target_page", int(target_p), sid)
                    SessionManager.set("current_page", int(target_p), sid)
            except (ValueError, TypeError, IndexError, AttributeError):
                pass

    SessionManager.set("is_generating_answer", False, sid)
    st.rerun()
