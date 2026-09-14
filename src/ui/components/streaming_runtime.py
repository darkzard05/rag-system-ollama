"""Streaming runtime — queue-thread bridge + st.write_stream helper.

Verbatim extraction from ``streaming.py``; no logic change.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import queue
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

from streamlit.runtime.scriptrunner import add_script_run_ctx

from api.streaming_handler import StreamChunk, get_streaming_handler
from common.config import (
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    UI_STREAMING_HARD_TIMEOUT,
    UI_STREAMING_SETUP_TIMEOUT,
    UI_STREAMING_TIMEOUT,
)
from common.stream_worker import (
    acquire_stream_slot,
    cancel_stream,
    register_stream,
    release_stream_slot,
    unregister_stream,
)
from core.session import SessionManager
from ui.components.streaming_extractors import _FinalAnswerExtractor

logger = logging.getLogger(__name__)

_clock = time.monotonic

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


def stream_chunks(
    query: str, model_name: str, session_id: str
) -> Iterator[StreamChunk]:
    """비동기 스트림을 동기 Streamlit 환경에서 소비하기 위한 브릿지 제너레이터

    스트림 RAG 작업은 공용 AsyncWorker 루프가 아닌 **스트림 전용 daemon 스레드
    루프**(``asyncio.run``)에서 실행한다. RAG 셋업/LLM 내부의 동기 블로킹
    구간에서 매달려도 해당 스트림만 지연되고 빌드/다른 스트림을 정지시키지
    않는다.

    취소: 타임아웃/사용자 중단 시 ``cancel_stream(run_id)``가 실제 취소를
    요청한다. ``loop.call_soon_threadsafe(task.cancel)``로 이벤트 루프에 즉시
    전달되어 협조적 ``_stop_event``와 무관하게 하드 취소되며, 이어서 3초
    바운드 join으로 스레드 종료를 기다린다. ``_stop_event``는 다음 await
    지점에서 즉시 반응하는 협조적 백스톱으로 함께 유지한다. join(3s)이
    타임아웃되면 daemon 스레드가 남을 수 있지만 루프/스트림 슬롯은
    finally에서 해제된다.
    """
    q: queue.Queue = queue.Queue()
    _stop_event = threading.Event()
    run_id = f"stream-{session_id}-{uuid.uuid4().hex[:8]}"

    def bg_task() -> None:
        if not acquire_stream_slot(timeout=30):
            logger.warning(
                f"[CHAT][STREAM] 동시 스트림 슬롯 획득 실패 (run_id={run_id})"
            )
            q.put(
                (
                    "error",
                    RuntimeError(
                        "동시 스트림 실행 한도를 초과했습니다. "
                        "잠시 후 다시 시도해 주세요."
                    ),
                )
            )
            q.put(("done", None))
            return
        SessionManager.set_session_id(session_id or "default")

        async def run() -> None:
            event_stream: Any = None
            remaining_chunks: list[Any] = []
            try:
                from core.rag_core import RAGSystem

                sid = session_id or "default"
                _t_setup = time.perf_counter()
                logger.info("[CHAT][STREAM] RAG 태스크 시작 (session=%s)", sid)
                rag_sys = RAGSystem(session_id=sid)
                event_generator = await rag_sys.astream(query, model_name=model_name)
                handler = get_streaming_handler()
                event_stream = handler.stream_graph_events(
                    event_generator,
                    _remaining=remaining_chunks,
                )
                logger.info(
                    "[CHAT][STREAM] astream 준비 완료 (%.2fs) — 첫 청크 대기",
                    time.perf_counter() - _t_setup,
                )

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
                # 루프 셧다운 전에 스트림을 먼저 닫아 중첩 async generator의
                # GeneratorExit 경고("async generator ignored GeneratorExit")를
                # 막는다. aclose()가 도중에 실패해도 무시한다.
                if event_stream is not None:
                    with contextlib.suppress(Exception):
                        await event_stream.aclose()
                for c in remaining_chunks:
                    q.put(("chunk", c))
                q.put(("done", None))

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                task = loop.create_task(run())
                register_stream(run_id, loop, task)
                loop.run_until_complete(task)
            finally:
                unregister_stream(run_id)
                release_stream_slot()
                if not loop.is_closed():
                    loop.close()
        except asyncio.CancelledError:
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
    _first_chunk_received = False
    _stream_started = _clock()

    try:
        while True:
            # 절대 상한 타임아웃 (hard ceiling) 점검
            if (
                UI_STREAMING_HARD_TIMEOUT > 0
                and (_clock() - _stream_started) >= UI_STREAMING_HARD_TIMEOUT
            ):
                cancel_stream(run_id)
                raise TimeoutError(
                    "스트리밍 절대 상한 타임아웃이 초과되었습니다. "
                    "네트워크 및 모델 상태를 확인해주세요."
                ) from None

            setup_to = (
                UI_STREAMING_SETUP_TIMEOUT
                if UI_STREAMING_SETUP_TIMEOUT > 0
                else UI_STREAMING_TIMEOUT
            )
            effective = setup_to if not _first_chunk_received else UI_STREAMING_TIMEOUT
            try:
                msg_type, data = q.get(timeout=effective)
                _timeout_count = 0  # 성공 시 카운터 리셋
                if msg_type == "done":
                    break
                elif msg_type == "error":
                    raise data
                else:
                    _first_chunk_received = True
                    yield data
            except queue.Empty:
                _timeout_count += 1
                if _timeout_count >= _max_timeouts:
                    logger.error(
                        f"[CHAT] 스트리밍 타임아웃 ({_max_timeouts}회 연속): "
                        "백그라운드 작업이 응답하지 않음"
                    )
                    cancel_stream(run_id)  # 실시간 취소: task.cancel() 전달
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
        # 하드 취소: 스레드가 아직 살아 있을 때만 시도한다. 정상 완료 경로
        # (done 수신)에서는 이미 unregister된 run_id에 대해 stream_worker가
        # 불필요한 경고를 남기는 것을 방지한다.
        if t.is_alive():
            cancel_stream(run_id)  # real cancel via task.cancel()
            t.join(timeout=3)
            if t.is_alive():
                logger.warning(
                    f"[CHAT] Stream thread did not exit after cancel: {t.name}"
                )
            else:
                logger.debug(f"[CHAT] Stream thread cleanup: {t.name}")


def stream_content(query: str, model_name: str, session_id: str) -> Iterator[str]:
    """``st.write_stream`` 호환 content-only 브릿지 제너레이터.

    표준 스트리밍 패턴(``st.chat_message`` 안에서 ``st.write_stream``)에서
    토큰 본문만 점진적으로 흘리기 위해 ``stream_chunks``의 ``content``
    델타만 yield 한다. structured 모드에서는 final_answer delta만 yield —
    원시 JSON은 절대 흘러나오지 않는다. thought/documents/metrics/citations
    같은 부가 정보는 호출자가 별도로 소비할 수 있도록 ``stream_chunks``를
    직접 사용한다. 키(``final_answer``) 도착 전에는 아무것도 yield하지
    않는다(delta 도착 지연).

    취소/타임아웃 가드는 ``stream_chunks`` 내부 큐 브릿지가 그대로 보장한다.
    """
    raw_json_extractor = _FinalAnswerExtractor()
    for chunk in stream_chunks(query, model_name, session_id):
        if chunk.content:
            if getattr(chunk, "raw_json", False):
                delta = raw_json_extractor.feed(chunk.content)
                if delta:
                    yield delta
            else:
                yield chunk.content
