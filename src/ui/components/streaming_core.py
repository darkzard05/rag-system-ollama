"""Streaming core — merged logic from extractors, runtime, and state.

Phase 2B consolidation: ``streaming_extractors.py`` + ``streaming_runtime.py``
+ ``streaming_state.py`` → single cohesive module.

Backward compatibility: the old module names become thin re-export stubs that
import everything from this file, so all existing ``from ui.components
.streaming_extractors import …`` / ``streaming_runtime`` / ``streaming_state``
paths continue to work unchanged.

Layering (bottom-up):
  1. Extractors  — pure functions/classes, no Streamlit, no I/O
  2. Runtime     — queue-thread bridge, daemon thread, ``stream_chunks``
  3. State       — session helpers, ``_content_generator``, aux-state

The public facade is ``streaming.py`` which re-exports from this module.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import queue
import sys
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from api.stream_pipeline import StreamChunk, get_streaming_handler
from common.async_worker import AsyncWorker
from common.config import (
    UI_STREAMING_HARD_TIMEOUT,
    UI_STREAMING_SETUP_TIMEOUT,
    UI_STREAMING_TIMEOUT,
)
from common.final_answer import FinalAnswerExtractor as _FinalAnswerExtractor
from common.final_answer import extract_final_answer_delta as _extract_impl
from common.final_answer import recover_final_answer as _recover_impl
from common.stream_worker import (
    acquire_stream_slot,
    active_stream_count,
    cancel_stream,
    peek_stream_task,
    register_stream,
    release_stream_slot,
    unregister_stream,
)
from common.utils import doc_stable_id, fast_hash
from core.session import SessionManager
from ui.components.common import get_doc_metadata
from ui.components.reference_plan import note_stream_references
from ui.strings import t

logger = logging.getLogger(__name__)

_extract_final_answer_delta = _extract_impl
_recover_final_answer = _recover_impl

_clock = time.monotonic


def _format_stuck_thread_stack(thread: threading.Thread, limit: int = 8) -> str:
    """Read-only stack snapshot of a lingering stream thread.

    Runs only on the join-expiry path (never hot): tells the next
    investigation exactly where the thread is parked (sync-blocked loop
    vs slow teardown await) without touching the target thread.
    """
    if thread.ident is None:
        return "<no ident>"
    frame = sys._current_frames().get(thread.ident)
    if frame is None:
        return "<no frame>"
    return "".join(traceback.format_stack(frame, limit=limit))


def _iter_live_asyncgens(limit: int = 12) -> Any:
    """Yield ``(code_name, frame)`` for live suspended async generators.

    A task parked in ``async for`` exposes only one coroutine frame, and
    the iterator it parks on is a temporary invisible in frame locals.
    Scanning the GC for suspended async-generator objects reveals the
    whole downstream chain (``stream_graph_events`` -> ``_consumer`` ->
    retry -> breaker -> ``astream_events`` -> ...), i.e. the real await
    point. Repo and langchain frames first, stdlib noise last.
    """
    import gc as _gc
    import inspect as _inspect
    import os as _os

    found: list[tuple[int, Any]] = []
    for obj in _gc.get_objects():
        try:
            if not _inspect.isasyncgen(obj):
                continue
            ag_frame = getattr(obj, "ag_frame", None)
            if ag_frame is None:
                continue
            filename = ag_frame.f_code.co_filename
        except Exception:  # noqa: BLE001 - never break diagnostics
            continue
        lname = filename.replace(_os.sep, "/").lower()
        if "rag-system-ollama/src" in lname:
            prio = 0
        elif "langchain" in lname or "ollama" in lname:
            prio = 1
        else:
            prio = 2
        found.append((prio, ag_frame))
        if len(found) >= 60:
            break
    found.sort(key=lambda item: (item[0], item[1].f_code.co_name))
    for _, ag_frame in found[:limit]:
        yield ag_frame.f_code.co_name, ag_frame


def _describe_stream_task(run_id: str) -> str:
    """One-line task state for the lingering stream (done/cancelled/pending).

    Decides the branch: task done but thread alive (stuck *after* the
    task: unregister/release/loop.close), cancel delivered but ignored
    (teardown stall), or cancel never delivered (registry/loop mismatch).
    """
    try:
        task = peek_stream_task(run_id)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return f"<registry unreadable: {exc}>"
    if task is None:
        return "task=unregistered"
    try:
        _cancelling = (
            task.cancelling()  # type: ignore[attr-defined]
            if hasattr(task, "cancelling")
            else "n/a(py310)"
        )
        return (
            f"task=done={task.done()} cancelled={task.cancelled()} "
            f"cancelling={_cancelling}"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return f"<state unreadable: {exc}>"


def _format_stream_task_stack(run_id: str, limit: int = 12) -> str:
    """Read-only coroutine-stack snapshot of the lingering stream task.
    The thread stack of a parked event loop only shows idle ``select``;
    the task stack names the actual await point (e.g. a ``wait_for``-guarded
    model call vs the Ollama socket read). Best effort: a racing teardown
    may unregister or finish the task mid-inspection.
    """
    try:
        task = peek_stream_task(run_id)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return f"<registry unreadable: {exc}>"
    if task is None:
        return "<task already unregistered>"
    try:
        stack = task.get_stack(limit=limit)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return f"<stack unreadable: {exc}>"
    if not stack:
        return "<task has no stack (done or never started)>"
    try:
        parts: list[str] = []
        for st in stack:
            parts.extend(traceback.format_list(traceback.extract_stack(st)))
        parts.append("  ~~ live suspended asyncgens ~~\n")
        for _name, _frame in _iter_live_asyncgens():
            parts.append(
                f"  ~~ asyncgen {_name} "
                f"({_frame.f_code.co_filename}"
                f":{_frame.f_lineno}) ~~\n"
            )
            parts.extend(
                traceback.format_list(traceback.extract_stack(_frame, limit=3))
            )
        return "".join(parts)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never raise
        return f"<stack unformattable: {exc}>"


# ─────────────────────────────────────────────────────────────────────
# §1  EXTRACTORS — shared implementation lives in common.final_answer
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# §2  RUNTIME — queue-thread bridge + st.write_stream helper
# ─────────────────────────────────────────────────────────────────────

# Pre-final placeholder: structured (raw_json) mode emits fragments before the
# "final_answer" key arrives, and ``_FinalAnswerExtractor.feed`` returns "" for
# those. Yield this once so ``st.write_stream`` shows progress instead of
# silence. Reuses the existing ``_content_generator`` first-content string
# (streaming_state imports it — no new literals).
_STATUS_PLACEHOLDER = ""

# ■ 중지(StopException) 폴링 간격(초). 단일 q.get(timeout=...)은 C 레벨
# Condition.wait에 메인 스레드를 묶어두어 Streamlit의 네이티브 중지(trace 훅
# 기반 StopException)가 발화할 수 없다. 이 간격으로 폴링을 쪼개면 최대
# poll 초 내에 바이트코드 경계로 복귀해 중지 요청이 전달된다.
_STOP_POLL_INTERVAL_SEC = 0.1


def _q_get_with_stop_poll(
    q: queue.Queue[tuple[str, Any]], timeout: float
) -> tuple[str, Any]:
    """Chunk 대기를 짧은 폴링으로 쪼개 ■ 중지(StopException) 응답성을 확보한다.

    단일 ``q.get(timeout=...)``은 C 레벨 ``Condition.wait``에 메인 스레드를
    묶어 두기 때문에, Streamlit의 네이티브 중지(``submit_mode="stop"``)가
    바이트코드 경계 trace 훅으로 주입하는 ``StopException``이 발화하지
    못한다. ``_STOP_POLL_INTERVAL_SEC`` 간격으로 폴링을 분할하면 각 폴링
    사이마다 바이트코드 경계로 복귀하여 중지 요청이 즉시 전달되고, 절대
    대기 ``timeout``은 ``queue.Empty`` 의미(semantics)를 그대로 유지한다.
    """
    deadline = _clock() + timeout
    while True:
        remaining = deadline - _clock()
        if remaining <= 0:
            raise queue.Empty
        try:
            return q.get(timeout=min(_STOP_POLL_INTERVAL_SEC, remaining))
        except queue.Empty:
            continue


async def _acquire_slot_async(timeout: float) -> bool:
    """공유 루프를 막지 않도록 슬롯 획득을 워커 스레드로 오프로딩한다.

    태스크가 취소되더라도 실행 중인 OS 스레드가 슬롯을 획득(True)하면
    스레드 완료 시점에 반드시 release_stream_slot()을 호출하여 슬롯 누수를 원천 차단한다.
    """
    acquired_event = threading.Event()
    is_aborted = False
    abort_lock = threading.Lock()

    def _blocking_acquire() -> bool:
        got = acquire_stream_slot(timeout)
        with abort_lock:
            if got:
                if is_aborted:
                    # 호출자가 이미 취소하고 떠난 경우 즉시 반납
                    release_stream_slot()
                    return False
                acquired_event.set()
                return True
            return False

    fut = asyncio.ensure_future(asyncio.to_thread(_blocking_acquire))
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        with abort_lock:
            is_aborted = True
            if acquired_event.is_set():
                # 취소 시점에 이미 획득된 상태였다면 즉시 반납
                release_stream_slot()
        raise


def stream_chunks(
    query: str, model_name: str, session_id: str
) -> Iterator[StreamChunk]:
    """비동기 스트림을 동기 Streamlit 환경에서 소비하기 위한 브릿지 제너레이터

    스트림 RAG 작업은 프로세스 수명주기를 따르는 공용 ``AsyncWorker`` 싱글톤
    루프에서 실행한다. 매 턴 새 스레드/이벤트 루프를 만들면 풀링된 LLM
    인스턴스의 ``ollama.AsyncClient``가 들고 있는 httpx 커넥션 풀이 턴마다
    폐기되어 HTTP Keep-Alive가 무력화되고(매 질문 TCP 3-way handshake 재수행)
    TIME_WAIT 소켓이 누적된다. 안정된 단일 루프를 재사용하면 AsyncClient가
    유지되어 연결이 재사용되고 TTFT가 개선된다.

    스트림 내부의 무거운 동기 연산(임베딩/PDF I/O/리랭킹 등)은 모두
    ``asyncio.to_thread``로 오프로딩되어 있어 공유 루프를 블로킹하지 않는다.
    슬롯 대기 역시 ``to_thread``로 루프 밖에서 수행하며, 취소로 인해 실행기
    스레드가 뒤늦게 슬롯을 획득하면 done 콜백이 반납해 누수를 막는다.

    취소: 타임아웃/사용자 중단 시 ``cancel_stream(run_id)``가 실제 취소를
    요청한다(등록된 스트림에 한함 — 슬롯 대기 중 중단은 등록이 없으므로
    cancel을 건너뛰고 즉시 복귀한다). ``loop.call_soon_threadsafe(task.cancel)``
    로 이벤트 루프에 즉시 전달되어 협조적 ``_stop_event``와 무관하게 하드
    취소되며, 이어서 3초 바운드 대기로 태스크 종료를 기다린다.
    ``_stop_event``는 다음 await 지점에서 즉시 반응하는 협조적 백스톱으로
    함께 유지한다. 3초 대기가 타임아웃되면 워커 태스크가 남을 수 있지만
    슬롯은 finally에서 해제된다. 이때 경고에는 페이즈(setup/streaming)와
    활성 슬롯 수가 함께 기록된다.
    """
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
    _stop_event = threading.Event()
    run_id = f"stream-{session_id}-{uuid.uuid4().hex[:8]}"
    # 워커 태스크와 정리 finally가 공유하는 상태: 스트림 등록 여부(미등록
    # 취소의 오경고 방지), 첫 청크 put 시각(경고 자가진단용 페이즈 판별).
    _teardown_state: dict[str, Any] = {
        "registered": False,
        "first_put_at": None,
    }
    # done은 턴별 큐에 정확히 1회만 보장한다. 슬롯대기 중단·바깥 예외 등
    # run() finally에 닿지 않는 경로에서도 소비자가 hang 없이 종료되게 한다.
    _done_once = threading.Event()

    def _emit_done() -> None:
        if not _done_once.is_set():
            _done_once.set()
            with contextlib.suppress(Exception):
                q.put(("done", None))

    # 영속 AsyncWorker 싱글톤 루프를 재사용한다 — 풀링된 LLM 인스턴스의
    # AsyncClient(httpx 커넥션 풀)가 턴 사이에 유지되어 Keep-Alive가 산다.
    worker = AsyncWorker()
    worker_loop = worker.loop

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
                if _teardown_state["first_put_at"] is None:
                    _teardown_state["first_put_at"] = time.perf_counter()
        except asyncio.CancelledError:
            logger.info("[CHAT] 스트리밍 작업이 취소되었습니다")
        except Exception as e:
            logger.error(f"[CHAT] RAG 스트림 처리 오류: {e}", exc_info=True)
            q.put(("error", e))
        finally:
            # 워커 루프는 계속 살아 있으므로 중첩 async generator가 열린 채
            # 남지 않도록 명시적으로 닫는다. aclose()가 실패해도 무시한다.
            if event_stream is not None:
                with contextlib.suppress(Exception):
                    await event_stream.aclose()
            for c in remaining_chunks:
                q.put(("chunk", c))
            _emit_done()

    async def runner_wrapper() -> None:
        # 슬롯 대기는 중단 가능해야 한다: 등록 전 대기는 cancel_stream이
        # no-op이라 대기만 만료돼 확정 경고가 되므로(슬롯 고갈 연쇄),
        # _stop_event 폴링으로 즉시 빠져나간다. 전체 대기 상한 30초 유지.
        # 미등록 취소도 done으로 즉시 복귀해 소비자의 3회 Empty hang을 막는다.
        _slot_deadline = time.perf_counter() + 30
        _slot_ok = False
        try:
            while time.perf_counter() < _slot_deadline:
                if _stop_event.is_set():
                    logger.debug(f"[CHAT][STREAM] 슬롯 대기 중 중단 (run_id={run_id})")
                    return
                if await _acquire_slot_async(0.5):
                    _slot_ok = True
                    break
            if not _slot_ok:
                if _stop_event.is_set():
                    return
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
                return
            SessionManager.set_session_id(session_id or "default")
            try:
                current_task = asyncio.current_task()
                if current_task is not None:
                    register_stream(run_id, worker_loop, current_task)
                    _teardown_state["registered"] = True
                await run()
            except asyncio.CancelledError:
                logger.info("[CHAT] 스트리밍 작업이 취소되었습니다")
                raise
            except Exception as e:
                logger.error(f"[CHAT] 백그라운드 작업 오류: {e}", exc_info=True)
                with contextlib.suppress(Exception):
                    q.put(("error", e))
        finally:
            if _slot_ok:
                unregister_stream(run_id)
                release_stream_slot()
            _emit_done()

    # 별도 스레드/루프를 만들지 않고 영속 워커 루프에 태스크로 등록한다.
    # 제출 실패(워커 루프 미가동 등)는 큐로 error+done을 흘려 소비자가
    # 3회 Empty hang 없이 실패를 표면화하게 한다.
    future: concurrent.futures.Future[Any] | None = None
    _run_coro = runner_wrapper()
    try:
        future = worker.submit(_run_coro)
    except Exception as e:
        logger.error(f"[CHAT] 스트림 워커 제출 실패: {e}", exc_info=True)
        _run_coro.close()
        q.put(("error", e))
        _emit_done()

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
                if peek_stream_task(run_id) is not None:
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
                msg_type, data = _q_get_with_stop_poll(q, effective)
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
                    if peek_stream_task(run_id) is not None:
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
        # 하드 취소: 등록된 스트림에 대해서만 시도한다. 정상 완료 경로
        # (done 수신)에서는 이미 unregister된 run_id에 대해 stream_worker가
        # 불필요한 경고를 남기는 것을 방지한다. 슬롯 대기 중에 중단된 경우
        # (미등록)는 cancel 자체를 건너뛴다 — cancel_stream의 "no active
        # stream" 오경고를 막기 위해서다.
        if _teardown_state["registered"] and peek_stream_task(run_id) is not None:
            cancel_stream(run_id)  # real cancel via task.cancel()
        # 공유 워커 루프는 계속 실행되므로 스레드 join 대신 워커 태스크의
        # 완료 future를 3초 바운드로 기다린다. 완료되지 않으면 태스크/await
        # 지점 스냅샷으로 원인을 남긴다.
        if future is not None and not future.done():
            concurrent.futures.wait([future], timeout=3.0)
            if not future.done():
                _phase = (
                    "streaming"
                    if _teardown_state["first_put_at"] is not None
                    else "setup"
                )
                _task_parked = _format_stream_task_stack(run_id)
                _task_state = _describe_stream_task(run_id)
                logger.warning(
                    "[CHAT] Stream task did not finish after cancel: "
                    f"(phase={_phase}, teardown_wait=3.0s, "
                    f"active_slots={active_stream_count()}, "
                    f"{_task_state})\n"
                    f"task parked at:\n{_task_parked}"
                )
            else:
                logger.debug("[CHAT] Stream task cleanup complete")


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
    _placeholder_live = False
    _real_yielded = False
    for chunk in stream_chunks(query, model_name, session_id):
        if chunk.content:
            if getattr(chunk, "raw_json", False):
                delta = raw_json_extractor.feed(chunk.content)
                if delta:
                    _real_yielded = True
                    _placeholder_live = False
                    yield delta
                elif not _real_yielded:
                    _placeholder_live = True
            else:
                _real_yielded = True
                _placeholder_live = False
                yield chunk.content
    if _placeholder_live:
        yield _STATUS_PLACEHOLDER


# ─────────────────────────────────────────────────────────────────────
# §3  STATE — session helpers, content generator, aux-state
# ─────────────────────────────────────────────────────────────────────


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


def _target_message_dict(sid: str, msg_id: str) -> dict[str, Any] | None:
    """세션에서 msg_id에 해당하는 메시지 dict를 반환한다."""
    messages = SessionManager.get_messages(session_id=sid)
    return next((m for m in messages if m.get("msg_id") == msg_id), None)


_AUX_STATE_KEY = "stream_active_aux"


def _write_aux_state(sid: str, state: dict) -> None:
    """세션에 스트리밍 부가 정보(thought/metrics 등)를 기록한다."""
    SessionManager.set(_AUX_STATE_KEY, state, session_id=sid)


def _clear_aux_state(sid: str) -> None:
    """세션의 스트리밍 부가 정보를 정리한다."""
    SessionManager.set(_AUX_STATE_KEY, None, session_id=sid)


def _note_reference_prewarm(
    sid: str, msg_id: str, documents: list[Any], citations: list[Any]
) -> None:
    """Task3 prefetch 훅: 실패해도 스트리밍을 깨지 않는다."""
    try:
        note_stream_references(sid, msg_id, documents, citations)
    except (RuntimeError, TypeError, ValueError, AttributeError) as exc:
        logger.debug("[PREFETCH] prewarm 훅 생략: %s", exc)


def _citation_dedupe_key(cit: dict[str, Any]) -> tuple[str, Any, Any, str]:
    """보조 dedupe 키: doc_id(str 정규화) + page + section + text_span 해시."""
    return (
        str(cit.get("doc_id")),
        cit.get("page"),
        cit.get("section"),
        fast_hash(str(cit.get("text_span") or "")),
    )


def merge_citations_add(
    incr: list[dict[str, Any]],
    *,
    items: list[dict[str, Any]],
    base_index: int,
    arr_len: int,
    valid_ids: set[Any] | None = None,
) -> list[dict[str, Any]]:
    """증분 인용 병합: 검증 → base_index+i 업서트 → arr_len 절단 → 보조 키 dedupe.

    `valid_ids`가 주어지면 `doc_id`(str 정규화)가 집합에 속한 항목만
    병합한다 — 생산자(`_generate_pass`) 검증의 두 번째 방어망. `None`이면
    레거시 passthrough(기존 호출·테스트 호환).
    """
    if valid_ids is not None:
        items = [it for it in items if str(it.get("doc_id")) in valid_ids]
    merged = list(incr)
    for i, item in enumerate(items):
        idx = base_index + i
        if idx < len(merged):
            merged[idx] = item
        else:
            merged.append(item)
    if arr_len is not None and arr_len < len(merged):
        merged = merged[:arr_len]
    seen: set[tuple[str, Any, Any, str]] = set()
    deduped: list[dict[str, Any]] = []
    for cit in merged:
        key = _citation_dedupe_key(cit)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cit)
    return deduped


def _content_generator(
    query: str,
    model_name: str,
    sid: str,
    msg_id: str,
    on_status: Callable[[str, float], None] | None = None,
    on_aux: Callable[[dict[str, Any]], None] | None = None,
) -> Iterator[str]:
    """``st.write_stream`` 호환 content 제너레이터 + 부가 정보 누적.

    스트리밍 중 content만 yield하고, thought/metrics/citations 등은
    session_state에 기록하여 완료 후 렌더링에 사용한다.
    ``generation_cancel`` 시에도 부분 응답을 영속화한다.

    ``on_status``는 상태 전환 시에만 호출된다 (상태 텍스트 변경 또는 첫
    콘텐츠 yield 직후). ``on_aux``는 aux-state(thought/documents/metrics/
    citations/process_steps)가 변경된 청크에서만 호출된다 (연속 동일 상태는
    재호출되지 않는다). yield 내용은 두 콜백 유무와 무관하게 동일하다.
    """
    accumulated_parts: list[str] = []
    thought = ""
    documents: list[Any] = []
    metrics: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    process_steps: list[str] = []
    _content_started = False
    _placeholder_live = False
    _real_yielded = False
    raw_json_extractor = _FinalAnswerExtractor()

    _t_status_start = _clock()
    _t_last_token: float | None = None
    _last_reported_status: str | None = None
    _last_on_aux_snapshot: tuple[Any, ...] | None = None
    _ghost_logged = False
    incr_citations: list[dict[str, Any]] = []
    incr_attempt: int | None = None
    _prewarm_first_add_done = False
    _prewarm_elapsed_done = False
    _t_first_citation_paint: float | None = None
    _has_error = False

    def _ghost_flags() -> dict[str, Any]:
        has_ghost = bool(documents) and not citations
        return {
            "has_ghost": has_ghost,
            "ghost_msg_id": msg_id,
            "ghost_ts": _clock() if has_ghost else None,
        }

    def _report_status(text: str) -> None:
        nonlocal _last_reported_status
        if on_status is None:
            return
        if text == _last_reported_status:
            return
        _last_reported_status = text
        on_status(text, _clock() - _t_status_start)

    # 초기 aux state 기록 — 첫 청크 전 중단 시 빈 expander 방지
    _write_aux_state(
        sid,
        {
            "thought": "",
            "documents": [],
            "metrics": {},
            "citations": [],
            "process_steps": [],
            "complete": False,
            "has_ghost": False,
            "ghost_msg_id": msg_id,
            "ghost_ts": None,
        },
    )

    try:
        for chunk in stream_chunks(query, model_name, sid):
            if SessionManager.get("generation_cancel", False, session_id=sid):
                break
            if chunk.status:
                _report_status(chunk.status)
            _yielded_this_chunk = False
            if chunk.content:
                if not _content_started:
                    _content_started = True
                    if not chunk.status:
                        _report_status(_STATUS_PLACEHOLDER)
                        if not _real_yielded:
                            _placeholder_live = True
                if getattr(chunk, "raw_json", False):
                    delta = raw_json_extractor.feed(chunk.content)
                    if delta:
                        accumulated_parts.append(delta)
                    if not delta and raw_json_extractor.closed:
                        _report_status(t("status_citation_cleanup"))
                    if delta:
                        _real_yielded = True
                        _placeholder_live = False
                        _yielded_this_chunk = True
                        _t_last_token = _clock()
                        yield delta
                else:
                    accumulated_parts.append(chunk.content)
                    _real_yielded = True
                    _placeholder_live = False
                    _yielded_this_chunk = True
                    _t_last_token = _clock()
                    yield chunk.content
            if (
                not chunk.status
                and not _yielded_this_chunk
                and not _real_yielded
                and getattr(chunk, "raw_json", False)
            ):
                _report_status(_STATUS_PLACEHOLDER)
                _placeholder_live = True
            if chunk.thought:
                thought += chunk.thought
            if chunk.status and (
                not process_steps or process_steps[-1] != chunk.status
            ):
                process_steps.append(chunk.status)
            meta = chunk.metadata or {}
            if meta.get("documents"):
                documents = meta["documents"]
                if not citations and not _ghost_logged:
                    _ghost_logged = True
                    logger.info(
                        "[GHOST] ghost snapshot msg_id=%s (%d docs)",
                        msg_id,
                        len(documents),
                    )
                _note_reference_prewarm(sid, msg_id, documents, citations)
            if chunk.performance:
                metrics = chunk.performance
            citations_add = getattr(chunk, "citations_add", None)
            if citations_add:
                attempt = citations_add.get("attempt")
                if attempt != incr_attempt:
                    incr_citations = []
                    incr_attempt = attempt
                arr_len_raw = citations_add.get("arr_len")
                incr_citations = merge_citations_add(
                    incr_citations,
                    items=citations_add.get("items") or [],
                    base_index=int(citations_add.get("base_index", 0) or 0),
                    arr_len=int(arr_len_raw)
                    if isinstance(arr_len_raw, int)
                    else len(incr_citations),
                    valid_ids={doc_stable_id(d) for d in documents}
                    if documents
                    else None,
                )
                citations = incr_citations
                if _t_first_citation_paint is None:
                    _t_first_citation_paint = _clock()
                    _t_tok = (
                        _t_last_token if _t_last_token is not None else _t_status_start
                    )
                    logger.info(
                        "[PERF] first-citation-paint msg_id=%s took %.3fs",
                        msg_id,
                        _t_first_citation_paint - _t_tok,
                    )
                if not _prewarm_first_add_done:
                    _prewarm_first_add_done = True
                    logger.info("[PREWARM] submit reason=first_add msg_id=%s", msg_id)
                    _note_reference_prewarm(sid, msg_id, documents, citations)
                    if (_clock() - _t_status_start) > 5.0:
                        _prewarm_elapsed_done = True
                elif not _prewarm_elapsed_done and (_clock() - _t_status_start) > 5.0:
                    _prewarm_elapsed_done = True
                    logger.info("[PREWARM] submit reason=elapsed_5s msg_id=%s", msg_id)
                    _note_reference_prewarm(sid, msg_id, documents, citations)
                else:
                    logger.debug("[PREWARM] skip middle increment msg_id=%s", msg_id)
            if getattr(chunk, "citations", None):
                citations = chunk.citations or []
                incr_citations = list(citations)
                _t_citations = _clock()
                _t_tok = _t_last_token if _t_last_token is not None else _t_status_start
                logger.info(
                    "[PERF] last-token->citations-event msg_id=%s took %.3fs",
                    msg_id,
                    _t_citations - _t_tok,
                )
                logger.info("[PREWARM] submit reason=final msg_id=%s", msg_id)
                _note_reference_prewarm(sid, msg_id, documents, citations)
            if on_aux is not None:
                # [최적화] 스냅샷을 길이/버킷 비교로 저렴하게 구성하고, thought
                # 갱신 버킷을 50자로 완화하여 고속 스트리밍 중 세션 락 획득 및
                # dict/tuple 할당을 줄인다. _aux_state는 실제 갱신이 있을 때만
                # 늦게 생성(Lazy)한다.
                thought_bucket = len(thought) // 50 if thought else 0
                snapshot = (
                    len(process_steps),
                    len(documents),
                    len(citations),
                    thought_bucket,
                )
                if snapshot != _last_on_aux_snapshot:
                    _last_on_aux_snapshot = snapshot
                    _aux_state = {
                        "thought": thought,
                        "documents": documents,
                        "metrics": metrics,
                        "citations": citations,
                        "process_steps": process_steps,
                        "complete": False,
                        **_ghost_flags(),
                    }
                    _write_aux_state(sid, _aux_state)
                    on_aux(_aux_state)
        if _placeholder_live:
            yield _STATUS_PLACEHOLDER
    except Exception as exc:
        # 삼킴 금지: aux에는 원문을 보존하되, 로그에 traceback을 남긴다.
        # (UI 정책상 사용자에게는 제네릭 메시지만 노출한다.)
        _has_error = True
        logger.exception("[CHAT] content_generator 오류: %s", exc)
        _write_aux_state(
            sid,
            {
                "thought": thought,
                "documents": documents,
                "metrics": metrics,
                "citations": citations,
                "process_steps": process_steps,
                "error": str(exc),
                "content": "".join(accumulated_parts),
                "complete": False,
                **_ghost_flags(),
            },
        )
        raise
    finally:
        if not _has_error:
            _write_aux_state(
                sid,
                {
                    "thought": thought,
                    "documents": documents,
                    "metrics": metrics,
                    "citations": citations,
                    "process_steps": process_steps,
                    "content": "".join(accumulated_parts),
                    "complete": True,
                    **_ghost_flags(),
                },
            )
