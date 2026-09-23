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
import contextlib
import logging
import queue
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from typing import Any

from streamlit.runtime.scriptrunner import add_script_run_ctx

from api.stream_pipeline import StreamChunk, get_streaming_handler
from common.config import (
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
from ui.components.common import get_doc_metadata

logger = logging.getLogger(__name__)

_clock = time.monotonic

# ─────────────────────────────────────────────────────────────────────
# §1  EXTRACTORS — pure final_answer delta state machine
# ─────────────────────────────────────────────────────────────────────

_ESCAPE_DECODE: dict[str, str] = {
    "n": "\n",
    '"': '"',
    "\\": "\\",
    "t": "\t",
    "/": "/",
}

_DELIMITERS = (":", ",", "}", "]")


def _extract_final_answer_delta(
    buffer: str,
    start: int,
    _key_pos: list[Any] | None = None,
) -> tuple[str, int]:
    """Incrementally pull the growing ``final_answer`` string value out of a
    partial JSON buffer. Returns ``(delta_text, new_scan_pos)``.

    O(1) amortized incremental scanner preserving all escape sequences
    (\\n, \\", \\uXXXX) and delimiter-aware close semantics.
    """
    n = len(buffer)
    key = '"final_answer"'

    # --- Phase 1: key search ---
    key_idx: int
    if _key_pos is not None and len(_key_pos) > 0 and _key_pos[0] >= 0:
        key_idx = _key_pos[0]
    else:
        key_idx = buffer.find(key, 0)
        if _key_pos is not None and len(_key_pos) > 0:
            _key_pos[0] = key_idx
        if key_idx == -1:
            return "", 0

    # --- Phase 2: verify key is followed by optional ws + colon ---
    after_key = key_idx + len(key)
    j = after_key
    while j < n and buffer[j] in " \t\n\r":
        j += 1
    if j >= n or buffer[j] != ":":
        return "", start

    # --- Phase 3: find value's opening quote ---
    i = j + 1
    while i < n and buffer[i] in " \t\n\r":
        i += 1
    if i >= n or buffer[i] != '"':
        return "", start

    open_quote = i
    value_start = open_quote + 1

    # --- Phase 4: incremental value scan ---
    # _key_pos 구조: [key_idx, raw_pos, pending_escape, pending_u, closed]
    is_incremental = _key_pos is not None and len(_key_pos) >= 5

    if is_incremental:
        assert _key_pos is not None  # is_incremental이 None 아님을 보장
        if _key_pos[4]:  # 이미 닫힌 경우
            return "", start
        i = _key_pos[1] if _key_pos[1] >= value_start else value_start
        pending_escape = bool(_key_pos[2])
        pending_u = _key_pos[3]
    else:
        i = value_start
        pending_escape = False
        pending_u = None

    delta_chars: list[str] = []

    while i < n:
        ch = buffer[i]

        # Deferred escape from previous call
        if pending_escape:
            pending_escape = False
            decoded = _ESCAPE_DECODE.get(ch)
            if decoded is not None:
                delta_chars.append(decoded)
                i += 1
                continue
            if ch == "u":
                pending_u = "\\u"
                i += 1
                continue
            delta_chars.append("\\")
            delta_chars.append(ch)
            i += 1
            continue

        # Deferred \uXXXX from previous call
        if pending_u is not None:
            if ch in "0123456789abcdefABCDEF" and len(pending_u) < 6:
                pending_u += ch
                i += 1
                if len(pending_u) == 6:
                    hex_str = pending_u[2:]
                    try:
                        delta_chars.append(chr(int(hex_str, 16)))
                    except ValueError:
                        delta_chars.append(pending_u)
                    pending_u = None
                continue
            else:
                delta_chars.append(pending_u)
                pending_u = None
                continue

        # Normal characters
        if ch == "\\":
            if i + 1 < n:
                nxt = buffer[i + 1]
                decoded = _ESCAPE_DECODE.get(nxt)
                if decoded is not None:
                    delta_chars.append(decoded)
                    i += 2
                    continue
                if nxt == "u":
                    pending_u = "\\u"
                    i += 2
                    continue
                delta_chars.append(ch)
                delta_chars.append(nxt)
                i += 2
                continue
            pending_escape = True
            i += 1
            continue

        if ch == '"':
            nxt = buffer[i + 1] if i + 1 < n else ""
            if nxt in _DELIMITERS or nxt == "":
                if is_incremental:
                    assert _key_pos is not None  # is_incremental이 None 아님을 보장
                    _key_pos[4] = 1  # closed = True
                if is_incremental:
                    new_delta = "".join(delta_chars)
                    return new_delta, start + len(new_delta)
                else:
                    new_delta = "".join(delta_chars)[start:]
                    return new_delta, len(delta_chars)
            delta_chars.append(ch)
            i += 1
            continue

        delta_chars.append(ch)
        i += 1

    if is_incremental:
        assert _key_pos is not None  # is_incremental이 None 아님을 보장
        _key_pos[1] = i
        _key_pos[2] = 1 if pending_escape else 0
        _key_pos[3] = pending_u
        new_delta = "".join(delta_chars)
        return new_delta, start + len(new_delta)
    else:
        new_delta = "".join(delta_chars)[start:]
        return new_delta, len(delta_chars)


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


class _FinalAnswerExtractor:
    """structured 모드 raw_json 청크 → final_answer 증분 추출 헬퍼 (O(1) amortized)."""

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._blob: str = ""
        self._scan_pos = 0
        # [key_idx, raw_pos, pending_escape, pending_u, closed]
        self._key_pos: list[Any] = [-1, -1, 0, None, 0]

    def feed(self, content: str) -> str:
        """raw_json 청크의 content를 누적하고 새 final_answer delta를 즉시 반환."""
        self._parts.append(content)
        self._blob += content  # CPython in-place realloc 최적화 적용
        delta, self._scan_pos = _extract_final_answer_delta(
            self._blob, self._scan_pos, self._key_pos
        )
        return delta


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
    q: queue.Queue[tuple[str, Any]] = queue.Queue()
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
    accumulated = ""
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
    _last_reported_status: str | None = None
    _last_on_aux_snapshot: tuple[Any, ...] | None = None

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
                    accumulated += delta
                    if delta:
                        _real_yielded = True
                        _placeholder_live = False
                        _yielded_this_chunk = True
                        yield delta
                else:
                    accumulated += chunk.content
                    _real_yielded = True
                    _placeholder_live = False
                    _yielded_this_chunk = True
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
            if chunk.performance:
                metrics = chunk.performance
            if getattr(chunk, "citations", None):
                citations = chunk.citations or []
            _aux_state = {
                "thought": thought,
                "documents": documents,
                "metrics": metrics,
                "citations": citations,
                "process_steps": process_steps,
                "complete": False,
            }
            if on_aux is not None:
                # thought가 진행되는 동안 40자 단위 또는 변경 시점에 UI를 점진적 갱신
                thought_bucket = len(thought) // 40 if thought else 0
                snapshot = (
                    tuple(process_steps),
                    tuple(documents),
                    tuple(metrics.items()),
                    tuple(citations),
                    thought_bucket,
                )
                if snapshot != _last_on_aux_snapshot:
                    _last_on_aux_snapshot = snapshot
                    _write_aux_state(sid, _aux_state)
                    on_aux(_aux_state)
        if _placeholder_live:
            yield _STATUS_PLACEHOLDER
    except Exception as exc:
        _write_aux_state(
            sid,
            {
                "thought": thought,
                "documents": documents,
                "metrics": metrics,
                "citations": citations,
                "process_steps": process_steps,
                "error": str(exc),
                "content": accumulated,
                "complete": False,
            },
        )
        raise
    finally:
        _write_aux_state(
            sid,
            {
                "thought": thought,
                "documents": documents,
                "metrics": metrics,
                "citations": citations,
                "process_steps": process_steps,
                "content": accumulated,
                "complete": True,
            },
        )
