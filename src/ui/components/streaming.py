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

from api.stream_events import chunk_to_stream_events
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
from common.utils import (
    extract_annotations_from_docs,
)
from core.session import SessionManager
from ui.components.common import get_doc_metadata

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
    _key_pos: list[int] | None = None,
) -> tuple[str, int]:
    """Incrementally pull the growing ``final_answer`` string value out of a
    partial JSON buffer.  Returns ``(delta_text, new_scan_pos)``.

    State-machine scanner that honours escape sequences (decoded, not
    literal), delimiter-aware close, pending-escape defer-to-next-call
    semantics, and optional key-position caching via *``_key_pos``* (an
    in-place ``list[int]``).
    """
    n = len(buffer)
    key = '"final_answer"'

    # --- Phase 1: key search (the ONLY str.find in the implementation) ---
    key_idx: int
    if _key_pos is not None and _key_pos[0] >= 0:
        key_idx = _key_pos[0]
    else:
        key_idx = buffer.find(key, 0)
        if _key_pos is not None:
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

    # --- Phase 3: find value's opening quote (char-by-char after colon) ---
    i = j + 1
    while i < n and buffer[i] in " \t\n\r":
        i += 1
    if i >= n or buffer[i] != '"':
        return "", start

    open_quote = i
    value_start = open_quote + 1

    # ``start`` counts decoded value chars already emitted (not buffer offset).
    emit_from = start

    # --- Phase 4: value scan with escape decode ---
    i = value_start
    delta_chars: list[str] = []
    pending_escape = False
    pending_u: str | None = None  # e.g. "\\u00" – holds incomplete \\uXXXX

    while i < n:
        ch = buffer[i]

        # Deferred escape from previous call --------------------------------
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
            # Unknown escape: emit both chars literally.
            delta_chars.append("\\")
            delta_chars.append(ch)
            i += 1
            continue

        # Deferred \\uXXXX from previous call -------------------------------
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
                # Non-hex terminates \\u: emit literal and reprocess ch.
                delta_chars.append(pending_u)
                pending_u = None
                continue

        # Normal characters --------------------------------------------------
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
                # Unknown escape: emit both literally.
                delta_chars.append(ch)
                delta_chars.append(nxt)
                i += 2
                continue
            # Trailing backslash at end-of-buffer: defer.
            pending_escape = True
            i += 1
            continue

        if ch == '"':
            # Delimiter-aware close: value terminates only when the next char
            # is a JSON delimiter or the buffer is exhausted.
            nxt = buffer[i + 1] if i + 1 < n else ""
            if nxt in _DELIMITERS or nxt == "":
                new_delta = "".join(delta_chars)[emit_from:]
                return new_delta, len(delta_chars)
            # Unescaped inner quote – treat as content.
            delta_chars.append(ch)
            i += 1
            continue

        delta_chars.append(ch)
        i += 1

    # Value still open – emit only chars not yet delivered; pending escape/u
    # are held for the next call and NOT emitted in this delta.
    new_delta = "".join(delta_chars)[emit_from:]
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
    """structured 모드 raw_json 청크 → final_answer 증분 추출 헬퍼.

    consume_stream_into_message / _content_generator / stream_content 세
    소비자가 공유한다. feed(content)마다 내부 blob에 원문을 순서대로
    누적하고, _extract_final_answer_delta의 상태 머신(스캔 위치 + 키 위치
    캐시)으로 새로 디코드된 delta만 반환한다. 모든 delta의 연결(join)은
    final_answer 값과 정확히 같다(순서 불변, \\uXXXX/CJK 디코드 포함).
    스트림/제너레이터당 인스턴스 1개를 만들고 chunk마다 feed() 호출.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []
        self._scan_pos = 0
        self._key_pos: list[int] = [-1]

    def feed(self, content: str) -> str:
        """raw_json 청크의 content를 누적하고 새 final_answer delta를 반환.

        키 미발견 등 아직 출력할 문자가 없으면 ""를 반환한다. 호출자는
        yield 직전에 ``if delta:`` 가드로 빈 문자열을 걸러야 한다.
        """
        self._parts.append(content)
        blob = "".join(self._parts)
        delta, self._scan_pos = _extract_final_answer_delta(
            blob, self._scan_pos, self._key_pos
        )
        return delta


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


def _target_message_dict(sid: str, msg_id: str) -> dict[str, Any] | None:
    """세션에서 msg_id에 해당하는 메시지 dict를 반환한다."""
    messages = SessionManager.get_messages(session_id=sid)
    return next((m for m in messages if m.get("msg_id") == msg_id), None)


# ---------------------------------------------------------------------------
# st.write_stream 호환 content 제너레이터 + 부가 정보 누적
# ---------------------------------------------------------------------------

_AUX_STATE_KEY = "stream_active_aux"


def _write_aux_state(sid: str, state: dict) -> None:
    """세션에 스트리밍 부가 정보(thought/metrics 등)를 기록한다."""
    SessionManager.set(_AUX_STATE_KEY, state, session_id=sid)


def _clear_aux_state(sid: str) -> None:
    """세션의 스트리밍 부가 정보를 정리한다."""
    SessionManager.set(_AUX_STATE_KEY, None, session_id=sid)


def _content_generator(
    query: str, model_name: str, sid: str, msg_id: str
) -> Iterator[str]:
    """``st.write_stream`` 호환 content 제너레이터 + 부가 정보 누적.

    스트리밍 중 content만 yield하고, thought/metrics/citations 등은
    session_state에 기록하여 완료 후 렌더링에 사용한다.
    ``generation_cancel`` 시에도 부분 응답을 영속화한다.
    """
    accumulated = ""
    thought = ""
    documents: list[Any] = []
    metrics: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    process_steps: list[str] = []
    raw_json_extractor = _FinalAnswerExtractor()

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
            if chunk.content:
                if getattr(chunk, "raw_json", False):
                    delta = raw_json_extractor.feed(chunk.content)
                    accumulated += delta
                    if delta:
                        yield delta
                else:
                    accumulated += chunk.content
                    yield chunk.content
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
            _write_aux_state(
                sid,
                {
                    "thought": thought,
                    "documents": documents,
                    "metrics": metrics,
                    "citations": citations,
                    "process_steps": process_steps,
                    "complete": False,
                },
            )
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
                "complete": True,
            },
        )
