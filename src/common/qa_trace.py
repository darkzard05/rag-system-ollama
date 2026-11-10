"""Structured Q&A trace logging for offline evaluation.

Each completed turn is written as one JSON object per line to a dedicated
``.jsonl`` file. Keeping evaluation data out of ``app.log`` avoids the
single-line formatter, log rotation, and log-injection problems that a raw
multi-line answer would introduce.

The writer runs on a dedicated background thread so the request path only
enqueues a record and never blocks on disk I/O.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import logging
import queue
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from common import config as _cfg

logger = logging.getLogger(__name__)

QA_TRACE_SCHEMA_VERSION = "v1"

_QUEUE_MAXSIZE = 1000

_queue: queue.Queue[dict[str, Any] | None] | None = None
_listener: _QATraceWriter | None = None
_lock = threading.Lock()
_dropped = 0


def _doc_id(doc: Any) -> str:
    """Return a stable identifier for a document or chunk if one is present."""
    keys = ("stable_id", "doc_id", "chunk_id", "id", "source")
    meta: Any = None
    if isinstance(doc, Mapping):
        meta = doc.get("metadata")
        if not isinstance(meta, Mapping):
            meta = doc
    else:
        meta = getattr(doc, "metadata", None)
    if isinstance(meta, Mapping):
        for key in keys:
            value = meta.get(key)
            if value:
                return str(value)
    return ""


def _as_text(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def build_qa_record(
    *,
    source: str,
    session_id: str,
    query: str,
    answer: str,
    model: str | None = None,
    citations: list[Any] | None = None,
    confidence: float = 0.0,
    prompt_version: str = "",
    is_cached: bool = False,
    regenerated: str = "",
    parse_failed: bool = False,
    echo_detected: bool = False,
    over_budget: bool = False,
    intent: str = "",
    relevant_docs: list[Any] | None = None,
    performance: Mapping[str, Any] | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    """Normalize a completed turn into a schema-versioned JSONL record."""
    answer_text = _as_text(answer)
    max_chars = _cfg.QA_LOG_MAX_ANSWER_CHARS
    if max_chars > 0:
        answer_text = answer_text[:max_chars]
    perf = dict(performance or {})
    return {
        "schema_version": QA_TRACE_SCHEMA_VERSION,
        "timestamp": time.time(),
        "source": source,
        "session_id": session_id,
        "file_name": file_name,
        "query": query,
        "answer": answer_text,
        "citations": citations or [],
        "confidence": confidence,
        "prompt_version": prompt_version,
        "is_cached": is_cached,
        "regenerated": regenerated,
        "parse_failed": parse_failed,
        "echo_detected": echo_detected,
        "over_budget": over_budget,
        "intent": intent,
        "relevant_doc_ids": [_doc_id(doc) for doc in (relevant_docs or [])],
        "input_tokens": perf.get("input_token_count", 0),
        "output_tokens": perf.get("output_token_count", 0),
        "generate_ms": perf.get("generate_ms", 0.0),
        "ttft_ms": perf.get("ttft_ms", 0.0),
        "model": model or "",
    }


class _QATraceWriter(threading.Thread):
    """Drains the queue and appends JSONL records with size-based rotation."""

    def __init__(
        self,
        path: Path,
        q: queue.Queue[dict[str, Any] | None],
        max_bytes: int,
        backups: int,
    ) -> None:
        super().__init__(name="qa-trace-writer", daemon=True)
        self._path = path
        self._queue = q
        self._max_bytes = max_bytes
        self._backups = backups
        self._fh: TextIO | None = None

    def run(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self._path.open("a", encoding="utf-8")
        except OSError as exc:  # noqa: BLE001 - writer must not crash the app
            logger.error("[QA_TRACE] 파일 열기 실패: %s", exc)
            return
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                self._write(item)
        finally:
            if self._fh is not None:
                with contextlib.suppress(OSError):
                    self._fh.close()

    def _write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            return
        try:
            line = json.dumps(record, ensure_ascii=False, default=str)
            self._rotate_if_needed()
            if self._fh is None:
                return
            self._fh.write(line + "\n")
            self._fh.flush()
        except OSError as exc:  # noqa: BLE001 - never raise into the writer loop
            logger.error("[QA_TRACE] 기록 실패: %s", exc)

    def _rotate_if_needed(self) -> None:
        if self._fh is None or self._max_bytes <= 0:
            return
        try:
            size = self._fh.tell()
        except OSError:
            return
        if size < self._max_bytes:
            return
        self._fh.close()
        self._fh = None
        try:
            if self._backups > 0:
                for index in range(self._backups - 1, 0, -1):
                    src = self._path.with_name(f"{self._path.name}.{index}")
                    dst = self._path.with_name(f"{self._path.name}.{index + 1}")
                    if src.exists():
                        src.replace(dst)
                self._path.replace(self._path.with_name(f"{self._path.name}.1"))
            else:
                self._path.unlink(missing_ok=True)
        except OSError as exc:  # noqa: BLE001 - keep writing even if rotate fails
            logger.warning("[QA_TRACE] 로테이션 실패: %s", exc)
        try:
            self._fh = self._path.open("a", encoding="utf-8")
        except OSError as exc:  # noqa: BLE001
            logger.error("[QA_TRACE] 로테이션 후 재열기 실패: %s", exc)


def _resolve_path() -> Path:
    path = Path(_cfg.QA_LOG_PATH)
    if not path.is_absolute():
        path = _cfg.PROJECT_ROOT / path
    return path


def _ensure_started() -> queue.Queue[dict[str, Any] | None]:
    global _queue, _listener
    with _lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
            _listener = _QATraceWriter(
                _resolve_path(),
                _queue,
                _cfg.QA_LOG_ROTATE_MAX_BYTES,
                _cfg.QA_LOG_ROTATE_BACKUPS,
            )
            _listener.start()
            atexit.register(_shutdown)
        return _queue


def record_qa_turn(**fields: Any) -> None:
    """Enqueue one completed turn. No-op when ``QA_LOG_ENABLED`` is false."""
    if not _cfg.QA_LOG_ENABLED:
        return
    global _dropped
    record = build_qa_record(**fields)
    q = _ensure_started()
    try:
        q.put_nowait(record)
    except queue.Full:
        _dropped += 1
        if _dropped == 1 or _dropped % 100 == 0:
            logger.warning("[QA_TRACE] 큐 포화 — 기록 드롭 누적=%d", _dropped)


def _shutdown() -> None:
    global _queue, _listener
    with _lock:
        q = _queue
        listener = _listener
    if q is None or listener is None:
        return
    with contextlib.suppress(queue.Full):
        q.put_nowait(None)
    listener.join(timeout=2.0)


def reset_for_tests() -> None:
    """Stop the writer and clear module state (test isolation helper)."""
    global _queue, _listener, _dropped
    _shutdown()
    with _lock:
        _queue = None
        _listener = None
        _dropped = 0
