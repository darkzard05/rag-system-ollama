"""Streaming state helpers — build/side-state/content generator.

Extracted from ``streaming.py`` (verbatim, no logic change).
Cohesive unit: ``_build_process``, ``_target_message_dict``,
aux-state (``_AUX_STATE_KEY`` / ``_write_aux_state`` /
``_clear_aux_state``) and ``_content_generator``. Depends on
``streaming_extractors`` and ``streaming_runtime`` (no back-import).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from typing import Any

from core.session import SessionManager
from ui.components.common import get_doc_metadata
from ui.components.streaming_extractors import _FinalAnswerExtractor
from ui.components.streaming_runtime import _clock, stream_chunks

logger = logging.getLogger(__name__)


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
) -> Iterator[str]:
    """``st.write_stream`` 호환 content 제너레이터 + 부가 정보 누적.

    스트리밍 중 content만 yield하고, thought/metrics/citations 등은
    session_state에 기록하여 완료 후 렌더링에 사용한다.
    ``generation_cancel`` 시에도 부분 응답을 영속화한다.

    ``on_status``는 상태 전환 시에만 호출된다 (상태 텍스트 변경 또는 첫
    콘텐츠 yield 직후). yield 내용은 ``on_status`` 유무와 무관하게 동일하다.
    """
    accumulated = ""
    thought = ""
    documents: list[Any] = []
    metrics: dict[str, Any] = {}
    citations: list[dict[str, Any]] = []
    process_steps: list[str] = []
    _content_started = False
    raw_json_extractor = _FinalAnswerExtractor()

    _t_status_start = _clock()
    _last_reported_status: str | None = None

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
            if chunk.content:
                if not _content_started:
                    _content_started = True
                    if not chunk.status:
                        _report_status("응답 생성 중...")
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
