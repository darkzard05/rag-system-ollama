"""Stable measurement contract for RAG query execution.

The evaluation harness and the UI must agree on what counts as the user-visible
answer. This module owns the single normalization point and exposes a versioned
measurement object. Pipeline internals (prompts, chunking, retrieval, graph
nodes) may change without touching the evaluation script.

Contract rule: consumers read only ``QueryMeasurement`` fields. If the event
stream or answer semantics change, update this adapter and bump
``QUERY_CONTRACT_VERSION``; the evaluator stays unchanged.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Protocol, cast

from api.stream_pipeline import get_streaming_handler
from common.final_answer import FinalAnswerExtractor

__all__ = [
    "QUERY_CONTRACT_VERSION",
    "QueryFlags",
    "QueryMeasurement",
    "QueryRAG",
    "QueryTiming",
    "run_query_measurement",
]

QUERY_CONTRACT_VERSION = 1


class QueryRAG(Protocol):
    """Minimal RAG surface needed by the measurement adapter."""

    async def astream(
        self, query: str, model_name: str | None = None
    ) -> AsyncIterator[tuple[str, Any]]: ...


@dataclass(frozen=True)
class QueryTiming:
    """Wall-clock timings in seconds.

    ``ttft_s`` keeps the legacy evaluator meaning (after ``astream`` returns to
    the first raw content chunk). ``ttft_e2e_s`` includes setup and is the
    user-perceived first-token latency. ``ttft_display_s`` measures the first
    user-visible answer delta, which may arrive after raw JSON scaffolding.
    """

    setup_s: float
    ttft_s: float | None
    ttft_e2e_s: float | None
    ttft_display_s: float | None
    ttft_display_e2e_s: float | None
    generation_s: float | None
    total_s: float


@dataclass(frozen=True)
class QueryFlags:
    """Non-fatal conditions that change how measurements should be read."""

    retrieval_only: bool
    raw_json_seen: bool
    display_extraction_failed: bool
    eval_count_missing: bool


@dataclass(frozen=True)
class QueryMeasurement:
    """Versioned result consumed by evaluation and diagnostics."""

    contract_version: int
    display_answer: str
    raw_answer: str
    documents: list[Any]
    performance: dict[str, Any]
    timing: QueryTiming
    flags: QueryFlags

    @property
    def answer_chars(self) -> int:
        return len(self.display_answer)

    @property
    def raw_answer_chars(self) -> int:
        return len(self.raw_answer)

    @property
    def eval_count(self) -> int | None:
        value = self.performance.get("eval_count")
        return value if isinstance(value, int) else None

    @property
    def tps(self) -> float | None:
        eval_count = self.eval_count
        generation_s = self.timing.generation_s
        if eval_count is None or generation_s is None or generation_s <= 0:
            return None
        return eval_count / generation_s

    @property
    def parse_failed(self) -> bool:
        return bool(self.performance.get("parse_failed"))

    @property
    def echo_detected(self) -> bool:
        return bool(self.performance.get("echo_detected"))

    @property
    def over_budget(self) -> bool:
        return bool(self.performance.get("over_budget"))


async def run_query_measurement(
    rag: QueryRAG,
    query: str,
    model_name: str | None = None,
    *,
    retrieval_only: bool = False,
) -> QueryMeasurement:
    """Execute one query and return the stable measurement contract.

    This is the only place the evaluator touches the streaming pipeline. The
    returned ``display_answer`` is derived with the same extractor used by the
    UI, so judge scores and answer lengths reflect what users actually see.
    """
    started = time.perf_counter()
    event_stream = await rag.astream(query, model_name=model_name)
    setup_s = time.perf_counter() - started
    stream_started = time.perf_counter()

    extractor = FinalAnswerExtractor()
    raw_parts: list[str] = []
    display_parts: list[str] = []
    documents: list[Any] = []
    performance: dict[str, Any] = {}
    first_raw: float | None = None
    last_raw: float | None = None
    first_display: float | None = None
    raw_json_seen = False

    handler = get_streaming_handler()
    event_generator = cast(
        AsyncGenerator[Any, None], handler.stream_graph_events(event_stream)
    )
    async with aclosing(event_generator) as stream:
        async for chunk in stream:
            now = time.perf_counter()

            if chunk.content:
                raw_parts.append(chunk.content)
                if first_raw is None:
                    first_raw = now
                last_raw = now

                if chunk.raw_json:
                    raw_json_seen = True
                    delta = extractor.feed(chunk.content)
                else:
                    delta = chunk.content
                if delta:
                    display_parts.append(delta)
                    if first_display is None:
                        first_display = now

            if chunk.metadata and "documents" in chunk.metadata:
                documents = list(chunk.metadata["documents"] or [])
                if retrieval_only:
                    break

            if chunk.performance:
                performance.update(chunk.performance)

    total_s = time.perf_counter() - started
    raw_answer = "".join(raw_parts)
    display_answer = "".join(display_parts)

    ttft_s = first_raw - stream_started if first_raw is not None else None
    ttft_display_s = (
        first_display - stream_started if first_display is not None else None
    )
    generation_s = (
        last_raw - first_raw if first_raw is not None and last_raw is not None else None
    )
    eval_count = performance.get("eval_count")

    return QueryMeasurement(
        contract_version=QUERY_CONTRACT_VERSION,
        display_answer=display_answer,
        raw_answer=raw_answer,
        documents=documents,
        performance=performance,
        timing=QueryTiming(
            setup_s=setup_s,
            ttft_s=ttft_s,
            ttft_e2e_s=setup_s + ttft_s if ttft_s is not None else None,
            ttft_display_s=ttft_display_s,
            ttft_display_e2e_s=(
                setup_s + ttft_display_s if ttft_display_s is not None else None
            ),
            generation_s=generation_s,
            total_s=total_s,
        ),
        flags=QueryFlags(
            retrieval_only=retrieval_only,
            raw_json_seen=raw_json_seen,
            display_extraction_failed=raw_json_seen and not display_answer,
            eval_count_missing=eval_count is None,
        ),
    )
