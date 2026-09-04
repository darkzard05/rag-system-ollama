"""Timing helpers for the RAG graph pipeline.

Extracted from ``graph_builder.py`` (Zone B) to keep that module under the
250-LOC ceiling.  These functions are pure timing bookkeeping — they do NOT
depend on ``graph_builder`` itself, so the import direction is strictly:
``graph_builder`` → ``_grading_glue`` (no cycle).
"""

import contextvars
import logging
from typing import Any

from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage timing buffer
# ---------------------------------------------------------------------------
# The buffer is a mutable dict captured into a local at generate entry so
# retries (which re-run retrieve/grade) accumulate correctly.  A speculative
# generate task created via ``asyncio.ensure_future`` inherits a shallow copy
# of the context that references the SAME dict, so accumulation across the
# main and speculative tasks is preserved while different queries (different
# tasks) keep distinct dicts.
# ---------------------------------------------------------------------------

_stage_timing_var: contextvars.ContextVar[dict[str, float]] = contextvars.ContextVar(
    "_stage_timing_var"
)


def _reset_stage_timings() -> None:
    """Clear per-query stage timing buffer (called at preprocess start)."""
    _stage_timing_var.set(
        {
            "preprocess_ms": 0.0,
            "retrieve_ms": 0.0,
            "grade_ms": 0.0,
            "generate_total_ms": 0.0,
            "ttft_ms": 0.0,
        }
    )


def _add_stage_ms(stage: str, ms: float) -> None:
    """Accumulate a stage duration into the per-query buffer."""
    stages = _stage_timing_var.get(None)
    if stages is None:
        _reset_stage_timings()
        stages = _stage_timing_var.get()
    stages[stage] = stages.get(stage, 0.0) + float(ms)


def _enter_stage(operation_type: OperationType, **metadata: Any) -> Any:
    """Begin a tracked operation, returning the OperationTracker context manager.

    Mirrors the existing ``with get_performance_monitor().track_operation(...)``
    pattern used in ``chunking.py`` but exposes the tracker so callers can exit
    it without re-indenting large node bodies.
    """
    return get_performance_monitor().track_operation(operation_type, dict(metadata))


def _emit_query_timing(timings: dict[str, float]) -> None:
    """Emit the single consolidated per-query timing line."""
    logger.info(
        f"[QUERY][TIMING] preprocess_ms={timings.get('preprocess_ms', 0.0):.1f} "
        f"retrieve_ms={timings.get('retrieve_ms', 0.0):.1f} "
        f"grade_ms={timings.get('grade_ms', 0.0):.1f} "
        f"generate_total_ms={timings.get('generate_total_ms', 0.0):.1f} "
        f"ttft_ms={timings.get('ttft_ms', 0.0):.1f}"
    )
