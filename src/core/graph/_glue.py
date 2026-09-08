"""Shared glue helpers for the RAG graph pipeline.

Extracted from ``graph_builder.py`` (Step 1 of the graph-builder split plan).
These helpers are used by multiple LangGraph nodes but do NOT depend on
``graph_builder`` itself, so the import direction is strictly:
``graph_builder`` → ``_glue`` (no cycle).  ``graph_builder`` re-exports
these symbols for backward compatibility.
"""

import asyncio
import logging
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from api.schemas import GraphState
from core.graph._speculative_gen import (
    _spec_generate_events,
    _spec_overlap_enabled,
    _spec_registry,
    _SpecEvent,
    _SpecGenerate,
)
from core.session import SessionManager

logger = logging.getLogger(__name__)


def _get_session_id(config: RunnableConfig | None = None) -> str:
    """Extract session_id from RunnableConfig or fall back to current context."""
    if config and "configurable" in config:
        sid = config["configurable"].get("session_id")
        if sid:
            return sid
        # config는 전달됐지만 session_id가 누락된 경우 — 전파 버그 신호.
        # 정상적인 "default" 세션 사용(비 Streamlit 모드)은 조용히 폴백합니다.
        logger.warning("[GRAPH] config에 session_id 누락 — 암묵적 세션 폴백")
    return SessionManager.get_session_id()


async def _dispatch_event(
    name: str,
    data: dict[str, Any],
    *,
    writer: StreamWriter | None,
    config: RunnableConfig,
) -> None:
    """Emit a custom event, or buffer it if a speculative generate is active.

    Buffering ensures a speculative (possibly-to-be-discarded) generate never
    leaks partial output to the user. The buffer is scoped to the current
    coroutine context via a ContextVar.
    """
    buf = _spec_generate_events.get(None)
    if buf is not None:
        buf.append(_SpecEvent(name=name, data=data, config=config))
        return
    await adispatch_custom_event(name, data, config=config)


def _start_speculative_generate(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> str | None:
    """Begin an eager generate task inside the current coroutine context.

    Returns the thread_id key under which the task is registered, or ``None``
    if overlap is disabled (bound==1) or no thread_id is available. The
    speculative generate's events are buffered until the route is decided.
    """
    if not _spec_overlap_enabled():
        return None
    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return None
    if thread_id in _spec_registry:
        stale = _spec_registry.pop(thread_id, None)
        if stale is not None and stale.adopter is None:
            stale.task.cancel()
            stale.buffer.clear()
            logger.warning(
                "[RAG] [SPEC] stale orphan cancelled (thread_id=%s)", thread_id
            )
    # fall through to register a fresh speculative task

    _spec_generate_events.set([])
    from core.graph._generate import generate

    task = asyncio.ensure_future(generate(state, config, writer=writer))
    # Tag the task so the speculative generate instance does NOT adopt itself
    # (it must run normally and buffer its events instead).
    task._is_speculative = True  # type: ignore[attr-defined]
    _spec_registry[thread_id] = _SpecGenerate(task=task, buffer=[])
    # The speculative task buffers into the ContextVar list we just captured.
    _spec_registry[thread_id].buffer = _spec_generate_events.get([])
    logger.info("[RAG] [SPEC] eager generate 시작 (route 결정 대기)")
    return thread_id
