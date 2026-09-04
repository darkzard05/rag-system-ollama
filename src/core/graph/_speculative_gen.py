"""Speculative generation infrastructure for grade→generate overlap.

Manages buffered events, per-thread speculative tasks, and the adopt/cancel
lifecycle.  Extracted from ``graph_builder`` to keep Zone C self-contained.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from common.config import MAX_CONCURRENT_INFERENCE

logger = logging.getLogger(__name__)

# Buffered events for the in-flight speculative generate of the current query.
_spec_generate_events: contextvars.ContextVar[list[_SpecEvent]] = (
    contextvars.ContextVar("_spec_generate_events")
)


@dataclass
class _SpecEvent:
    """A single buffered adispatch_custom_event payload."""

    name: str
    data: dict[str, Any]
    config: RunnableConfig


@dataclass
class _SpecGenerate:
    """In-flight speculative generate for one thread_id."""

    task: asyncio.Task[dict[str, Any]]
    buffer: list[_SpecEvent]
    adopter: str | None = None  # thread_id that adopted the task, prevents reuse


# Per-thread_id registry of the currently speculative generate task. Populated
# by grade_documents, consumed (adopted or cancelled) by generate.
_spec_registry: dict[str, _SpecGenerate] = {}


def _spec_overlap_enabled() -> bool:
    """True iff two LLM calls can genuinely run concurrently this session."""
    return MAX_CONCURRENT_INFERENCE > 1


def _adopt_speculative_generate(
    config: RunnableConfig,
) -> tuple[asyncio.Task[dict[str, Any]], list[_SpecEvent]] | None:
    """Adopt a warm speculative generate task, if one exists for this thread_id.

    Marks it adopted so it cannot be reused. Returns ``(task, buffered_events)``
    for the adopting node to replay, or ``None`` if there is nothing to adopt
    (normal sequential path).
    """
    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return None
    spec = _spec_registry.pop(thread_id, None)
    if spec is None:
        return None
    if spec.adopter is not None:
        return None
    spec.adopter = thread_id
    return spec.task, spec.buffer


def _cancel_speculative_generate(config: RunnableConfig) -> None:
    """Cancel any speculative generate for this thread_id and drop its buffer.

    Used when grade routes to transform/rewrite — the speculative output must
    never reach the user.
    """
    cfg = config.get("configurable", {})
    thread_id = cfg.get("thread_id")
    if not thread_id:
        return None
    spec = _spec_registry.pop(thread_id, None)
    if spec is None:
        return None
    if spec.adopter is not None:
        return None
    spec.task.cancel()
    spec.buffer.clear()
    logger.info("[RAG] [SPEC] route=transform → speculative generate 취소 (미노출)")


def _replay_spec_events(
    buffer_token: contextvars.Token[list[_SpecEvent]],
) -> list[_SpecEvent]:
    """Return the buffered speculative events and clear them from the context."""
    events = _spec_generate_events.get([])
    _spec_generate_events.reset(buffer_token)
    return events
