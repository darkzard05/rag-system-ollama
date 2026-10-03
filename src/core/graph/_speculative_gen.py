"""Speculative generation infrastructure for grade→generate overlap.

Manages buffered events, per-thread speculative tasks, and the adopt/cancel
lifecycle.  Extracted from ``graph_builder`` to keep Zone C self-contained.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import logging
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig

from common.config import MAX_CONCURRENT_INFERENCE

logger = logging.getLogger(__name__)

# T1 growth bound: abandoned thread_ids (session never completes the generate
# path) are never reaped by adopt/cancel alone, so the registry needs its own
# TTL sweeper (run on every touch-point) plus a hard size cap. TTL is checked
# against time.monotonic(); entries without a timestamp (created_at == 0.0)
# are treated as expired — fail-closed toward boundedness.
_SPEC_ORPHAN_TTL_SECONDS = 300.0
_SPEC_REGISTRY_MAX_SIZE = 64

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
    created_at: float = 0.0  # time.monotonic() at registration; 0.0 = unknown


# Per-thread_id registry of the currently speculative generate task. Populated
# by grade_documents, consumed (adopted or cancelled) by generate.
_spec_registry: dict[str, _SpecGenerate] = {}


def _reap_spec_entry(thread_id: str) -> None:
    """Drop one registry entry, cancelling its task and discarding its buffer.

    Cancellation is safe: the owning query falls back to the normal sequential
    generate path when adopt later finds nothing. Never raises.
    """
    spec = _spec_registry.pop(thread_id, None)
    if spec is None:
        return
    if not spec.task.done():
        with contextlib.suppress(RuntimeError, Exception):
            spec.task.cancel()
    else:
        # Retrieve a faulted result so GC never logs "exception was never
        # retrieved" for reaped orphans.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            spec.task.exception()
    spec.buffer.clear()


def _purge_expired_spec_entries(now: float | None = None) -> int:
    """Reap registry entries older than the orphan TTL. Returns reaped count."""
    moment = now if now is not None else time.monotonic()
    expired = [
        thread_id
        for thread_id, spec in _spec_registry.items()
        if moment - spec.created_at >= _SPEC_ORPHAN_TTL_SECONDS
    ]
    for thread_id in expired:
        _reap_spec_entry(thread_id)
    if expired:
        logger.info("[RAG] [SPEC] orphan TTL 정리: %d개", len(expired))
    return len(expired)


def _enforce_spec_registry_bound() -> None:
    """Evict oldest entries beyond the hard cap (abandoned-session backstop)."""
    while len(_spec_registry) > _SPEC_REGISTRY_MAX_SIZE:
        oldest = min(_spec_registry.items(), key=lambda item: item[1].created_at)[0]
        _reap_spec_entry(oldest)


def delete_spec_thread(thread_id: str) -> bool:
    """Remove one thread's speculative entry (session-teardown hook).

    Called from SessionManager.delete_session / RAGSystem.clear_session so an
    abandoned session's speculative task cannot linger. Returns True when an
    entry was present.
    """
    if thread_id not in _spec_registry:
        return False
    _reap_spec_entry(thread_id)
    return True


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
    _purge_expired_spec_entries()
    if spec is None:
        return None
    if spec.adopter is not None:
        return None
    if spec.task.cancelled() or (
        spec.task.done() and spec.task.exception() is not None
    ):
        logger.warning("[RAG] [SPEC] dead spec task — falling back to fresh generate")
        spec.buffer.clear()
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
    spec = _spec_registry.get(thread_id)
    if spec is None:
        _purge_expired_spec_entries()
        return None
    if spec.adopter is not None:
        return None
    _reap_spec_entry(thread_id)
    logger.info("[RAG] [SPEC] route=transform → speculative generate 취소 (미노출)")
    return None
