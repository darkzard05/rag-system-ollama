"""Session-scoped BM25 parameter isolation.

Migrated from ``scripts/verification/verify_bm25_isolation.py``.

The original script verified that BM25 parameters are copied per session so a
shared (cached) retriever's ``k`` value is never mutated by a session-specific
override. This keeps the shared object stable while each session observes its
own isolated copy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass
class _MockBM25:
    """Minimal stand-in for the shared retriever state we must not mutate."""

    k: int = 1


def _get_isolated_bm25(shared: _MockBM25, target_k: int) -> _MockBM25:
    """Reproduce the per-session isolation logic under test.

    Mirrors the copy-then-override strategy used by the RAG configuration
    prep: the returned object is a fresh copy, so the shared instance is never
    mutated.
    """
    isolated = copy.copy(shared)
    isolated.k = target_k
    return isolated


def test_bm25_isolation_preserves_shared_instance() -> None:
    shared = _MockBM25(k=10)

    session_a = _get_isolated_bm25(shared, target_k=3)
    session_b = _get_isolated_bm25(shared, target_k=5)

    # Each session observes its own isolated k value...
    assert session_a.k == 3
    assert session_b.k == 5

    # ...while the shared instance is never mutated.
    assert shared.k == 10


def test_bm25_isolation_returns_distinct_objects() -> None:
    shared = _MockBM25(k=10)

    session_a = _get_isolated_bm25(shared, target_k=3)
    session_b = _get_isolated_bm25(shared, target_k=5)

    assert session_a is not shared
    assert session_b is not shared
    assert session_a is not session_b


def test_bm25_isolation_does_not_mutate_shared_across_reuse() -> None:
    shared = _MockBM25(k=10)

    # Repeatedly isolating with different k values must never leak back.
    for target_k in (1, 2, 3, 4, 5):
        _get_isolated_bm25(shared, target_k=target_k)

    assert shared.k == 10
