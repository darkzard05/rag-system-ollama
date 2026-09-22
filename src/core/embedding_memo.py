"""Backward-compat shim: ``core.embedding_memo`` → ``core.model_loader`` (Phase 3A).

Canonical home is now ``core.model_loader`` (memo 섹션). This module
re-exports every public name so existing
``from core.embedding_memo import X`` imports keep working.
No logic lives here.

Live-read note: scalar ``_WAIT_BOUND`` is NOT statically copied — it is
served via module ``__getattr__`` from ``core.model_loader`` so reads stay
live, and ``monkeypatch.setattr(memo_mod, "_WAIT_BOUND", ...)`` creates a
shim-namespace override that ``model_loader._current_wait_bound`` honors.

# backward compat: Phase 3A only
"""

from __future__ import annotations

from typing import Any

from core import model_loader as _ml
from core.model_loader import (  # noqa: F401 — re-exports for backward compat
    _DEFAULT_MAXSIZE,
    _DEFAULT_TTL,
    MemoizingEmbedding,
    _InFlight,
    _memo_instances,
    clear_memo_instances,
)


def __getattr__(name: str) -> Any:
    """Serve non-overridden names live from the canonical module."""
    return getattr(_ml, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_ml)))


__all__ = [
    "_DEFAULT_MAXSIZE",
    "_DEFAULT_TTL",
    "_InFlight",
    "_WAIT_BOUND",  # noqa: F822 — served live via module __getattr__
    "_memo_instances",
    "MemoizingEmbedding",
    "clear_memo_instances",
]
