"""Backward-compat alias: ``core.graph._glue`` → ``core.graph._graph_core``.

Phase 2A merge. Canonical home is now ``core.graph._graph_core``. This module
aliases ``sys.modules`` so deep patch targets keep hitting the live namespace.
"""

import sys as _sys
from typing import Any

from core.graph import _graph_core as _canonical

_sys.modules[__name__] = _canonical


def __getattr__(name: str) -> Any:
    """Delegate attribute reads to the canonical module."""
    return getattr(_canonical, name)
