"""Engine cache compat shim — ``EngineCacheManager`` now lives in ``core.rag_core``.

Moved in pruning/batch-4: ``core.rag_core`` is the class's sole production
consumer, so the class was inlined there. This module re-exports it so legacy
importers (``core.pipeline_builder`` (removed in batch-4), tests) keep
resolving. Log namespace ``cache.engine_cache`` is preserved by the class
itself (see ``core.rag_core._engine_cache_logger``).
"""

import logging

from core.rag_core import EngineCacheManager  # noqa: F401 — re-export for compat

logger = logging.getLogger(__name__)

__all__ = ["EngineCacheManager"]
