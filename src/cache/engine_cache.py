"""Engine cache compat shim — ``EngineCacheManager`` now lives in ``cache.vector_cache``.

Moved in Phase 3C: ``cache.vector_cache`` is the single cache home for
engine + vector. This module re-exports it so legacy importers (tests) keep
resolving. Log namespace ``cache.engine_cache`` is preserved by the class
itself (see ``cache.vector_cache._engine_cache_logger``).
"""

import logging

from cache.vector_cache import EngineCacheManager  # noqa: F401 — re-export for compat

logger = logging.getLogger(__name__)

__all__ = ["EngineCacheManager"]
