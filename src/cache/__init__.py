"""Cache package — single home for engine + vector caches (Phase 3C).

``cache.vector_cache`` holds ``VectorStoreCache`` + ``EngineCacheManager``.
``cache.coord_cache`` stays the untouched side-cache for PDF coordinates.
``cache.engine_cache`` is a compat shim re-exporting ``EngineCacheManager``.
"""

from cache.coord_cache import COORD_CACHE_DB, CoordCacheManager, coord_cache
from cache.vector_cache import EngineCacheManager, VectorStoreCache

__all__ = [
    "COORD_CACHE_DB",
    "CoordCacheManager",
    "EngineCacheManager",
    "VectorStoreCache",
    "coord_cache",
]
