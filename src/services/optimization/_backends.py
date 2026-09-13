"""
Cache backends - re-export shim (Batch 5.1).

Original monolith (921 lines) split into focused modules; this file
re-exports all public names for backward compatibility so existing
``from services.optimization._backends import X`` and
``from services.optimization.caching_optimizer import X`` imports
remain valid. No logic lives here.
"""

from services.optimization._backends_common import (  # noqa: F401
    CACHE_FORMAT,
    CACHE_FORMAT_KEY,
    CacheBackend,
    CacheEntry,
    R,
    T,
    _json_default,
)
from services.optimization._backends_disk import DiskCache  # noqa: F401
from services.optimization._backends_memory import MemoryCache  # noqa: F401
from services.optimization._backends_object import (  # noqa: F401
    ObjectCache,
    SyncCacheBridge,
)
from services.optimization._backends_semantic import SemanticCache  # noqa: F401

__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_KEY",
    "CacheBackend",
    "CacheEntry",
    "DiskCache",
    "MemoryCache",
    "ObjectCache",
    "R",
    "SemanticCache",
    "SyncCacheBridge",
    "T",
    "_json_default",
]
