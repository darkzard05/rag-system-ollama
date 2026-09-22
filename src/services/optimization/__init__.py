"""Optimization service package - canonical re-exports."""

from services.optimization._metrics import CacheStatistics
from services.optimization.backends import (
    CACHE_FORMAT,
    CACHE_FORMAT_KEY,
    CacheBackend,
    CacheEntry,
    DiskCache,
    MemoryCache,
    ObjectCache,
    SemanticCache,
    SyncCacheBridge,
)
from services.optimization.caching_optimizer import CacheManager, get_cache_manager

__all__ = [
    "CACHE_FORMAT",
    "CACHE_FORMAT_KEY",
    "CacheBackend",
    "CacheEntry",
    "CacheManager",
    "CacheStatistics",
    "DiskCache",
    "MemoryCache",
    "ObjectCache",
    "SemanticCache",
    "SyncCacheBridge",
    "get_cache_manager",
]
