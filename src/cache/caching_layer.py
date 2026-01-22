"""
Multi-Layer Caching System for RAG Performance Optimization
"""

import os
import json
import time
import logging
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Generic, TypeVar, Callable
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from threading import RLock
import hashlib


T = TypeVar('T')


class CacheLevel(Enum):
    """Cache Levels (L1, L2, L3)"""
    L1 = "memory"      # In-memory cache
    L2 = "disk"        # Disk-based cache
    L3 = "distributed" # Distributed cache


class EvictionPolicy(Enum):
    """Cache Eviction Policies"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    FIFO = "fifo"         # First In First Out
    TTL = "ttl"           # Time To Live


@dataclass
class CacheEntry:
    """Cache Entry"""
    key: str
    value: Any
    timestamp: float
    ttl_seconds: Optional[int] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.timestamp) > self.ttl_seconds
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class CacheStats:
    """Cache Statistics"""
    total_hits: int = 0
    total_misses: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    eviction_count: int = 0
    expired_entries: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate hit rate"""
        total = self.total_hits + self.total_misses
        return self.total_hits / total if total > 0 else 0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class MemoryCache(Generic[T]):
    """In-memory L1 cache"""
    
    def __init__(self, max_size: int = 1000, 
                 eviction_policy: EvictionPolicy = EvictionPolicy.LRU):
        """
        Initialize memory cache
        
        Args:
            max_size: Maximum number of entries
            eviction_policy: Eviction policy to use
        """
        self.max_size = max_size
        self.eviction_policy = eviction_policy
        self.entries: Dict[str, CacheEntry] = {}
        self.stats = CacheStats()
        self._lock = RLock()
        self.logger = logging.getLogger(__name__)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return 0
    
    def _evict_if_needed(self):
        """Evict entries if cache is full"""
        if len(self.entries) >= self.max_size:
            if self.eviction_policy == EvictionPolicy.LRU:
                # Remove least recently used
                lru_key = min(
                    self.entries.keys(),
                    key=lambda k: self.entries[k].last_accessed
                )
                del self.entries[lru_key]
                self.stats.eviction_count += 1
            
            elif self.eviction_policy == EvictionPolicy.LFU:
                # Remove least frequently used
                lfu_key = min(
                    self.entries.keys(),
                    key=lambda k: self.entries[k].access_count
                )
                del self.entries[lfu_key]
                self.stats.eviction_count += 1
            
            elif self.eviction_policy == EvictionPolicy.FIFO:
                # Remove oldest entry
                fifo_key = min(
                    self.entries.keys(),
                    key=lambda k: self.entries[k].timestamp
                )
                del self.entries[fifo_key]
                self.stats.eviction_count += 1
    
    def put(self, key: str, value: T, ttl_seconds: Optional[int] = None) -> bool:
        """
        Put value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            Success status
        """
        with self._lock:
            self._evict_if_needed()
            
            size = self._calculate_size(value)
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds,
                access_count=0,
                size_bytes=size,
            )
            
            self.entries[key] = entry
            self.stats.total_size_bytes += size
            
            return True
    
    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            if key not in self.entries:
                self.stats.total_misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check if expired
            if entry.is_expired():
                del self.entries[key]
                self.stats.expired_entries += 1
                self.stats.total_misses += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = time.time()
            self.stats.total_hits += 1
            
            return entry.value
    
    def delete(self, key: str) -> bool:
        """
        Delete entry from cache
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        with self._lock:
            if key in self.entries:
                del self.entries[key]
                return True
            return False
    
    def clear(self) -> int:
        """
        Clear all entries
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self.entries)
            self.entries.clear()
            return count
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            stats = CacheStats(
                total_hits=self.stats.total_hits,
                total_misses=self.stats.total_misses,
                total_entries=len(self.entries),
                total_size_bytes=self.stats.total_size_bytes,
                eviction_count=self.stats.eviction_count,
                expired_entries=self.stats.expired_entries,
            )
            return stats


class DiskCache:
    """Disk-based L2 cache"""
    
    def __init__(self, cache_dir: str = "./cache"):
        """
        Initialize disk cache
        
        Args:
            cache_dir: Directory for cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = CacheStats()
        self._lock = RLock()
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Put value in disk cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            Success status
        """
        with self._lock:
            try:
                cache_file = self._get_cache_file(key)
                
                entry = CacheEntry(
                    key=key,
                    value=value,
                    timestamp=time.time(),
                    ttl_seconds=ttl_seconds,
                )
                
                with open(cache_file, 'wb') as f:
                    pickle.dump(entry.to_dict(), f)
                
                return True
            
            except Exception as e:
                self.logger.error(f"Failed to write disk cache: {str(e)}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from disk cache
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        with self._lock:
            try:
                cache_file = self._get_cache_file(key)
                
                if not cache_file.exists():
                    self.stats.total_misses += 1
                    return None
                
                with open(cache_file, 'rb') as f:
                    entry_data = pickle.load(f)
                
                entry = CacheEntry(**entry_data)
                
                # Check if expired
                if entry.is_expired():
                    cache_file.unlink()
                    self.stats.expired_entries += 1
                    self.stats.total_misses += 1
                    return None
                
                self.stats.total_hits += 1
                return entry.value
            
            except Exception as e:
                self.logger.error(f"Failed to read disk cache: {str(e)}")
                self.stats.total_misses += 1
                return None
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache"""
        with self._lock:
            try:
                cache_file = self._get_cache_file(key)
                if cache_file.exists():
                    cache_file.unlink()
                    return True
                return False
            except Exception as e:
                self.logger.error(f"Failed to delete disk cache: {str(e)}")
                return False
    
    def clear(self) -> int:
        """Clear all cache files"""
        with self._lock:
            try:
                count = 0
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
                    count += 1
                return count
            except Exception as e:
                self.logger.error(f"Failed to clear disk cache: {str(e)}")
                return 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            return CacheStats(
                total_hits=self.stats.total_hits,
                total_misses=self.stats.total_misses,
                total_entries=len(list(self.cache_dir.glob("*.cache"))),
                eviction_count=self.stats.eviction_count,
                expired_entries=self.stats.expired_entries,
            )


class MultiLayerCache:
    """Multi-layer caching system (L1: Memory, L2: Disk, L3: Distributed)"""
    
    def __init__(self, l1_max_size: int = 1000,
                 l2_cache_dir: str = "./cache",
                 l3_enabled: bool = False):
        """
        Initialize multi-layer cache
        
        Args:
            l1_max_size: Maximum L1 cache entries
            l2_cache_dir: Directory for L2 cache
            l3_enabled: Enable L3 distributed cache
        """
        self.l1_cache = MemoryCache(max_size=l1_max_size)
        self.l2_cache = DiskCache(cache_dir=l2_cache_dir)
        self.l3_enabled = l3_enabled
        self.l3_store = {}  # Simulated distributed cache
        
        self.logger = logging.getLogger(__name__)
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            cache_levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        Put value in cache (multi-level)
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live
            cache_levels: Which levels to use (defaults to all)
            
        Returns:
            Success status
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1, CacheLevel.L2]
            if self.l3_enabled:
                cache_levels.append(CacheLevel.L3)
        
        success = True
        
        if CacheLevel.L1 in cache_levels:
            success &= self.l1_cache.put(key, value, ttl_seconds)
        
        if CacheLevel.L2 in cache_levels:
            success &= self.l2_cache.put(key, value, ttl_seconds)
        
        if CacheLevel.L3 in cache_levels and self.l3_enabled:
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl_seconds,
            )
            self.l3_store[key] = entry.to_dict()
        
        return success
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (multi-level with fallback)
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Populate L1 for faster access
            self.l1_cache.put(key, value)
            return value
        
        # Try L3
        if self.l3_enabled and key in self.l3_store:
            entry_data = self.l3_store[key]
            entry = CacheEntry(**entry_data)
            
            if not entry.is_expired():
                value = entry.value
                # Populate L1 and L2
                self.l1_cache.put(key, value, entry.ttl_seconds)
                self.l2_cache.put(key, value, entry.ttl_seconds)
                return value
            else:
                del self.l3_store[key]
        
        return None
    
    def delete(self, key: str) -> bool:
        """Delete from all cache levels"""
        success = True
        success &= self.l1_cache.delete(key)
        success &= self.l2_cache.delete(key)
        if self.l3_enabled and key in self.l3_store:
            del self.l3_store[key]
        return success
    
    def clear(self) -> Dict[str, int]:
        """Clear all cache levels"""
        return {
            "l1": self.l1_cache.clear(),
            "l2": self.l2_cache.clear(),
            "l3": len(self.l3_store) if self.l3_enabled else 0,
        }
    
    def get_stats(self) -> Dict[str, CacheStats]:
        """Get statistics for all cache levels"""
        return {
            "l1": self.l1_cache.get_stats(),
            "l2": self.l2_cache.get_stats(),
        }
    
    def warmup(self, data: Dict[str, Any], ttl_seconds: Optional[int] = None) -> int:
        """
        Warmup cache with initial data
        
        Args:
            data: Dictionary of key-value pairs
            ttl_seconds: Time to live for all entries
            
        Returns:
            Number of entries loaded
        """
        count = 0
        for key, value in data.items():
            if self.put(key, value, ttl_seconds):
                count += 1
        return count


class CacheInvalidator:
    """Cache invalidation manager"""
    
    def __init__(self, cache: MultiLayerCache):
        """
        Initialize cache invalidator
        
        Args:
            cache: Multi-layer cache instance
        """
        self.cache = cache
        self.invalidation_patterns: Dict[str, List[str]] = {}
        self._lock = RLock()
        self.logger = logging.getLogger(__name__)
    
    def register_pattern(self, pattern: str, keys: List[str]):
        """
        Register cache invalidation pattern
        
        Args:
            pattern: Pattern name
            keys: List of cache keys matching pattern
        """
        with self._lock:
            self.invalidation_patterns[pattern] = keys
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern
        
        Args:
            pattern: Pattern name
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            if pattern not in self.invalidation_patterns:
                return 0
            
            keys = self.invalidation_patterns[pattern]
            count = 0
            
            for key in keys:
                if self.cache.delete(key):
                    count += 1
            
            return count
    
    def invalidate_by_prefix(self, prefix: str) -> int:
        """
        Invalidate entries by key prefix
        
        Args:
            prefix: Key prefix
            
        Returns:
            Number of entries invalidated
        """
        # Note: This is a simplified implementation
        # In production, this would need to track all keys
        count = 0
        return count
