"""
Resource Pool 관리 모듈 (Simplified)

무거운 객체(FAISS, BM25)들을 LRU 방식으로 캐싱하여 메모리/VRAM 고갈을 방지합니다.
"""

import asyncio
import contextlib
import gc
import logging
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ResourcePool:
    """
    LRU 기반 리소스 관리자 (Thread-safe & Async-safe).
    단일 인스턴스로 작동하여 시스템 전역 리소스를 관리합니다.
    """

    _instance: "ResourcePool | None" = None
    _lock = asyncio.Lock()
    _pool: OrderedDict[str, tuple[Any, Any]]
    max_size: int

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._pool = OrderedDict()
            cls._instance.max_size = kwargs.get("max_size", 3)
        elif "max_size" in kwargs:
            cls._instance.max_size = kwargs["max_size"]
        return cls._instance

    async def register(self, file_hash: str, vector_store: Any, bm25_retriever: Any):
        """리소스 등록 (LRU 정책 적용)"""
        async with self._lock:
            if file_hash in self._pool:
                self._pool.move_to_end(file_hash)
            else:
                while len(self._pool) >= self.max_size:
                    _, (old_vs, old_bm) = self._pool.popitem(last=False)
                    del old_vs, old_bm
                    self._cleanup_memory()

            self._pool[file_hash] = (vector_store, bm25_retriever)
            logger.info(
                f"[ResourcePool] 등록: {file_hash[:8]}... (사이즈: {len(self._pool)})"
            )

    async def get(self, file_hash: str | None) -> tuple[Any | None, Any | None]:
        """리소스 조회 및 순서 갱신"""
        if not file_hash:
            return None, None
        async with self._lock:
            if file_hash in self._pool:
                self._pool.move_to_end(file_hash)
                return self._pool[file_hash]
            return None, None

    def _cleanup_memory(self):
        """메모리 강제 정리"""
        gc.collect()
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()

    async def clear(self):
        """모든 리소스 해제"""
        async with self._lock:
            self._pool.clear()
            self._cleanup_memory()


def get_resource_pool() -> ResourcePool:
    return ResourcePool()
