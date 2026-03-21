"""
Resource Pool 관리 모듈 (Simplified)

무거운 객체(FAISS, BM25)들을 LRU 방식으로 캐싱하여 메모리/VRAM 고갈을 방지합니다.
"""

import asyncio
import contextlib
import gc
import logging
import threading
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
    _creation_lock = threading.Lock()
    _pool: OrderedDict[str, tuple[Any, Any]]
    _local = threading.local()
    max_size: int

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._pool = OrderedDict()
                    cls._instance.max_size = kwargs.get("max_size", 3)
        elif "max_size" in kwargs:
            cls._instance.max_size = kwargs["max_size"]
        return cls._instance

    @property
    def _lock(self) -> asyncio.Lock:
        """
        현재 스레드에 한정된 asyncio.Lock을 반환하여 다중 스레드 환경에서의 안전성을 보장합니다.
        각 스레드는 자신만의 이벤트 루프와 잠금 객체를 갖게 됩니다.
        """
        if not hasattr(self._local, "lock"):
            self._local.lock = asyncio.Lock()
        return self._local.lock

    async def register(self, file_hash: str, vector_store: Any, bm25_retriever: Any):
        """리소스 등록 (LRU 정책 적용)"""
        if not file_hash:
            return

        async with self._lock:
            # 1. 기존 리소스 갱신
            if file_hash in self._pool:
                self._pool.move_to_end(file_hash)
                self._pool[file_hash] = (vector_store, bm25_retriever)
                return

            # 2. 용량 초과 시 오래된 리소스 제거
            while len(self._pool) >= self.max_size:
                old_hash, (old_vs, old_bm) = self._pool.popitem(last=False)
                logger.info(
                    f"[ResourcePool] 용량 초과로 리소스 해제: {old_hash[:8]}..."
                )
                # 명시적 참조 해제
                del old_vs
                del old_bm
                self._cleanup_memory()

            # 3. 신규 리소스 등록
            self._pool[file_hash] = (vector_store, bm25_retriever)
            logger.info(
                f"[ResourcePool] 신규 리소스 등록: {file_hash[:8]}... (현재 풀: {len(self._pool)}/{self.max_size})"
            )

    async def unregister(self, file_hash: str):
        """특정 리소스를 즉시 제거합니다."""
        if not file_hash:
            return
        async with self._lock:
            if file_hash in self._pool:
                old_vs, old_bm = self._pool.pop(file_hash)
                del old_vs
                del old_bm
                self._cleanup_memory()
                logger.info(f"[ResourcePool] 리소스 명시적 해제: {file_hash[:8]}...")

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
        """가비지 컬렉션 및 VRAM 정리를 수행합니다."""
        gc.collect()
        if torch.cuda.is_available():
            try:
                # 파편화된 메모리 정리
                torch.cuda.empty_cache()
                # 가능한 경우 메모리 관리자 리셋 시도
                with contextlib.suppress(Exception):
                    torch.cuda.ipc_collect()
            except Exception:
                pass

    async def clear(self):
        """모든 리소스 해제"""
        async with self._lock:
            self._pool.clear()
            self._cleanup_memory()


def get_resource_pool() -> ResourcePool:
    return ResourcePool()
