"""
Unified Resource Management System for the RAG system.
Coordinates specialized pools for Models, Retrievers, and Clients.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import sys
import threading
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Generic, ParamSpec, TypeVar

from common.config import (
    CACHE_DIR,
    MAX_CACHED_MODELS,
    MAX_CONCURRENT_INFERENCE,
    MAX_RESOURCE_POOL_SIZE,
    MAX_RESOURCE_POOL_SIZE_BYTES,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    RERANKER_MODEL_NAME,
)
from common.exceptions import LLMInferenceError

T = TypeVar("T")
P = ParamSpec("P")

logger = logging.getLogger(__name__)


class BaseResourcePool(Generic[T]):
    """
    LRU 기반의 기본 리소스 풀입니다.
    아이템 개수 및 바이트 크기 기반의 퇴출 정책을 관리합니다.
    """

    def __init__(self, name: str, item_limit: int, byte_limit: int):
        self.name = name
        self.item_limit = item_limit
        self.byte_limit = byte_limit
        self._pool: OrderedDict[str, T] = OrderedDict()
        self._pinned_keys: set[str] = set()
        self._current_bytes = 0
        self._lock = threading.Lock()

    def pin(self, key: str):
        """리소스가 사용 중임을 표시하여 퇴출을 방지합니다."""
        with self._lock:
            self._pinned_keys.add(key)

    def unpin(self, key: str):
        """리소스 사용 완료를 표시하여 퇴출 가능하게 합니다."""
        with self._lock:
            self._pinned_keys.discard(key)

    async def _evict_one(self) -> bool:
        """가장 오래된 unpinned 리소스를 하나 퇴출합니다."""
        with self._lock:
            for key in list(self._pool.keys()):
                if key not in self._pinned_keys:
                    res = self._pool.pop(key)
                    self._current_bytes -= self._get_resource_size(res)
                    logger.info(f"[{self.name}Pool] Evicting: {key}")
                    del res
                    return True
            return False

    def _get_resource_size(self, resource: Any) -> int:
        """리소스의 예상 메모리 점유율(bytes)을 계산합니다."""
        if isinstance(resource, tuple):
            return sum(self._get_resource_size(item) for item in resource)

        index = getattr(resource, "index", None) or resource
        if not (hasattr(index, "ntotal") and hasattr(index, "d")):
            return sys.getsizeof(resource)

        # 기본 벡터 데이터 크기 (float32 = 4 bytes)
        base_size = int(index.ntotal * index.d * 4)
        overhead = 0

        # IVF 인덱스 오버헤드 (nlist * d * 4)
        if hasattr(index, "nlist"):
            overhead += int(index.nlist * index.d * 4)

        # HNSW 인덱스 오버헤드 (ntotal * M * 4)
        if hasattr(index, "hnsw"):
            m = getattr(index.hnsw, "M", 16)
            overhead += int(index.ntotal * m * 4)
        elif hasattr(index, "M"):
            overhead += int(index.ntotal * index.M * 4)

        return base_size + overhead

    def get(self, key: str) -> T | None:
        """리소스 조회 및 LRU 순서 업데이트."""
        with self._lock:
            if key in self._pool:
                self._pool.move_to_end(key)
                return self._pool[key]
            return None

    async def put(self, key: str, resource: T):
        """리소스 등록 및 용량 초과 시 퇴출 수행."""
        resource_size = self._get_resource_size(resource)

        with self._lock:
            if key in self._pool:
                old_res = self._pool[key]
                self._current_bytes -= self._get_resource_size(old_res)
                self._pool.move_to_end(key)

        # 용량 제한 도달 시 가장 오래된 unpinned 리소스 퇴출
        while (
            len(self._pool) >= self.item_limit
            or (self._current_bytes + resource_size) > self.byte_limit
        ):
            if not await self._evict_one():
                # 더 이상 퇴출할 수 있는(unpinned) 리소스가 없으면 중단
                break

        with self._lock:
            self._pool[key] = resource
            self._current_bytes += resource_size

    async def remove(self, key: str):
        """리소스 즉시 제거."""
        with self._lock:
            if key in self._pool:
                res = self._pool.pop(key)
                self._current_bytes -= self._get_resource_size(res)
                del res

    def clear(self):
        """풀 전체 초기화."""
        with self._lock:
            self._pool.clear()
            self._current_bytes = 0


class ModelPool(BaseResourcePool[Any]):
    """LLM 및 임베딩 모델 전용 풀. VRAM 압력을 감지하여 퇴출을 유도합니다."""

    async def check_vram_pressure(self) -> bool:
        try:
            import torch

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total = torch.cuda.get_device_properties(device).total_memory / (
                    1024**2
                )
                reserved = torch.cuda.memory_reserved(device) / (1024**2)
                if (reserved / total) * 100 > 90:
                    logger.warning(
                        "[ModelPool] VRAM pressure detected. Triggering eviction."
                    )
                    await self._evict_one()
                    return True
        except Exception as e:
            logger.debug(f"VRAM check failed: {e}")
        return False

    async def _evict_one(self) -> bool:
        success = await super()._evict_one()
        if success:
            self._cleanup_cuda()
        return success

    def _cleanup_cuda(self):
        with contextlib.suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


class RetrieverPool(BaseResourcePool[Any]):
    """벡터 및 BM25 리트리버 전용 풀."""

    async def check_memory_pressure(self) -> bool:
        """시스템 RAM 사용량을 확인하고 임계값 초과 시 리소스를 퇴출합니다."""
        try:
            import psutil

            mem = psutil.virtual_memory()
            if mem.percent > 85:
                logger.warning(
                    f"[{self.name}Pool] Memory pressure detected ({mem.percent}%). Triggering eviction."
                )
                # 메모리 압력이 높을 때 하나씩 퇴출하며 확인
                evicted = False
                while psutil.virtual_memory().percent > 80:
                    if not await self._evict_one():
                        break
                    evicted = True
                return evicted
        except Exception as e:
            logger.debug(f"Memory pressure check failed: {e}")
        return False


class ClientPool:
    """Ollama API 클라이언트 관리 풀. 이벤트 루프별 캐싱을 지원합니다."""

    def __init__(self):
        self._sync_client = None
        self._async_client = None
        self._client_loop = None
        self._lock = threading.Lock()

    def get_sync_client(self, host: str):
        with self._lock:
            if (
                self._sync_client is None
                or getattr(self._sync_client, "base_url", "") != host
            ):
                import ollama

                self._sync_client = ollama.Client(host=host)
            return self._sync_client

    async def get_async_client(self, host: str):
        import ollama

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return ollama.AsyncClient(host=host)

        # 루프가 바뀌었거나 호스트가 바뀌었으면 재생성
        if (
            self._async_client is None
            or self._client_loop != current_loop
            or getattr(self._async_client, "base_url", "") != host
        ):
            if self._async_client:
                with contextlib.suppress(Exception):
                    await getattr(
                        self._async_client, "close", lambda: asyncio.sleep(0)
                    )()

            self._async_client = ollama.AsyncClient(host=host)
            self._client_loop = current_loop

        return self._async_client


class ResourceCoordinator:
    """
    리소스 관리 시스템의 최상위 파사드.
    각 리소스 타입별 전담 풀에 작업을 위임합니다.
    """

    _instance: ResourceCoordinator | None = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_coordinator()
            return cls._instance

    def _init_coordinator(self):
        self.models = ModelPool(
            "Model", MAX_CACHED_MODELS, MAX_RESOURCE_POOL_SIZE_BYTES
        )
        self.retrievers = RetrieverPool(
            "Retriever", MAX_RESOURCE_POOL_SIZE, MAX_RESOURCE_POOL_SIZE_BYTES
        )
        self.clients = ClientPool()
        self._build_locks: dict[str, asyncio.Lock] = {}
        self._build_call_counter = 0
        self._inference_semaphore: asyncio.Semaphore | None = None
        self._semaphore_loop: asyncio.AbstractEventLoop | None = None

    def reset(self) -> None:
        """
        전체 리소스 풀을 초기화하고 세마포어를 재구성합니다.
        유닛 테스트 간 상태 격리를 위해 사용됩니다.
        """
        self.models.clear()
        self.retrievers.clear()
        self.clients = ClientPool()
        self._build_locks.clear()
        self._build_call_counter = 0
        self._inference_semaphore = None
        self._semaphore_loop = None

    def _get_build_lock(self, key: str) -> asyncio.Lock:
        if key not in self._build_locks:
            self._build_locks[key] = asyncio.Lock()
        return self._build_locks[key]

    def _cleanup_build_locks(self) -> None:
        """Remove build locks for keys no longer tracked in any pool."""
        active_keys: set[str] = set()
        for pool in [self.models, self.retrievers]:
            if hasattr(pool, "_pool"):
                active_keys.update(pool._pool.keys())

        stale_keys = [k for k in self._build_locks if k not in active_keys]
        for k in stale_keys:
            del self._build_locks[k]

        if stale_keys:
            logger.debug(f"[RESOURCE] Cleaned up {len(stale_keys)} stale build locks")

    @property
    def inference_semaphore(self) -> asyncio.Semaphore | None:
        return self._inference_semaphore

    @inference_semaphore.setter
    def inference_semaphore(self, sem: asyncio.Semaphore) -> None:
        self._inference_semaphore = sem
        try:
            self._semaphore_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._semaphore_loop = None

    async def acquire_inference_lock(self, timeout: float | None = None) -> None:
        """LLM 추론을 위한 세마포어 락을 획득합니다.

        timeout(초) 내에 획득하지 못하면 LLMInferenceError(reason="timeout")를
        발생시킵니다. 기본값은 config의 OLLAMA_TIMEOUT입니다.
        """
        loop = asyncio.get_running_loop()
        if self._inference_semaphore is None or self._semaphore_loop is not loop:
            self._inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
            self._semaphore_loop = loop
        wait_seconds = timeout if timeout is not None else OLLAMA_TIMEOUT
        if wait_seconds is None or wait_seconds <= 0:
            await self._inference_semaphore.acquire()
            return
        try:
            await asyncio.wait_for(
                self._inference_semaphore.acquire(), timeout=wait_seconds
            )
        except asyncio.TimeoutError as e:
            raise LLMInferenceError(
                reason="timeout",
                details={
                    "operation": "acquire_inference_lock",
                    "timeout_seconds": wait_seconds,
                },
            ) from e

    def release_inference_lock(self) -> None:
        """획득한 LLM 추론 락을 해제합니다."""
        if self._inference_semaphore is not None:
            self._inference_semaphore.release()

    @asynccontextmanager
    async def inference_session(self, timeout: float | None = None):
        """
        LLM 추론을 위한 컨텍스트 매니저입니다.
        진입 시 락을 획득하고, 종료 시 자동으로 락을 해제합니다.
        """
        await self.acquire_inference_lock(timeout)
        try:
            yield
        finally:
            self.release_inference_lock()

    async def get(self, pool_name: str, key: str | None) -> Any | None:
        """Retrieves a resource from the specified pool."""
        if not key:
            return None

        # Pool mapping
        pools = {
            "models": self.models,
            "retrievers": self.retrievers,
        }
        pool = pools.get(pool_name)
        if pool:
            return pool.get(key)
        return None

    async def get_or_build(
        self,
        pool: BaseResourcePool[Any],
        key: str,
        build_fn: Callable[..., Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Retrieves a resource or builds it atomically if not present.
        If build_fn is None and resource is missing, raises ValueError.
        """
        # [Proactive Eviction] Trigger pressure checks before acquisition
        if pool is self.models:
            await self.models.check_vram_pressure()
        elif pool is self.retrievers:
            await self.retrievers.check_memory_pressure()

        lock = self._get_build_lock(key)
        async with lock:
            self._build_call_counter += 1
            if self._build_call_counter >= 50:
                self._build_call_counter = 0
                self._cleanup_build_locks()

            res = pool.get(key)
            if res is not None:
                return res

            if build_fn is None:
                raise ValueError(
                    f"Resource '{key}' not found in {pool.name} and no build_fn provided."
                )

            if asyncio.iscoroutinefunction(build_fn):
                res = await build_fn(*args, **kwargs)
            else:
                # [이벤트 루프 차단 방지] 무거운 sync 모델 로드는 워커 스레드에서 실행
                res = await asyncio.to_thread(build_fn, *args, **kwargs)
            await pool.put(key, res)
            return res

    async def get_llm(self, model_name: str, **kwargs) -> Any:
        from core.model_loader import load_llm

        res = await self.get_or_build(
            self.models, f"llm_{model_name}", load_llm, model_name
        )
        return res.bind(**kwargs) if kwargs else res

    async def get_embedder(self, model_name: str | None = None) -> Any:
        from common.config import DEFAULT_EMBEDDING_MODEL
        from core.model_loader import load_embedding_model

        name = model_name or DEFAULT_EMBEDDING_MODEL
        return await self.get_or_build(self.models, name, load_embedding_model, name)

    async def get_flashranker(self, model_name: str | None = None) -> Any:
        target = model_name or RERANKER_MODEL_NAME

        def _build(name):
            from flashrank import Ranker

            return Ranker(model_name=name, cache_dir=CACHE_DIR)

        return await self.get_or_build(
            self.models, f"flashrank_{target}", _build, target
        )

    async def get_retrievers(
        self,
        file_hash: str,
        build_fn: Callable[..., Any] | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        res = await self.get_or_build(
            self.retrievers, file_hash, build_fn, *args, **kwargs
        )
        self.retrievers.pin(file_hash)
        return res

    def unpin_retrievers(self, file_hash: str):
        """사용이 끝난 리트리버의 핀을 해제하여 퇴출 가능하게 합니다."""
        self.retrievers.unpin(file_hash)

    async def register_retrievers(
        self, file_hash: str, vector_store: Any, bm25_retriever: Any
    ):
        # [Proactive Eviction] Ensure space before registering new retrievers
        await self.retrievers.check_memory_pressure()
        await self.retrievers.put(file_hash, (vector_store, bm25_retriever))

    async def unregister_retrievers(self, file_hash: str):
        await self.retrievers.remove(file_hash)

    def get_client(self, host: str):
        return self.clients.get_sync_client(host)

    async def get_async_client(self, host: str):
        return await self.clients.get_async_client(host)

    async def clear_vram(self):
        # Ollama 모델 언로드 API 호출
        import ollama

        client = ollama.Client(host=OLLAMA_BASE_URL)
        for key in list(self.models._pool.keys()):
            if key.startswith("llm_"):
                model_name = key.replace("llm_", "")
                with contextlib.suppress(Exception):
                    client.generate(model=model_name, keep_alive=0)

        self.models.clear()
        self._cleanup_build_locks()
        gc.collect()

    async def clear_all(self):
        self.models.clear()
        self.retrievers.clear()
        self._cleanup_build_locks()
        gc.collect()

    def get_faiss_gpu_resources(self):
        import faiss

        return getattr(faiss, "StandardGpuResources", lambda: None)()  # type: ignore


# Backward Compatibility Alias
class ResourceManager(ResourceCoordinator):
    """기존 ResourceManager 호출을 위해 ResourceCoordinator를 상속받아 제공합니다."""

    pass


def get_resource_manager() -> ResourceCoordinator:
    return ResourceManager()
