"""
Unified Resource Management System for the RAG system.
Coordinates specialized pools for Models, Retrievers, and Clients.

Layout: pool primitives (BaseResourcePool, ModelPool, RetrieverPool, ClientPool,
EVICT_COOLDOWN_SECONDS) live in ``core.resource_pools`` (PHASE 3-P2 split) and
are re-exported here for backward compatibility. ``_host_pressure_exceeded`` /
``_ollama_backend_active`` stay here as the patching seam, because tests patch
``core.resource_manager.<name>`` and resource_pools forwards to them at runtime.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import threading
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, cast

from common.config import (
    ENABLE_OLLAMA_PRESSURE_FALLBACK,  # noqa: F401 - patch seam (tests patch this attr; resource_pools forwards)
    MAX_CACHED_MODELS,
    MAX_CONCURRENT_INFERENCE,
    MAX_RESOURCE_POOL_SIZE,
    MAX_RESOURCE_POOL_SIZE_BYTES,
    MODEL_CACHE_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    RERANKER_MODEL_NAME,
)
from common.exceptions import LLMInferenceError, ResourceBuildError
from common.system_pressure import host_pressure_exceeded
from core.resource_pools import (
    EVICT_COOLDOWN_SECONDS,  # noqa: F401 - re-exported for backward compatibility
    BaseResourcePool,
    ClientPool,
    ModelPool,
    RetrieverPool,
)

# [R3b-03] 빌드 실패 네거티브 캐시(회로 차단기) 상수.
# build_fn이 연속 `_BUILD_FAILURE_LIMIT`회 실패하면 `_BUILD_CIRCUIT_TTL_SECONDS` 동안
# 재빌드를 차단하고 즉시 ResourceBuildError(reason="circuit_open")를 던진다.
# → FlashRank 같은 무거운 로드가 오프라인/차단 상태에서 매 쿼리 재시도되는 지연 누적을 방지.
_BUILD_FAILURE_LIMIT = 3
_BUILD_CIRCUIT_TTL_SECONDS = 60.0


logger = logging.getLogger(__name__)


def _host_pressure_exceeded() -> bool:
    """동적 참조로 호스트 RAM 압력을 확인합니다.

    ``sys.modules`` 에서 모듈을 조회해 호출하므로, ``common.system_pressure`` 와
    ``src.common.system_pressure`` 별칭이 다른 객체로 매핑된 환경(conftest)에서도
    테스트 패치가 일관되게 적용됩니다 (로컬 import 는 패치 무효화).
    """
    import common.system_pressure as sp

    return sp.host_pressure_exceeded()


def _ollama_backend_active() -> bool:
    """동적 참조로 Ollama 백엔드 여부를 확인합니다.

    ``sys.modules`` 에서 모듈을 조회해 호출하므로, ``common.system_pressure`` 와
    ``src.common.system_pressure`` 별칭이 다른 객체로 매핑된 환경(conftest)에서도
    테스트 패치가 일관되게 적용됩니다 (로컬 import 는 패치 무효화).
    """
    import common.system_pressure as sp

    return sp.ollama_backend_active()


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
        self._in_flight: set[str] = set()
        self._build_call_counter = 0
        self._build_failures: dict[str, tuple[int, float]] = {}
        self._inference_semaphore: asyncio.Semaphore | None = None
        self._inference_semaphore_bound: int | None = None
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
        self._in_flight.clear()
        self._build_call_counter = 0
        self._build_failures.clear()
        self._inference_semaphore = None
        self._inference_semaphore_bound = None
        self._semaphore_loop = None

    def _get_build_lock(self, key: str) -> asyncio.Lock:
        # Stable per-key lock: a concurrent caller for the same key always
        # receives the SAME lock object, so only one build can run at a time
        # (prevents double-build via lock recycling).
        #
        # asyncio.Lock binds to the running event loop on first acquire (its
        # bound loop is None until then). Under per-test event-loop swapping
        # (pytest-asyncio mode=auto) a lock cached from a previous loop raises
        # ``bound to a different event loop`` on the next test. We therefore
        # reuse an unbound lock, but recreate one whose bound loop differs from
        # the current loop.
        existing = self._build_locks.get(key)
        if existing is None:
            self._build_locks[key] = asyncio.Lock()
        else:
            # asyncio.Lock._loop is private; cast to Any to read the bound loop
            # (None until first acquire). Recreate if it is bound to a different
            # loop than the current one (per-test loop swapping breaks reuse).
            bound_loop = cast("Any", existing)._loop
            if bound_loop is not None and bound_loop is not asyncio.get_event_loop():
                self._build_locks[key] = asyncio.Lock()
        return self._build_locks[key]

    def _cleanup_build_locks(self) -> None:
        """Remove build locks for keys no longer tracked in any pool.

        Keys currently being built (in-flight) are NEVER deleted: deleting an
        in-flight lock would let a concurrent caller acquire a fresh lock and
        double-build the same key.
        """
        active_keys: set[str] = set()
        for pool in [self.models, self.retrievers]:
            if hasattr(pool, "_pool"):
                active_keys.update(pool._pool.keys())

        stale_keys = [
            k
            for k in self._build_locks
            if k not in active_keys and k not in self._in_flight
        ]
        for k in stale_keys:
            del self._build_locks[k]

        if stale_keys:
            logger.debug(f"[RESOURCE] Cleaned up {len(stale_keys)} stale build locks")

    def _check_build_circuit(self, key: str) -> None:
        """[R3b-03] 실패 네거티브 캐시 판정 — 연속 실패 N회 && TTL 미경과면 즉시 차단."""
        now = time.monotonic()
        fail_info = self._build_failures.get(key)
        if fail_info is None:
            return
        count, last_fail = fail_info
        if count >= _BUILD_FAILURE_LIMIT:
            elapsed = now - last_fail
            if elapsed < _BUILD_CIRCUIT_TTL_SECONDS:
                raise ResourceBuildError(
                    key=key,
                    reason="circuit_open",
                    details={
                        "failures": count,
                        "ttl_seconds": _BUILD_CIRCUIT_TTL_SECONDS,
                        "remaining_seconds": round(
                            _BUILD_CIRCUIT_TTL_SECONDS - elapsed, 1
                        ),
                    },
                )
            # TTL 경과 → 재시도 허용 (카운터 리셋)
            self._build_failures.pop(key, None)

    def _record_build_failure(self, key: str) -> None:
        """[R3b-03] 빌드 실패를 네거티브 캐시에 기록하고 회로 차단 전환을 로그로 노출."""
        now = time.monotonic()
        count, _ = self._build_failures.get(key, (0, 0.0))
        count += 1
        self._build_failures[key] = (count, now)
        if count >= _BUILD_FAILURE_LIMIT:
            logger.error(
                f"[RESOURCE] '{key}' 빌드 {count}회 연속 실패 — "
                f"{_BUILD_CIRCUIT_TTL_SECONDS:.0f}초 회로 차단 (재시도 → 즉시 폴백 전환)"
            )
        else:
            logger.warning(
                f"[RESOURCE] '{key}' 빌드 실패 ({count}/{_BUILD_FAILURE_LIMIT}) — 재시도 허용"
            )

    @property
    def inference_semaphore(self) -> asyncio.Semaphore | None:
        return self._inference_semaphore

    @inference_semaphore.setter
    def inference_semaphore(self, sem: asyncio.Semaphore) -> None:
        self._inference_semaphore = sem
        self._inference_semaphore_bound = None
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
        # [WAVE4] VRAM 압력 시 >1 동시 추론을 1로 강등. 기본값(1)에서는
        # host_pressure_exceeded 호출 없이 기존 흐름과 동일(동작/성능 변화 없음).
        effective_bound = (
            1
            if (MAX_CONCURRENT_INFERENCE > 1 and host_pressure_exceeded())
            else MAX_CONCURRENT_INFERENCE
        )
        if self._inference_semaphore is None or self._semaphore_loop is not loop:
            # First creation or loop change -> (re)create the managed default.
            self._inference_semaphore = asyncio.Semaphore(effective_bound)
            self._inference_semaphore_bound = effective_bound
            self._semaphore_loop = loop
        elif self._inference_semaphore_bound is not None:
            # Managed semaphore only: re-create on VRAM-pressure bound change.
            # Track the CREATION bound (not _value) to avoid permit-leak recreation.
            if self._inference_semaphore_bound != effective_bound:
                self._inference_semaphore = asyncio.Semaphore(effective_bound)
                self._inference_semaphore_bound = effective_bound
        # Injected semaphores (bound is None) are respected as-is — never recreated.
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
        # [Proactive Eviction] Models: check VRAM pressure before acquisition.
        # Retrievers: defer the memory-pressure check until AFTER a missing
        # get (see below) — a present doc must never be evicted just to be
        # immediately rebuilt.
        if pool is self.models:
            await self.models.check_vram_pressure()

        lock = self._get_build_lock(key)
        async with lock:
            self._in_flight.add(key)
            try:
                self._build_call_counter += 1
                if self._build_call_counter >= 50:
                    self._build_call_counter = 0
                    self._cleanup_build_locks()

                # [R3b-03] 실패 네거티브 캐시 — 회로 차단 상태면 재빌드를 시도하지 않고 즉시 실패.
                self._check_build_circuit(key)

                res = pool.get(key)
                if res is not None:
                    return res

                # [Proactive Eviction] Only trigger retriever eviction when the
                # resource is actually missing and must be rebuilt. A present doc
                # is never evicted just to be immediately rebuilt (churn loop).
                if pool is self.retrievers:
                    await self.retrievers.check_memory_pressure()

                if build_fn is None:
                    raise ValueError(
                        f"Resource '{key}' not found in {pool.name} and no build_fn provided."
                    )

                try:
                    if asyncio.iscoroutinefunction(build_fn):
                        res = await build_fn(*args, **kwargs)
                    else:
                        # [이벤트 루프 차단 방지] 무거운 sync 모델 로드는 워커 스레드에서 실행
                        res = await asyncio.to_thread(build_fn, *args, **kwargs)
                except Exception:
                    self._record_build_failure(key)
                    raise
                self._build_failures.pop(key, None)
                await pool.put(key, res)
                return res
            finally:
                self._in_flight.discard(key)

    async def get_or_pin(
        self,
        pool: BaseResourcePool[Any],
        key: str,
        build_fn: Callable[..., Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Acquire + pin atomically (fixes use-after-free eviction race).

        Must pin inside the pool's ``_lock`` critical section right after the
        object is fetched/inserted, so no concurrent ``check_vram_pressure`` can
        evict the key in the gap between acquire and a separate caller-side pin.
        Returns ``(resource, key)``; caller must ``unpin(key)`` when done.
        """
        if pool is self.models:
            await self.models.check_vram_pressure()

        lock = self._get_build_lock(key)
        async with lock:
            self._in_flight.add(key)
            try:
                self._build_call_counter += 1
                if self._build_call_counter >= 50:
                    self._build_call_counter = 0
                    self._cleanup_build_locks()

                self._check_build_circuit(key)

                res = pool.get(key)
                if res is not None:
                    pool.pin(key)
                    return res, key

                if pool is self.retrievers:
                    await self.retrievers.check_memory_pressure()

                if build_fn is None:
                    raise ValueError(
                        f"Resource '{key}' not found in {pool.name} and no build_fn provided."
                    )

                try:
                    if asyncio.iscoroutinefunction(build_fn):
                        res = await build_fn(*args, **kwargs)
                    else:
                        res = await asyncio.to_thread(build_fn, *args, **kwargs)
                except Exception:
                    self._record_build_failure(key)
                    raise
                self._build_failures.pop(key, None)
                # put() already holds _lock internally; pin() also acquires it,
                # so do NOT wrap pin in a second `with pool._lock` (non-reentrant
                # threading.Lock would deadlock). Atomicity is preserved: this
                # coroutine runs put() then pin() with no await in between.
                await pool.put(key, res)
                pool.pin(key)
                return res, key
            finally:
                self._in_flight.discard(key)

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

    async def get_llm_for_session(
        self, session_id: str = "default", model_name: str | None = None, **kwargs
    ) -> Any:
        # [R10] LAZY import: SessionManager imports streamlit at module top;
        # importing it here avoids a circular import and keeps Streamlit out of core.
        from common.config import DEFAULT_OLLAMA_MODEL
        from core.session import SessionManager

        current_model = SessionManager.get("last_selected_model", session_id=session_id)
        target_model = model_name or current_model or DEFAULT_OLLAMA_MODEL

        # [R7] 세션별 모델 전환 로그
        if current_model and current_model != target_model:
            logger.info(
                f"[MODEL] [SWITCH] LLM 전환 (Session: {session_id}) | {current_model} -> {target_model}"
            )

        # [R16] POOLING: delegate to the EXISTING name-keyed LRU pool
        # (f"llm_{target_model}"). Do NOT re-key by session_id.
        llm = await self.get_llm(target_model, **kwargs)
        SessionManager.set("last_selected_model", target_model, session_id=session_id)
        return llm

    async def get_embedder_for_session(
        self, session_id: str = "default", model_name: str | None = None
    ) -> Any:
        # [R10] LAZY import: SessionManager imports streamlit at module top;
        # importing it here avoids a circular import and keeps Streamlit out of core.
        from common.config import DEFAULT_EMBEDDING_MODEL
        from core.session import SessionManager

        current_embedder = SessionManager.get(
            "last_selected_embedding_model", session_id=session_id
        )
        target_model = model_name or current_embedder or DEFAULT_EMBEDDING_MODEL

        # [R7] 세션별 임베딩 모델 전환 로그
        if current_embedder and current_embedder != target_model:
            logger.info(
                f"[MODEL] [SWITCH] 임베딩 모델 전환 (Session: {session_id}) | {current_embedder} -> {target_model}"
            )

        # [R16] POOLING: delegate to the EXISTING name-keyed LRU pool (model name).
        embedder = await self.get_embedder(target_model)
        SessionManager.set(
            "last_selected_embedding_model", target_model, session_id=session_id
        )
        return embedder

    async def get_flashranker(self, model_name: str | None = None) -> Any:
        target = model_name or RERANKER_MODEL_NAME

        def _build(name):
            from flashrank import Ranker

            return Ranker(model_name=name, cache_dir=MODEL_CACHE_DIR)

        return await self.get_or_build(
            self.models, f"flashrank_{target}", _build, target
        )

    # --- 사용부 컨텍스트 매니저: get_or_pin 이 원자 획득+pin, CM 은 unpin 만. ---
    # 풀 객체를 이미 가진 호출부는 embedder=/ranker= 전달(key_for_object 역산,
    # 풀 밖 객체면 KeyError). 아니면 model_name 으로 획득.

    @asynccontextmanager
    async def use_embedder(
        self, model_name: str | None = None, embedder: Any | None = None
    ):
        if embedder is not None:
            # Pin only when the embedder is a pool-managed resource. Callers may
            # pass an external (e.g. test/mock) embedder that is never subject to
            # pool eviction; pinning it would raise KeyError via key_for_object.
            try:
                key = self.models.key_for_object(embedder)
            except KeyError:
                key = None
            if key is not None:
                self.models.pin(key)
            emb = embedder
        else:
            from common.config import DEFAULT_EMBEDDING_MODEL
            from core.model_loader import load_embedding_model

            name = model_name or DEFAULT_EMBEDDING_MODEL
            emb, key = await self.get_or_pin(
                self.models, name, load_embedding_model, name
            )
        try:
            yield emb
        finally:
            if key is not None:
                self.models.unpin(key)

    @asynccontextmanager
    async def use_llm(self, model_name: str, **kwargs):
        key = f"llm_{model_name}"
        from core.model_loader import load_llm

        llm, _ = await self.get_or_pin(self.models, key, load_llm, model_name, **kwargs)
        try:
            yield llm
        finally:
            self.models.unpin(key)

    @asynccontextmanager
    async def use_flashranker(
        self, model_name: str | None = None, ranker: Any | None = None
    ):
        target = model_name or RERANKER_MODEL_NAME
        key = f"flashrank_{target}"
        if ranker is not None:
            # Pin only when the ranker is a pool-managed resource; external
            # (test/mock) rankers are never evicted, so key_for_object raises.
            try:
                k2 = self.models.key_for_object(ranker)
            except KeyError:
                k2 = None
            if k2 is not None:
                self.models.pin(k2)
            rk = ranker
        else:

            def _build(name):
                from flashrank import Ranker

                return Ranker(model_name=name, cache_dir=MODEL_CACHE_DIR)

            rk, k2 = await self.get_or_pin(self.models, key, _build, target)
        try:
            yield rk
        finally:
            if k2 is not None:
                self.models.unpin(k2)

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
        # 새로 빌드된 리트리버를 등록 단계에서 자체 퇴출하지 않는다.
        # 호스트 압력은 get_or_build 에서 부재 시에만 처리한다.
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
