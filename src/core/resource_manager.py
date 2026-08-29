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
import time
from collections import OrderedDict
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Generic, ParamSpec, TypeVar, cast

from common.config import (
    ENABLE_OLLAMA_PRESSURE_FALLBACK,
    HOST_PRESSURE_THRESHOLD,
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
from common.system_pressure import (
    eviction_allowed,
    host_pressure_exceeded,
    ollama_backend_active,
)

T = TypeVar("T")
P = ParamSpec("P")

# [R3b-03] 빌드 실패 네거티브 캐시(회로 차단기) 상수.
# build_fn이 연속 `_BUILD_FAILURE_LIMIT`회 실패하면 `_BUILD_CIRCUIT_TTL_SECONDS` 동안
# 재빌드를 차단하고 즉시 ResourceBuildError(reason="circuit_open")를 던진다.
# → FlashRank 같은 무거운 로드가 오프라인/차단 상태에서 매 쿼리 재시도되는 지연 누적을 방지.
_BUILD_FAILURE_LIMIT = 3
_BUILD_CIRCUIT_TTL_SECONDS = 60.0

logger = logging.getLogger(__name__)


def _host_pressure_exceeded() -> bool:
    """동적 참조로 호스트 RAM 압력을 확인합니다.

    모듈 레벨 import 가 아닌 런타임 조회를 쓰므로, 테스트가
    ``common.system_pressure.host_pressure_exceeded`` 를 패치해 동작을 격리할 수
    있습니다 (로컬 import 는 패치 무효화).
    """
    from common.system_pressure import host_pressure_exceeded

    return host_pressure_exceeded()


# [PRESSURE] 퇴출 스로틀 상수 (초). 호스트 RAM 압력 기반 퇴출은 Ollama처럼 별도
# 프로세스 백엔드에서 파이썬 측 핸들 퇴출이 호스트 RAM을 줄이지 못해 조건이 영구히
# 참이 되어 매 호출 "퇴출→즉시 재로드" 쓰래시가 난다. 풀 인스턴스별 쿨다운으로
# 최대 빈도를 제한한다.
EVICT_COOLDOWN_SECONDS: float = 30.0


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
        self._pinned_keys: dict[str, int] = {}
        self._current_bytes = 0
        self._lock = threading.Lock()

    def pin(self, key: str):
        """리소스가 사용 중임을 표시하여 퇴출을 방지합니다 (참조 카운트)."""
        with self._lock:
            self._pinned_keys[key] = self._pinned_keys.get(key, 0) + 1

    def unpin(self, key: str):
        """리소스 사용 완료를 표시하여 퇴출 가능하게 합니다 (참조 카운트)."""
        with self._lock:
            count = self._pinned_keys.get(key, 0) - 1
            if count <= 0:
                self._pinned_keys.pop(key, None)
            else:
                self._pinned_keys[key] = count

    def is_pinned(self, key: str) -> bool:
        """해당 키가 현재 사용 중(pin 카운트 > 0)인지 반환합니다."""
        with self._lock:
            return self._pinned_keys.get(key, 0) > 0

    def key_for_object(self, obj: Any) -> str:
        """풀에 저장된 객체에서 역산한 키를 반환합니다.

        동일성(identity) 매칭을 사용합니다 (모델/리트리버는 풀 내 싱글톤).
        객체가 풀에 없으면 KeyError — 호출부가 잘못된/이미 퇴출된 객체를
        전달한 경우이며, DEFAULT 키로 silently 폴백해선 안 됩니다.
        """
        with self._lock:
            for k, v in self._pool.items():
                if v is obj:
                    return k
            raise KeyError(
                f"Object {obj!r} is not present in pool '{self.name}'; "
                "cannot derive pin key (it may have been evicted)."
            )

    def _evict_one_locked(self) -> bool:
        """가장 오래된 unpinned 리소스를 하나 퇴출합니다 (호출자가 _lock 보유 가정).

        ``put``이 용량 체크→퇴출→삽입을 하나의 임계구역에서 처리하도록
        락 외부에서 대기하지 않고 동기적으로 퇴출합니다. 하위 풀(ModelPool)은
        CUDA 정리 훅을 이 지점에서 수행할 수 있습니다.
        """
        for key in list(self._pool.keys()):
            if key in self._pinned_keys:
                continue
            res = self._pool.pop(key)
            self._current_bytes -= self._get_resource_size(res)
            logger.info(f"[{self.name}Pool] Evicting: {key}")
            del res
            return True
        return False

    async def _evict_one(self) -> bool:
        """가장 오래된 unpinned 리소스를 하나 퇴출합니다."""
        with self._lock:
            return self._evict_one_locked()

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
        """리소스 등록 및 용량 초과 시 퇴출 수행.

        용량 체크→퇴출→삽입 전체를 단일 임계구역에서 수행합니다. 이전 구현은
        용량 초과 판정과 실제 삽입 사이에 ``_lock``을 해제(``await _evict_one``
        대기)하여, 동시 ``put`` 다수가 동시에 한도를 통과해 용량 보장 계약을
        위반(초과 적재 → VRAM OOM)하는 경쟁 상태가 있었습니다.
        """
        resource_size = self._get_resource_size(resource)

        # pinned 리소스는 퇴출 대상이 아니므로, 퇴출로 확보 가능한 용량만 따진다.
        with self._lock:
            if key in self._pool:
                old_res = self._pool[key]
                self._current_bytes -= self._get_resource_size(old_res)
                self._pool.move_to_end(key)

            # 용량 초과 시 가장 오래된 unpinned 리소스를 하나씩 퇴출.
            # 락을 유지한 채 동기 퇴출하므로 동시 put과의 경쟁이 없습니다.
            while (
                len(self._pool) >= self.item_limit
                or (self._current_bytes + resource_size) > self.byte_limit
            ):
                if not self._evict_one_locked():
                    # 더 이상 퇴출할 수 있는(unpinned) 리소스가 없으면 중단
                    break

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
        # 동적 참조: 테스트가 common.config.ENABLE_OLLAMA_PRESSURE_FALLBACK 를
        # monkeypatch 해 동작을 격리할 수 있도록 (모듈 레벨 import 는 패치 무효화).

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
            logger.warning(f"VRAM check failed: {e}")

        if (
            ENABLE_OLLAMA_PRESSURE_FALLBACK
            and ollama_backend_active()
            and _host_pressure_exceeded()
        ):
            if not eviction_allowed(self.name):
                return False
            # 주의: Ollama는 별도 프로세스라 이 핸들 evict는 호스트 RAM을 해방하지
            # 못한다. 효과 없는 퇴출→즉시 재로드(~수십 초) 비용만 발생하므로
            # 기본값은 config에서 비활성화되어 있다. 활성 시에도 쿨다운(30s)으로
            # 무한 쓰래시만 막을 뿐, 실제 메모리 반납은 Ollama keep_alive=0 호출이
            # 필요하다(후속 개선 과제).
            logger.warning(
                "[ModelPool] Host RAM pressure detected (Ollama fallback). "
                "Triggering eviction."
            )
            await self._evict_one()
            return True
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
        # 단일 문서 풀에서는 퇴출할 다른 문서가 없다. 유일 문서를 퇴출하면
        # 즉시 재빌드(전체 재파싱) 루프로 이어지므로 퇴출하지 않는다.
        if len(self._pool) <= 1:
            return False
        try:
            import psutil

            mem = psutil.virtual_memory()
            # HOST_PRESSURE_THRESHOLD 사용(모델 풀과 정책 통일). 동일 mem 객체로
            # 임계값·while 체크를 모두 수행해 불필요한 psutil 호출을 피한다.
            if mem.percent > HOST_PRESSURE_THRESHOLD:
                if not eviction_allowed(self.name):
                    return False
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
