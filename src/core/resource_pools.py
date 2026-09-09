"""
Resource pool primitives for the unified resource management system.

Holds the LRU pools (BaseResourcePool, ModelPool, RetrieverPool, ClientPool)
plus the OLLAMA/RAM pressure eviction policy. Split out of
``core.resource_manager`` (PHASE 3-P2 decomposition); ``resource_manager``
re-exports these symbols for backward compatibility.

Patch-seam note: ``ModelPool.check_vram_pressure`` reads three names at call
time — ``ENABLE_OLLAMA_PRESSURE_FALLBACK``, ``_host_pressure_exceeded`` and
``_ollama_backend_active``. Those bindings belong to ``core.resource_manager``
(tests ``patch("core.resource_manager.*")`` them). This module binds them to
call-time forwarders that re-read resource_manager's current namespace via
``sys.modules``, keeping the test patches visible without a circular import.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import threading
from collections import OrderedDict
from typing import Any, Generic, ParamSpec, TypeVar

from common.config import HOST_PRESSURE_THRESHOLD
from common.system_pressure import eviction_allowed


def _resource_manager_module():
    """현재 로드된 resource_manager 모듈 객체 (core / src.core 별칭 대응)."""
    return sys.modules.get("core.resource_manager") or sys.modules.get(
        "src.core.resource_manager"
    )


class _LiveOllamaFlag:
    """ENABLE_OLLAMA_PRESSURE_FALLBACK 의 호출 시점 truthy 값 (res_manager 패치 연동)."""

    def __bool__(self) -> bool:
        mod = _resource_manager_module()
        if mod is None:
            return False
        return bool(getattr(mod, "ENABLE_OLLAMA_PRESSURE_FALLBACK", False))


def _host_pressure_exceeded() -> bool:
    """core.resource_manager의 현재 바인딩으로 위임한다 (테스트 patch 연동)."""
    mod = _resource_manager_module()
    if mod is None:
        return False
    return mod._host_pressure_exceeded()


def _ollama_backend_active() -> bool:
    """core.resource_manager의 현재 바인딩으로 위임한다 (테스트 patch 연동)."""
    mod = _resource_manager_module()
    if mod is None:
        return False
    return mod._ollama_backend_active()


# ModelPool.check_vram_pressure는 위 세 이름을 모듈 전역으로 읽는다. 원본에서는
# 이들이 core.resource_manager 네임스페이스에 있었고 테스트가 거기서 patch하므로
# (test_ollama_pressure.py), 호출 시점의 resource_manager 바인딩으로 위임한다.
# 로드 시점 복사는 패치를 무시하고, 정방향 import는 순환 의존이 된다.
ENABLE_OLLAMA_PRESSURE_FALLBACK = _LiveOllamaFlag()


T = TypeVar("T")
P = ParamSpec("P")


logger = logging.getLogger(__name__)


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
        # ENABLE_OLLAMA_PRESSURE_FALLBACK 는 모듈 레벨 import 를 사용하므로,
        # 테스트는 core.resource_manager.ENABLE_OLLAMA_PRESSURE_FALLBACK 를 패치해
        # 동작을 격리할 수 있다 (runtime 재import 는 conftest 별칭으로 인해
        # 서로 다른 모듈 객체를 볼 수 있어 패치가 누수된다).

        try:
            import torch
            import torch.cuda

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
            and _ollama_backend_active()
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
        self._sync_host: str = ""
        self._async_host: str = ""

    def get_sync_client(self, host: str):
        with self._lock:
            if self._sync_host != host or self._sync_client is None:
                import ollama

                self._sync_client = ollama.Client(host=host)
                self._sync_host = host
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
            or self._async_host != host
        ):
            if self._async_client:
                with contextlib.suppress(Exception):
                    await getattr(
                        self._async_client, "close", lambda: asyncio.sleep(0)
                    )()

            self._async_client = ollama.AsyncClient(host=host)
            self._client_loop = current_loop
            self._async_host = host

        return self._async_client


__all__ = [
    "BaseResourcePool",
    "ClientPool",
    "EVICT_COOLDOWN_SECONDS",
    "ModelPool",
    "RetrieverPool",
]
