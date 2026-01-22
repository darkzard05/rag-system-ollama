"""
리소스 라이프사이클 관리 모듈 - 자동 정리, 리소스 풀링.
"""

import logging
import asyncio
import weakref
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Dict, List, Optional, Set, Type, TypeVar
from threading import RLock, Event
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class ResourceStats:
    """리소스 통계."""
    total_created: int = 0
    currently_allocated: int = 0
    total_released: int = 0
    peak_usage: int = 0


class Resource:
    """관리되는 리소스의 기본 클래스."""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        self.is_allocated = True
    
    async def initialize(self):
        """초기화 (서브클래스에서 오버라이드)."""
        pass
    
    async def cleanup(self):
        """정리 (서브클래스에서 오버라이드)."""
        pass


class ResourcePool:
    """리소스 풀 - 재사용 가능한 리소스 관리."""
    
    def __init__(
        self,
        resource_type: Type[T],
        max_size: int = 10,
        timeout_seconds: float = 30.0,
    ):
        self.resource_type = resource_type
        self.max_size = max_size
        self.timeout = timeout_seconds
        
        self._lock = RLock()
        self._available: List[T] = []
        self._in_use: Set[T] = set()
        self._stats = ResourceStats()
        self._creation_count = 0
    
    async def acquire(self) -> T:
        """리소스 획득."""
        with self._lock:
            # 사용 가능한 리소스 확인
            if self._available:
                resource = self._available.pop()
                self._in_use.add(resource)
                return resource
            
            # 새로운 리소스 생성
            if self._creation_count < self.max_size:
                resource = self.resource_type()
                self._creation_count += 1
                self._stats.total_created += 1
                self._in_use.add(resource)
                
                if isinstance(resource, Resource):
                    await resource.initialize()
                
                self._stats.currently_allocated = len(self._in_use)
                self._stats.peak_usage = max(
                    self._stats.peak_usage,
                    self._stats.currently_allocated
                )
                
                return resource
        
        # 타임아웃 대기
        start_time = asyncio.get_event_loop().time()
        while True:
            await asyncio.sleep(0.1)
            
            with self._lock:
                if self._available:
                    resource = self._available.pop()
                    self._in_use.add(resource)
                    return resource
            
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"리소스 획득 타임아웃 ({self.timeout}s)")
    
    async def release(self, resource: T):
        """리소스 해제."""
        with self._lock:
            if resource not in self._in_use:
                logger.warning(f"관리되지 않는 리소스 해제 시도: {resource}")
                return
            
            self._in_use.remove(resource)
            self._available.append(resource)
            self._stats.total_released += 1
            self._stats.currently_allocated = len(self._in_use)
    
    async def clear(self):
        """풀 정리."""
        with self._lock:
            # 사용 중인 리소스 정리
            for resource in self._in_use:
                if isinstance(resource, Resource):
                    await resource.cleanup()
            
            # 사용 가능한 리소스 정리
            for resource in self._available:
                if isinstance(resource, Resource):
                    await resource.cleanup()
            
            self._available.clear()
            self._in_use.clear()
            self._creation_count = 0
            logger.info(f"리소스 풀 정리 완료: {self.resource_type.__name__}")
    
    def get_stats(self) -> ResourceStats:
        """통계 조회."""
        with self._lock:
            return ResourceStats(
                total_created=self._stats.total_created,
                currently_allocated=self._stats.currently_allocated,
                total_released=self._stats.total_released,
                peak_usage=self._stats.peak_usage,
            )


class ResourceManager:
    """리소스 관리자 - 리소스 풀 통합 관리."""
    
    def __init__(self):
        self._lock = RLock()
        self._pools: Dict[str, ResourcePool] = {}
        self._tracked_resources: List[weakref.ref] = []
    
    def create_pool(
        self,
        name: str,
        resource_type: Type[T],
        max_size: int = 10,
    ) -> ResourcePool:
        """리소스 풀 생성."""
        with self._lock:
            if name in self._pools:
                logger.warning(f"이미 존재하는 풀: {name}")
                return self._pools[name]
            
            pool = ResourcePool(resource_type, max_size)
            self._pools[name] = pool
            logger.info(f"리소스 풀 생성: {name} (최대 {max_size})")
            
            return pool
    
    def get_pool(self, name: str) -> Optional[ResourcePool]:
        """리소스 풀 조회."""
        with self._lock:
            return self._pools.get(name)
    
    def register_resource(self, resource: Any) -> str:
        """리소스 등록."""
        resource_id = f"resource_{len(self._tracked_resources)}"
        
        with self._lock:
            self._tracked_resources.append(weakref.ref(resource))
        
        return resource_id
    
    async def cleanup_all(self):
        """모든 풀 정리."""
        with self._lock:
            pools_to_cleanup = list(self._pools.values())
        
        for pool in pools_to_cleanup:
            await pool.clear()
        
        with self._lock:
            self._pools.clear()
            self._tracked_resources.clear()
        
        logger.info("모든 리소스 정리 완료")
    
    def get_stats(self) -> Dict[str, ResourceStats]:
        """모든 풀의 통계 조회."""
        with self._lock:
            return {
                name: pool.get_stats()
                for name, pool in self._pools.items()
            }


class ManagedResource:
    """자동 정리되는 관리 리소스."""
    
    def __init__(self):
        self.is_open = True
        self._cleanup_callbacks: List[Any] = []
    
    def register_cleanup(self, callback: Any):
        """정리 콜백 등록."""
        self._cleanup_callbacks.append(callback)
    
    async def close(self):
        """리소스 종료."""
        if not self.is_open:
            return
        
        self.is_open = False
        
        # 역순으로 정리 콜백 실행
        for callback in reversed(self._cleanup_callbacks):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"정리 콜백 실행 실패: {e}")
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class ContextualResourceManager:
    """컨텍스트 기반 리소스 관리."""
    
    def __init__(self):
        self._lock = RLock()
        self._resource_stacks: Dict[str, List[Any]] = {}
        self.manager = ResourceManager()
    
    @asynccontextmanager
    async def managed_resource(self, resource_type: str):
        """관리되는 리소스 획득."""
        with self._lock:
            if resource_type not in self._resource_stacks:
                self._resource_stacks[resource_type] = []
        
        pool = self.manager.get_pool(resource_type)
        if not pool:
            logger.error(f"리소스 풀 없음: {resource_type}")
            raise ValueError(f"Unknown resource type: {resource_type}")
        
        resource = await pool.acquire()
        
        with self._lock:
            self._resource_stacks[resource_type].append(resource)
        
        try:
            yield resource
        finally:
            with self._lock:
                if self._resource_stacks[resource_type]:
                    self._resource_stacks[resource_type].pop()
            
            await pool.release(resource)
    
    def get_context_resources(self, resource_type: str) -> List[Any]:
        """현재 컨텍스트의 리소스 조회."""
        with self._lock:
            return self._resource_stacks.get(resource_type, []).copy()
    
    async def cleanup(self):
        """모든 리소스 정리."""
        await self.manager.cleanup_all()
        
        with self._lock:
            self._resource_stacks.clear()


class DeferredCleanup:
    """지연된 정리 - 컨텍스트 종료 시점에 실행."""
    
    def __init__(self):
        self._cleanups: List[Any] = []
        self._lock = RLock()
    
    def defer(self, cleanup_func: Any):
        """정리 작업 등록."""
        with self._lock:
            self._cleanups.append(cleanup_func)
    
    async def cleanup(self):
        """등록된 모든 정리 작업 실행."""
        with self._lock:
            cleanups = list(reversed(self._cleanups))
            self._cleanups.clear()
        
        for cleanup_func in cleanups:
            try:
                if asyncio.iscoroutinefunction(cleanup_func):
                    await cleanup_func()
                else:
                    cleanup_func()
            except Exception as e:
                logger.error(f"지연 정리 실패: {e}")


# 전역 인스턴스
_resource_manager_instance: Optional[ResourceManager] = None
_contextual_manager_instance: Optional[ContextualResourceManager] = None


def get_resource_manager() -> ResourceManager:
    """전역 리소스 관리자 조회."""
    global _resource_manager_instance
    if _resource_manager_instance is None:
        _resource_manager_instance = ResourceManager()
    return _resource_manager_instance


def get_contextual_resource_manager() -> ContextualResourceManager:
    """전역 컨텍스트 기반 리소스 관리자 조회."""
    global _contextual_manager_instance
    if _contextual_manager_instance is None:
        _contextual_manager_instance = ContextualResourceManager()
    return _contextual_manager_instance


def reset_resource_manager():
    """리소스 관리자 리셋 (테스트용)."""
    global _resource_manager_instance
    _resource_manager_instance = None


def reset_contextual_resource_manager():
    """컨텍스트 기반 리소스 관리자 리셋 (테스트용)."""
    global _contextual_manager_instance
    _contextual_manager_instance = None
