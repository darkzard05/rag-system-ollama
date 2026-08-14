"""
캐싱 최적화 - Task 13
응답 캐싱, 세맨틱 캐싱, TTL 관리, 캐시 일관성
"""

import asyncio
import contextlib
import hashlib
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Generic, TypeVar

import numpy as np

from common.utils import fast_hash
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)


# 타입 변수
T = TypeVar("T")
R = TypeVar("R")


@dataclass
class CacheEntry:
    """캐시 항목"""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    ttl_seconds: float
    hit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """TTL 만료 여부 확인"""
        if self.ttl_seconds <= 0:
            return False
        return time.time() - self.created_at > self.ttl_seconds

    def get_age(self) -> float:
        """항목 나이 (초)"""
        return time.time() - self.created_at

    def touch(self) -> None:
        """접근 시간 업데이트"""
        self.accessed_at = time.time()
        self.hit_count += 1


@dataclass
class CacheStatistics:
    """캐시 통계"""

    total_hits: int = 0
    total_misses: int = 0
    total_evictions: int = 0
    total_expirations: int = 0
    cache_size: int = 0
    total_memory_bytes: int = 0
    hit_rate: float = 0.0
    avg_age_seconds: float = 0.0

    @property
    def total_requests(self) -> int:
        return self.total_hits + self.total_misses

    def update_hit_rate(self) -> None:
        """히트율 계산"""
        if self.total_requests > 0:
            self.hit_rate = self.total_hits / self.total_requests


class CacheBackend(ABC, Generic[T]):
    """캐시 백엔드 추상 클래스"""

    @abstractmethod
    async def get(self, key: str) -> T | None:
        """값 조회"""
        pass

    @abstractmethod
    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        """값 삭제"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """캐시 전체 삭제"""
        pass

    @abstractmethod
    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        pass


class MemoryCache(CacheBackend[T]):
    """
    메모리 기반 캐시

    특징:
    - LRU 제거 정책
    - TTL 만료 처리
    - 메모리 사용량 추적
    - 통계 수집
    """

    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.default_ttl = ttl_seconds
        self.cache: dict[str, CacheEntry] = {}
        self.lock = RLock()
        self.stats = CacheStatistics()

    async def get(self, key: str) -> T | None:
        """값 조회"""
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                self.stats.total_misses += 1
                return None

            # 만료 확인
            if entry.is_expired():
                logger.debug(f"[Cache] 만료된 항목 제거: {key}")
                del self.cache[key]
                self.stats.total_misses += 1
                self.stats.total_expirations += 1
                return None

            # 접근 업데이트
            entry.touch()
            self.stats.total_hits += 1
            self.stats.update_hit_rate()

            logger.debug(f"[Cache] 캐시 히트: {key} (히트 수: {entry.hit_count})")
            return entry.value

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정 (사이즈 추적 최적화)"""
        with self.lock:
            ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

            # [최적화] 기존 항목이 있으면 사이즈 차감
            if key in self.cache:
                self.stats.total_memory_bytes -= self.cache[key].metadata.get(
                    "size_bytes", 0
                )

            # 메모리 확인 및 정리
            self._cleanup_if_needed(value)

            # 대략적인 사이즈 계산
            import sys

            size_bytes = sys.getsizeof(value)

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl,
                metadata={"size_bytes": size_bytes},
            )

            self.cache[key] = entry
            self.stats.total_memory_bytes += size_bytes
            self.stats.cache_size = len(self.cache)

            logger.debug(f"[Cache] 값 저장: {key} (TTL: {ttl}초, Size: {size_bytes}B)")

    async def delete(self, key: str) -> None:
        """값 삭제 (사이즈 차감 포함)"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.stats.total_memory_bytes -= entry.metadata.get("size_bytes", 0)
                del self.cache[key]
                self.stats.cache_size = len(self.cache)
                logger.debug(f"[Cache] 값 삭제: {key}")

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.stats.cache_size = 0
            logger.info("[Cache] 캐시 전체 삭제")

    def get_stats(self) -> CacheStatistics:
        """통계 조회 (계산 오버헤드 최적화)"""
        with self.lock:
            stats = self.stats
            stats.cache_size = len(self.cache)

            # [최적화] 모든 항목을 순회하며 JSON 직렬화를 반복하는 대신,
            # 저장 시 계산된 total_size_bytes를 즉시 활용
            ages = [entry.get_age() for entry in self.cache.values()]
            stats.avg_age_seconds = sum(ages) / len(ages) if ages else 0

            return stats

    def _cleanup_if_needed(self, new_value: T) -> None:
        """메모리 및 크기 조건에 따라 정리 (계산 최적화)"""
        # 1. 크기 초과 확인 (O(1))
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        # 2. 메모리 초과 확인
        # [최적화] 매번 전체 캐시를 순회하지 않고, 새로 추가될 값의 크기만 계산
        try:
            # sys.getsizeof()는 실제 메모리 점유율을 정확히 반영하지 못하므로
            # 직렬화된 크기를 기준으로 하되, 이미 계산된 total_size_bytes를 활용
            import sys

            estimated_bytes = sys.getsizeof(new_value)
            if isinstance(new_value, (str, bytes)):
                estimated_bytes = len(new_value)
            elif isinstance(new_value, (list, dict, set, tuple)):
                estimated_bytes = len(new_value) * 8  # 대략적인 포인터 크기
            else:
                estimated_bytes = 1024
        except Exception:
            estimated_bytes = 1024  # 폴백: 1KB

        if (
            self.stats.total_memory_bytes + estimated_bytes
            > self.max_memory_mb * 1024 * 1024
        ):
            self._evict_lru()

    def _evict_lru(self) -> None:
        """LRU 항목 제거"""
        if not self.cache:
            return

        # 접근 시간이 가장 오래된 항목 찾기
        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)

        del self.cache[lru_key]
        self.stats.total_evictions += 1
        logger.debug(f"[Cache] LRU 제거: {lru_key}")


class SemanticCache(CacheBackend[T]):
    """
    세맨틱 캐시 - 임베딩 기반 유사성 캐싱

    특징:
    - 벡터 기반 유사성 검색
    - 의미적으로 유사한 쿼리 매칭
    - 거리 기반 검색
    - 메모리 효율적
    """

    def __init__(
        self,
        embedding_model=None,
        similarity_threshold: float = 0.95,
        max_entries: int = 500,
        ttl_seconds: float = 3600.0,
    ):
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.default_ttl = ttl_seconds
        self.embeddings: dict[str, np.ndarray] = {}
        self.cache: dict[str, CacheEntry] = {}
        self.lock = RLock()
        self.stats = CacheStatistics()
        self._cached_matrix: np.ndarray | None = None
        self._cached_keys: list[str] = []

    @property
    def cache_size(self) -> int:
        """캐시 크기"""
        return len(self.cache)

    def _update_matrix(self, key: str | None = None, action: str = "add") -> None:
        """캐시된 행렬 업데이트 (증분 방식)"""
        if action == "add" and key is not None:
            self._cached_keys.append(key)
            new_embedding = self.embeddings[key]
            if self._cached_matrix is None:
                self._cached_matrix = np.array([new_embedding])
            else:
                self._cached_matrix = np.vstack([self._cached_matrix, new_embedding])
        elif action == "remove" and key is not None:
            try:
                idx = self._cached_keys.index(key)
                self._cached_keys.pop(idx)
                if self._cached_matrix is not None:
                    self._cached_matrix = np.delete(self._cached_matrix, idx, axis=0)
                    if self._cached_matrix.size == 0:
                        self._cached_matrix = None
            except ValueError:
                pass
        else:
            self._cached_keys = list(self.embeddings.keys())
            if not self._cached_keys:
                self._cached_matrix = None
            else:
                self._cached_matrix = np.array(
                    [self.embeddings[k] for k in self._cached_keys]
                )

    async def _embed(self, text: str) -> np.ndarray:
        """텍스트 임베딩"""
        if self.embedding_model is not None and hasattr(
            self.embedding_model, "embed_query"
        ):
            embedding = await self.embedding_model.embed_query(text)
            return np.array(embedding)

        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        np.random.seed(hash_int % (2**32))
        return np.random.randn(384)

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            self.stats.update_hit_rate()
            return self.stats

    async def get(
        self, key: str, similarity_threshold: float | None = None
    ) -> T | None:
        """
        의미적으로 유사한 항목 조회 (NumPy 벡터화 최적화)
        """
        if not self.embedding_model or not self.embeddings:
            return None

        with self.lock:
            threshold = similarity_threshold or self.similarity_threshold

            try:
                # 쿼리 임베딩
                query_embedding = await self._embed(key)
                query_embedding = query_embedding / (
                    np.linalg.norm(query_embedding) + 1e-10
                )

                # [최적화] 캐시된 행렬 사용
                if self._cached_matrix is None:
                    self._update_matrix()

                if self._cached_matrix is None:
                    return None

                # 코사인 유사도 계산 (행렬-벡터 내적)
                similarities = np.dot(self._cached_matrix, query_embedding)

                # 가장 유사한 항목 찾기
                max_idx = np.argmax(similarities)
                best_similarity = similarities[max_idx]
                best_match = self._cached_keys[max_idx]

                # 임계값 이상인 경우 반환
                if best_similarity >= threshold:
                    entry = self.cache[best_match]
                    entry.touch()
                    self.stats.total_hits += 1
                    logger.debug(f"[SemanticCache] 히트: 유사도 {best_similarity:.3f}")
                    return entry.value

                self.stats.total_misses += 1
                return None

            except Exception as e:
                logger.error(f"[SemanticCache] 조회 오류: {e}")
                self.stats.total_misses += 1
                return None

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 저장 및 벡터 정규화"""
        with self.lock:
            try:
                ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

                if len(self.cache) >= self.max_entries:
                    self._evict_oldest()

                cache_key = fast_hash(key)

                query_embedding = None
                if self.embedding_model:
                    query_embedding = await self._embed(key)
                    # [최적화] 저장 시 미리 정규화하여 get 단계의 연산 감소
                    norm = np.linalg.norm(query_embedding)
                    if norm > 0:
                        query_embedding = query_embedding / norm

                entry = CacheEntry(
                    key=cache_key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl,
                    metadata={"query": key[:100]},
                )

                self.cache[cache_key] = entry
                if query_embedding is not None:
                    self.embeddings[cache_key] = query_embedding
                    self._update_matrix()  # 행렬 업데이트
                self.stats.cache_size = len(self.cache)

            except Exception as e:
                logger.error(f"[SemanticCache] 저장 오류: {e}")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            if key in self.embeddings:
                del self.embeddings[key]
                self._update_matrix()  # 행렬 업데이트
            self.stats.cache_size = len(self.cache)

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.embeddings.clear()
            self._cached_matrix = None
            self._cached_keys = []
            self.stats.cache_size = 0
            logger.info("[SemanticCache] 캐시 전체 삭제")

    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거"""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)

        del self.cache[oldest_key]
        if oldest_key in self.embeddings:
            del self.embeddings[oldest_key]
            self._update_matrix()  # 행렬 업데이트

        self.stats.total_evictions += 1
        logger.debug(f"[SemanticCache] 가장 오래된 항목 제거: {oldest_key}")


class DiskCache(CacheBackend[T]):
    """
    보안 강화된 디스크 기반 캐시 (L3)

    특징:
    - 영구 저장 지원
    - CacheSecurityManager를 통한 무결성 검증 (HMAC)
    - 역직렬화 전 보안 체크
    - TTL 기반 만료 처리
    """

    def __init__(self, cache_dir: str = "./.model_cache/response_cache"):
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.stats = CacheStatistics()

        # 보안 관리자 초기화 (공유 인스턴스 사용으로 중복 로그 방지)
        from security.cache_security import get_security_manager

        self.security_manager = get_security_manager()

    def _get_cache_path(self, key: str) -> Path:
        """키에 대한 파일 경로 생성"""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"

    async def get(self, key: str) -> T | None:
        """값 조회 (보안 검증 포함)"""
        return await asyncio.to_thread(self._get_sync, key)

    def _get_sync(self, key: str) -> T | None:
        """동기 파일 조회 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        with self.lock:
            cache_file = self._get_cache_path(key)
            if not cache_file.exists():
                self.stats.total_misses += 1
                return None

            try:
                # 1. 보안 검증 (Full Verification)
                success, error = self.security_manager.full_verification(
                    str(cache_file)
                )
                if not success:
                    # [개선] 오류 성격에 따라 로그 레벨 조정
                    if "HMAC" in str(error) or "신뢰" in str(error):
                        logger.warning(
                            f"[DiskCache] 잠재적 보안 이슈로 캐시 무효화: {error}"
                        )
                    else:
                        logger.info(
                            f"[DiskCache] 유효하지 않은 캐시 항목 정리 (사유: {error})"
                        )

                    self._delete_file(cache_file)
                    self.stats.total_misses += 1
                    return None

                # 2. 안전하게 로드
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)  # nosec B301

                entry = CacheEntry(**data) if isinstance(data, dict) else data

                # 3. 만료 확인
                if entry.is_expired():
                    logger.debug(f"[DiskCache] 만료된 항목 제거: {key}")
                    self._delete_file(cache_file)
                    self.stats.total_misses += 1
                    self.stats.total_expirations += 1
                    return None

                entry.touch()
                self.stats.total_hits += 1
                return entry.value

            except Exception as e:
                logger.error(f"[DiskCache] 로드 오류: {e}")
                # 손상된 캐시 파일 정리 (포맷 불일치/부분 쓰기 등)
                self._delete_file(cache_file)
                self.stats.total_misses += 1
                return None

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 저장 (보안 메타데이터 생성 및 권한 강제 포함)"""
        await asyncio.to_thread(self._set_sync, key, value, ttl_seconds)

    def _set_sync(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """동기 파일 저장 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        with self.lock:
            cache_file = self._get_cache_path(key)
            try:
                # 디렉토리 권한 보장
                if not self.cache_dir.exists():
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    self.security_manager.enforce_directory_permissions(
                        str(self.cache_dir)
                    )

                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    accessed_at=time.time(),
                    ttl_seconds=ttl_seconds if ttl_seconds > 0 else 86400.0,
                )

                # [최적화] 전체 디렉터리 스캔(glob) 대신 증분 카운터 유지 (쓰기당 O(N) 제거)
                if not cache_file.exists():
                    self.stats.cache_size += 1

                # 1. 파일 저장 — 클래스 식별자 문제를 피해, 직렬화는 기본 dict로 수행합니다.
                with open(cache_file, "wb") as f:
                    pickle.dump(entry.__dict__, f)

                # 2. 파일 권한 강제 적용
                self.security_manager.enforce_file_permissions(str(cache_file))

                # 3. 보안 메타데이터 생성 및 저장
                metadata = self.security_manager.create_metadata_for_file(
                    str(cache_file), description=f"Cache entry: {key[:30]}"
                )
                self.security_manager.save_cache_metadata(
                    str(cache_file) + ".meta", metadata
                )

            except Exception as e:
                logger.error(f"[DiskCache] 저장 오류: {e}")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        await asyncio.to_thread(self._delete_sync, key)

    def _delete_sync(self, key: str) -> None:
        """동기 파일 삭제 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        with self.lock:
            self._delete_file(self._get_cache_path(key))

    async def clear(self) -> None:
        """전체 삭제"""
        await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> None:
        """동기 전체 삭제 — 이벤트 루프 블로킹 방지를 위해 스레드에서 실행"""
        import contextlib

        with self.lock:
            for f in self.cache_dir.glob("*.cache*"):
                with contextlib.suppress(Exception):
                    f.unlink()
            self.stats.cache_size = 0

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.update_hit_rate()
            return self.stats

    def _delete_file(self, path: Path) -> None:
        """파일 및 메타데이터 삭제 (증분 카운터 동기화 포함)"""
        try:
            removed = False
            if path.exists():
                path.unlink()
                removed = True
            meta = Path(str(path) + ".meta")
            if meta.exists():
                meta.unlink()
            if removed and self.stats.cache_size > 0:
                self.stats.cache_size -= 1
        except Exception:
            pass


class CacheManager:
    """
    캐시 관리자 - 다중 캐시 백엔드 통합

    특징:
    - 다중 캐시 레이어 (L1: 메모리, L2: 세맨틱)
    - 자동 캐시 선택
    - 캐시 동기화
    - 통합 통계
    """

    def __init__(
        self,
        enable_memory_cache: bool = True,
        enable_semantic_cache: bool = True,
        enable_disk_cache: bool = True,
        embedding_model=None,
        memory_cache_size: int = 1000,
        semantic_cache_size: int = 500,
        disk_cache_dir: str = "./.model_cache/response_cache",
    ):
        self.enable_memory_cache = enable_memory_cache
        self.enable_semantic_cache = enable_semantic_cache
        self.enable_disk_cache = enable_disk_cache

        self.memory_cache: MemoryCache | None = None
        self.semantic_cache: SemanticCache | None = None
        self.disk_cache: DiskCache | None = None

        if enable_memory_cache:
            self.memory_cache = MemoryCache(max_size=memory_cache_size)

        if enable_semantic_cache:
            self.semantic_cache = SemanticCache(
                embedding_model=embedding_model, max_entries=semantic_cache_size
            )

        if enable_disk_cache:
            self.disk_cache = DiskCache(cache_dir=disk_cache_dir)

        self.lock = RLock()

    async def get(self, key: str, use_semantic: bool = False) -> Any | None:
        """값 조회 (L1 -> L2 -> L3)"""
        with get_performance_monitor().track_operation(
            OperationType.QUERY_PROCESSING,
            {"stage": "cache_lookup", "semantic": use_semantic},
        ) as op:
            # 1. L1 메모리 캐시 확인 (가장 빠름)
            if self.memory_cache and not use_semantic:
                result = await self.memory_cache.get(key)
                if result is not None:
                    op.metadata = {"cache_level": "L1"}
                    return result

            # 2. L2 세맨틱 캐시 확인 (의미적 유사성)
            if self.semantic_cache and use_semantic:
                result = await self.semantic_cache.get(key)
                if result is not None:
                    op.metadata = {"cache_level": "L2"}

                    # L1으로 승격 (Promotion)
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            # 3. L3 디스크 캐시 확인 (영구 저장소)
            if self.disk_cache and not use_semantic:
                result = await self.disk_cache.get(key)
                if result is not None:
                    op.metadata = {"cache_level": "L3"}

                    # L1으로 승격
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: float = 0,
        use_semantic: bool = False,
        persist_to_disk: bool = True,
    ) -> None:
        """값 저장 (L1, L2, L3)"""
        with self.lock:
            # L1에 저장
            if self.memory_cache:
                await self.memory_cache.set(key, value, ttl_seconds)

            # L2 (세맨틱) 저장
            if self.semantic_cache and use_semantic:
                await self.semantic_cache.set(key, value, ttl_seconds)

            # L3 (디스크) 저장
            # [최적화] persist_to_disk=False 시 디스크 저장 생략
            # (문장 임베딩은 FAISS 벡터 캐시에 영속화되므로 중복 디스크 저장 불필요)
            if self.disk_cache and persist_to_disk:
                await self.disk_cache.set(key, value, ttl_seconds)

    async def delete(self, key: str) -> None:
        """전체 레이어에서 삭제"""
        with self.lock:
            if self.memory_cache:
                await self.memory_cache.delete(key)
            if self.semantic_cache:
                await self.semantic_cache.delete(key)
            if self.disk_cache:
                await self.disk_cache.delete(key)

    async def clear(self) -> None:
        """모든 캐시 비우기"""
        with self.lock:
            if self.memory_cache:
                await self.memory_cache.clear()
            if self.semantic_cache:
                await self.semantic_cache.clear()
            if self.disk_cache:
                await self.disk_cache.clear()

    def get_stats(self) -> dict[str, CacheStatistics]:
        """레이어별 통계"""
        stats = {}
        if self.memory_cache:
            stats["memory"] = self.memory_cache.get_stats()
        if self.semantic_cache:
            stats["semantic"] = self.semantic_cache.get_stats()
        if self.disk_cache:
            stats["disk"] = self.disk_cache.get_stats()
        return stats

    def get_combined_stats(self) -> CacheStatistics:
        """통합 통계"""
        combined = CacheStatistics()
        for cache_stats in self.get_stats().values():
            combined.total_hits += cache_stats.total_hits
            combined.total_misses += cache_stats.total_misses
            combined.total_evictions += cache_stats.total_evictions
            combined.total_expirations += cache_stats.total_expirations
            combined.cache_size += cache_stats.cache_size
            combined.total_memory_bytes += cache_stats.total_memory_bytes
        combined.update_hit_rate()
        return combined


# 전역 캐시 관리자 인스턴스
_cache_manager: CacheManager | None = None


def get_cache_manager(
    enable_memory_cache: bool = True,
    enable_semantic_cache: bool = True,
    enable_disk_cache: bool = True,
    embedding_model=None,
) -> CacheManager:
    """캐시 관리자 인스턴스 반환"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(
            enable_memory_cache=enable_memory_cache,
            enable_semantic_cache=enable_semantic_cache,
            enable_disk_cache=enable_disk_cache,
            embedding_model=embedding_model,
        )
    return _cache_manager


def reset_cache_manager() -> None:
    """캐시 관리자 리셋"""
    global _cache_manager
    _cache_manager = None


class ObjectCache(CacheBackend[T]):
    """
    객체 캐시 - 임의의 메모리 파이썬 객체 보관 (컴파일된 그래프, 파싱된
    벡터스토어 등).

    MemoryCache와 동일한 LRU/제거/TTL/락 관용구를 따르되, 값을 pickle/직렬화
    하지 않고 메모리에 그대로 보관한다. CacheBackend ABC를 구현하므로 모든
    메서드는 비동기(async)이다.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 0.0,
    ):
        self.max_size = max_size
        self.default_ttl = ttl_seconds
        self.cache: dict[str, CacheEntry] = {}
        self.lock = RLock()
        self.stats = CacheStatistics()

    async def get(self, key: str) -> T | None:
        """값 조회"""
        with self.lock:
            entry = self.cache.get(key)

            if entry is None:
                self.stats.total_misses += 1
                return None

            if entry.is_expired():
                logger.debug(f"[ObjectCache] 만료된 항목 제거: {key}")
                del self.cache[key]
                self.stats.total_misses += 1
                self.stats.total_expirations += 1
                self.stats.cache_size = len(self.cache)
                return None

            entry.touch()
            self.stats.total_hits += 1
            self.stats.update_hit_rate()

            logger.debug(f"[ObjectCache] 캐시 히트: {key}")
            return entry.value

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 설정 (객체는 직렬화 없이 보관)"""
        with self.lock:
            ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

            # 크기 초과 시 LRU 제거
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            entry = CacheEntry(
                key=key,
                value=value,
                created_at=time.time(),
                accessed_at=time.time(),
                ttl_seconds=ttl,
            )

            self.cache[key] = entry
            self.stats.cache_size = len(self.cache)

            logger.debug(f"[ObjectCache] 값 저장: {key} (TTL: {ttl}초)")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.stats.cache_size = len(self.cache)
                logger.debug(f"[ObjectCache] 값 삭제: {key}")

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.stats.cache_size = 0
            logger.info("[ObjectCache] 캐시 전체 삭제")

    def get_stats(self) -> CacheStatistics:
        """통계 조회 (cache_size 정확도 보장)"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            ages = [entry.get_age() for entry in self.cache.values()]
            self.stats.avg_age_seconds = sum(ages) / len(ages) if ages else 0
            return self.stats

    def _evict_lru(self) -> None:
        """LRU 항목 제거"""
        if not self.cache:
            return

        lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].accessed_at)
        del self.cache[lru_key]
        self.stats.total_evictions += 1
        self.stats.cache_size = len(self.cache)
        logger.debug(f"[ObjectCache] LRU 제거: {lru_key}")


class SyncCacheBridge:
    """
    동기식 브릿지 - 비동기 ObjectCache를 동기 컨텍스트(일반 def)에서 사용.

    Consumer인 VectorStoreCache.load/save는 평범한 def(비동기 아님)라서
    await할 수 없다. 따라서 브릿지만의 전용 이벤트 루프 + 데몬 스레드를
    두고, run_coroutine_threadsafe(...)로 비동기 메서드를 호출한다.

    전용 루프/스레드 방식은 호출자가 동기 컨텍스트에 있든 이미 실행 중인
    다른 이벤트 루프 안에 있든 어디서든 안전하다. (loop.run_until_complete는
    호출자가 이미 실행 중인 루프 안에 있으면 "This event loop is already
    running" RuntimeError를 발생시키므로 사용하지 않는다.)
    """

    def __init__(self, cache: ObjectCache[T]):
        self._cache: ObjectCache[Any] = cache
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="SyncCacheBridge"
        )
        self._thread.start()

    def _run(self, coro: Coroutine[Any, Any, R]) -> R:
        """전용 루프/스레드에서 코루틴 실행 (호출자 루프 컨텍스트 무관)."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close_sync(self) -> None:
        """브릿지 전용 루프/스레드 정리."""
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            # 스레드가 run_forever를 빠져나올 때까지 대기 후 close().
            self._thread.join(timeout=5.0)
        if not self._loop.is_closed():
            self._loop.close()

    def __del__(self) -> None:
        # __del__ 내부 리소스 정리 보조: 예외는 삼킨다.
        with contextlib.suppress(Exception):
            self.close_sync()

    def get_sync(self, key: str) -> T | None:
        """동기 조회"""
        result: T | None = self._run(self._cache.get(key))
        return result

    def set_sync(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """동기 설정"""
        self._run(self._cache.set(key, value, ttl_seconds))

    def delete_sync(self, key: str) -> None:
        """동기 삭제"""
        self._run(self._cache.delete(key))

    def clear_sync(self) -> None:
        """동기 전체 삭제"""
        self._run(self._cache.clear())
