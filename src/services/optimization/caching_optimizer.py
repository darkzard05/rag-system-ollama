"""
캐싱 최적화 - Task 13
응답 캐싱, 세맨틱 캐싱, TTL 관리, 캐시 일관성
"""

import hashlib
import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Generic, TypeVar

import numpy as np

from common.config import (
    CACHE_CHECK_PERMISSIONS,
    CACHE_HMAC_SECRET,
    CACHE_SECURITY_LEVEL,
    CACHE_TRUSTED_PATHS,
)
from security.cache_security import CacheSecurityManager
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

# 타입 변수
T = TypeVar("T")


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
            elif hasattr(new_value, "__len__"):
                estimated_bytes = len(new_value) * 8  # 대략적인 포인터 크기
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

    @property
    def cache_size(self) -> int:
        """캐시 크기"""
        return len(self.cache)

    async def get(
        self, query: str, similarity_threshold: float | None = None
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
                query_embedding = await self._embed(query)
                query_embedding = query_embedding / (
                    np.linalg.norm(query_embedding) + 1e-10
                )

                # [최적화] 모든 캐시된 벡터를 하나의 행렬로 구성하여 한 번에 행렬 연산 수행
                keys = list(self.embeddings.keys())
                # 이미 정규화된 상태로 저장되어 있다고 가정 (set에서 처리)
                cached_matrix = np.array([self.embeddings[k] for k in keys])

                # 코사인 유사도 계산 (행렬-벡터 내적)
                similarities = np.dot(cached_matrix, query_embedding)

                # 가장 유사한 항목 찾기
                max_idx = np.argmax(similarities)
                best_similarity = similarities[max_idx]
                best_match = keys[max_idx]

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

    async def set(self, query: str, value: T, ttl_seconds: float = 0) -> None:
        """값 저장 및 벡터 정규화"""
        with self.lock:
            try:
                ttl = ttl_seconds if ttl_seconds > 0 else self.default_ttl

                if len(self.cache) >= self.max_entries:
                    self._evict_oldest()

                cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]

                query_embedding = None
                if self.embedding_model:
                    query_embedding = await self._embed(query)
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
                    metadata={"query": query[:100]},
                )

                self.cache[cache_key] = entry
                if query_embedding is not None:
                    self.embeddings[cache_key] = query_embedding
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
            self.stats.cache_size = len(self.cache)

    async def clear(self) -> None:
        """전체 캐시 삭제"""
        with self.lock:
            self.cache.clear()
            self.embeddings.clear()
            self.stats.cache_size = 0
            logger.info("[SemanticCache] 캐시 전체 삭제")

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.cache_size = len(self.cache)
            self.stats.update_hit_rate()
            return self.stats

    async def _embed(self, text: str) -> np.ndarray:
        """텍스트 임베딩"""
        # 실제 구현에서는 임베딩 모델 사용
        # 현재는 간단한 시뮬레이션
        if hasattr(self.embedding_model, "embed_query"):
            embedding = await self.embedding_model.embed_query(text)
            return np.array(embedding)

        # 폴백: 간단한 해시 기반 벡터
        hash_obj = hashlib.sha256(text.encode())
        hash_int = int(hash_obj.hexdigest(), 16)
        np.random.seed(hash_int % (2**32))
        return np.random.randn(384)

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거"""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)

        del self.cache[oldest_key]
        if oldest_key in self.embeddings:
            del self.embeddings[oldest_key]

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

        # 보안 관리자 초기화
        self.security_manager = CacheSecurityManager(
            security_level=CACHE_SECURITY_LEVEL,
            hmac_secret=CACHE_HMAC_SECRET,
            trusted_paths=CACHE_TRUSTED_PATHS + [str(self.cache_dir)],
            check_permissions=CACHE_CHECK_PERMISSIONS,
        )

    def _get_cache_path(self, key: str) -> Path:
        """키에 대한 파일 경로 생성"""
        hashed_key = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{hashed_key}.cache"

    async def get(self, key: str) -> T | None:
        """값 조회 (보안 검증 포함)"""
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
                    logger.critical(
                        f"[DiskCache] 보안 위협 감지: {error}. 캐시를 삭제합니다."
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
                self.stats.total_misses += 1
                return None

    async def set(self, key: str, value: T, ttl_seconds: float = 0) -> None:
        """값 저장 (보안 메타데이터 생성 및 권한 강제 포함)"""
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

                # 1. 파일 저장
                with open(cache_file, "wb") as f:
                    pickle.dump(entry, f)

                # 2. 파일 권한 강제 적용
                self.security_manager.enforce_file_permissions(str(cache_file))

                # 3. 보안 메타데이터 생성 및 저장
                metadata = self.security_manager.create_metadata_for_file(
                    str(cache_file), description=f"Cache entry: {key[:30]}"
                )
                self.security_manager.save_cache_metadata(
                    str(cache_file) + ".meta", metadata
                )

                self.stats.cache_size = len(list(self.cache_dir.glob("*.cache")))

            except Exception as e:
                logger.error(f"[DiskCache] 저장 오류: {e}")

    async def delete(self, key: str) -> None:
        """값 삭제"""
        with self.lock:
            self._delete_file(self._get_cache_path(key))

    async def clear(self) -> None:
        """전체 삭제"""
        import contextlib

        with self.lock:
            for f in self.cache_dir.glob("*.cache*"):
                with contextlib.suppress(Exception):
                    f.unlink()
            self.stats.cache_size = 0

    def get_stats(self) -> CacheStatistics:
        """통계 조회"""
        with self.lock:
            self.stats.cache_size = len(list(self.cache_dir.glob("*.cache")))
            self.stats.update_hit_rate()
            return self.stats

    def _delete_file(self, path: Path) -> None:
        """파일 및 메타데이터 삭제"""
        try:
            if path.exists():
                path.unlink()
            meta = Path(str(path) + ".meta")
            if meta.exists():
                meta.unlink()
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
        with monitor.track_operation(
            OperationType.QUERY_PROCESSING,
            {"stage": "cache_lookup", "semantic": use_semantic},
        ) as op:
            # 1. L1 메모리 캐시 확인 (가장 빠름)
            if self.memory_cache and not use_semantic:
                result = await self.memory_cache.get(key)
                if result is not None:
                    logger.debug(f"[CacheManager] L1 히트: {key}")
                    op.metadata = {"cache_level": "L1"}
                    return result

            # 2. L2 세맨틱 캐시 확인 (의미적 유사성)
            if self.semantic_cache and use_semantic:
                result = await self.semantic_cache.get(key)
                if result is not None:
                    logger.debug(f"[CacheManager] L2 히트: {key}")
                    op.metadata = {"cache_level": "L2"}

                    # L1으로 승격 (Promotion)
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            # 3. L3 디스크 캐시 확인 (영구 저장소)
            if self.disk_cache and not use_semantic:
                result = await self.disk_cache.get(key)
                if result is not None:
                    logger.debug(f"[CacheManager] L3 히트: {key}")
                    op.metadata = {"cache_level": "L3"}

                    # L1으로 승격
                    if self.memory_cache:
                        await self.memory_cache.set(key, result)

                    return result

            logger.debug(f"[CacheManager] 캐시 미스: {key}")
            return None

            logger.debug(f"[CacheManager] 캐시 미스: {key}")
            return None

    async def set(
        self, key: str, value: Any, ttl_seconds: float = 0, use_semantic: bool = False
    ) -> None:
        """값 저장 (L1, L2, L3)"""
        with self.lock:
            # L1에 저장
            if self.memory_cache:
                await self.memory_cache.set(key, value, ttl_seconds)

            # L2 (세맨틱) 저장
            if self.semantic_cache and use_semantic:
                await self.semantic_cache.set(key, value, ttl_seconds)

            # L3 (디스크) 저장 - 세맨틱이 아닐 때만 저장하거나 정책에 따라 결정
            # 여기서는 모든 실답변을 디스크에 백업함
            if self.disk_cache:
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
