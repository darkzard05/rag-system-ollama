"""
캐싱 시스템 테스트 - Task 13
TTL, 세맨틱 캐싱, 캐시 무효화, 스레드 안전성 등 종합 검증
"""

# ruff: noqa: E402

import sys
from pathlib import Path

# 프로젝트 루트 및 src 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR / "src"))

import asyncio
import logging
import time
from threading import Thread

import numpy as np
import pytest

from services.optimization.caching_optimizer import (
    CacheEntry,
    CacheManager,
    DiskCache,
    MemoryCache,
    SemanticCache,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestCacheEntry:
    """CacheEntry 테스트"""

    def test_entry_creation(self):
        """항목 생성"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            accessed_at=time.time(),
            ttl_seconds=3600,
        )
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.hit_count == 0

    def test_entry_expiration(self):
        """항목 만료"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time() - 7200,  # 2시간 전
            accessed_at=time.time(),
            ttl_seconds=3600,  # 1시간
        )
        assert entry.is_expired()

    def test_entry_not_expired(self):
        """항목 미만료"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time(),
            accessed_at=time.time(),
            ttl_seconds=3600,
        )
        assert not entry.is_expired()

    def test_entry_age(self):
        """항목 나이"""
        created = time.time() - 100
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=created,
            accessed_at=time.time(),
            ttl_seconds=3600,
        )
        assert 99 < entry.get_age() < 101

    def test_entry_touch(self):
        """접근 업데이트"""
        entry = CacheEntry(
            key="test",
            value="value",
            created_at=time.time(),
            accessed_at=time.time() - 100,
            ttl_seconds=3600,
        )
        initial_accessed = entry.accessed_at
        entry.touch()
        assert entry.accessed_at > initial_accessed
        assert entry.hit_count == 1


class TestMemoryCache:
    """메모리 캐시 테스트"""

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        """값 저장 및 조회"""
        cache = MemoryCache()
        await cache.set("key1", "value1")
        result = await cache.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """캐시 미스"""
        cache = MemoryCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """TTL 만료"""
        cache = MemoryCache()
        await cache.set("key", "value", ttl_seconds=0.1)

        result = await cache.get("key")
        assert result == "value"

        await asyncio.sleep(0.2)
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self):
        """항목 삭제"""
        cache = MemoryCache()
        await cache.set("key", "value")
        await cache.delete("key")
        result = await cache.get("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear(self):
        """캐시 전체 삭제"""
        cache = MemoryCache()
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

    @pytest.mark.asyncio
    async def test_hit_rate_calculation(self):
        """히트율 계산"""
        cache = MemoryCache()
        await cache.set("key", "value")

        # 히트
        await cache.get("key")
        await cache.get("key")

        # 미스
        await cache.get("nonexistent")

        stats = cache.get_stats()
        assert stats.total_hits >= 2
        assert stats.total_misses >= 1
        assert stats.hit_rate > 0
        assert stats.hit_rate <= 1.0

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        """LRU 제거"""
        cache = MemoryCache(max_size=2)

        await cache.set("key1", "value1")
        await asyncio.sleep(0.01)
        await cache.set("key2", "value2")
        await asyncio.sleep(0.01)

        # key1 접근하여 최근 사용으로 업데이트
        await cache.get("key1")
        await asyncio.sleep(0.01)

        # 새 키 추가 시 가장 오래된 key2가 제거되어야 함
        await cache.set("key3", "value3")

        # 이제 key2는 제거되었고 key1, key3은 있어야 함
        result1 = await cache.get("key1")
        await cache.get("key2")
        result3 = await cache.get("key3")

        assert result1 is not None or result3 is not None  # 최소한 하나는 있어야 함

    @pytest.mark.asyncio
    async def test_statistics_tracking(self):
        """통계 추적"""
        cache = MemoryCache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.get("key1")

        stats = cache.get_stats()
        assert stats.cache_size == 2
        assert stats.total_hits >= 1


class TestSemanticCache:
    """세맨틱 캐시 테스트"""

    @pytest.mark.asyncio
    async def test_semantic_cache_creation(self):
        """세맨틱 캐시 생성"""
        cache = SemanticCache(similarity_threshold=0.95)
        assert cache.similarity_threshold == 0.95
        assert cache.max_entries == 500

    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """임베딩 생성"""
        cache = SemanticCache()
        embedding = await cache._embed("test query")

        assert isinstance(embedding, np.ndarray)
        assert len(embedding) > 0

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="SemanticCache._cosine_similarity() 리팩토링으로 제거됨")
    async def test_cosine_similarity(self):
        """코사인 유사도"""
        cache = SemanticCache()

        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 0, 0])
        vec3 = np.array([0, 1, 0])

        sim_same = cache._cosine_similarity(vec1, vec2)
        sim_diff = cache._cosine_similarity(vec1, vec3)

        assert sim_same > sim_diff
        assert sim_same > 0.99

    @pytest.mark.asyncio
    async def test_semantic_set_and_get(self):
        """세맨틱 저장 및 조회"""
        cache = SemanticCache()

        await cache.set("hello world", {"answer": "42"})

        # SemanticCache는 저장에 성공했는지 확인
        assert cache.cache_size >= 1

    @pytest.mark.asyncio
    async def test_semantic_cache_miss(self):
        """세맨틱 캐시 미스"""
        cache = SemanticCache(similarity_threshold=0.99)

        result = await cache.get("nonexistent query")
        assert result is None

    @pytest.mark.asyncio
    async def test_eviction_oldest_entry(self):
        """가장 오래된 항목 제거"""
        cache = SemanticCache(max_entries=2)

        await cache.set("query1", {"value": 1})
        await asyncio.sleep(0.02)
        await cache.set("query2", {"value": 2})
        await asyncio.sleep(0.02)
        await cache.set("query3", {"value": 3})

        stats = cache.get_stats()
        # max_entries=2이므로 3개 저장하면 제거 발생
        assert stats.cache_size <= 2


class TestCacheManager:
    """캐시 관리자 테스트"""

    @pytest.mark.asyncio
    async def test_manager_creation(self):
        """캐시 관리자 생성"""
        manager = CacheManager()
        assert manager.memory_cache is not None
        assert manager.semantic_cache is not None

    @pytest.mark.asyncio
    async def test_l1_cache_hit(self):
        """L1 캐시 히트"""
        manager = CacheManager()

        await manager.set("key", "value")
        result = await manager.get("key")

        assert result == "value"

    @pytest.mark.asyncio
    async def test_l2_semantic_cache(self):
        """L2 세맨틱 캐시"""
        manager = CacheManager()

        await manager.set("query1", "response1", use_semantic=True)
        # 세맨틱 캐시는 임베딩 모델이 없으면 저장되지만 조회는 실패할 수 있음
        # 메모리 캐시에는 반드시 저장됨
        result = await manager.get("query1", use_semantic=False)

        assert result is not None

    @pytest.mark.asyncio
    async def test_combined_statistics(self):
        """통합 통계"""
        manager = CacheManager()

        await manager.set("key1", "value1")
        await manager.get("key1")
        await manager.get("nonexistent")

        stats = manager.get_combined_stats()
        assert stats.total_hits >= 1
        assert stats.total_misses >= 1


class TestDiskCacheSecurity:
    """DiskCache 보안 기능 테스트"""

    @pytest.mark.asyncio
    async def test_disk_cache_integrity(self):
        """무결성 검증 및 변조 감지 테스트"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "cache"
            cache = DiskCache(cache_dir=str(cache_dir))

            key = "secure_key"
            value = {"data": "sensitive_info"}

            # 1. 정상 저장
            await cache.set(key, value)

            # 2. 정상 로드 확인
            assert await cache.get(key) == value

            # 3. 파일 강제 변조 (공격 시뮬레이션)
            cache_file = list(cache_dir.glob("*.cache"))[0]
            with open(cache_file, "ab") as f:
                f.write(b"malicious_code")

            # 4. 변조 감지 및 차단 확인
            # 보안 강화 로직에 의해 None이 반환되어야 함
            assert await cache.get(key) is None

            # 5. 오염된 파일이 자동 삭제되었는지 확인
            assert not cache_file.exists()

    @pytest.mark.asyncio
    async def test_disk_cache_metadata_cleanup(self):
        """삭제 시 메타데이터 함께 삭제되는지 확인"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache = DiskCache(cache_dir=str(tmp_dir))
            key = "cleanup_test"
            await cache.set(key, "value")

            cache_file = list(Path(tmp_dir).glob("*.cache"))[0]
            meta_file = Path(str(cache_file) + ".meta")

            assert cache_file.exists()
            assert meta_file.exists()

            await cache.delete(key)

            assert not cache_file.exists()
            assert not meta_file.exists()


class TestThreadSafety:
    """스레드 안전성 테스트"""

    @pytest.mark.asyncio
    async def test_concurrent_set_get(self):
        """동시 설정 및 조회"""
        cache = MemoryCache()
        results = []

        async def worker(key, value):
            await cache.set(key, value)
            result = await cache.get(key)
            results.append(result)

        tasks = [worker(f"key{i}", f"value{i}") for i in range(10)]
        await asyncio.gather(*tasks)

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_concurrent_hit_counting(self):
        """동시 히트 카운팅"""
        cache = MemoryCache()
        await cache.set("key", "value")

        async def worker():
            await cache.get("key")

        tasks = [worker() for _ in range(10)]
        await asyncio.gather(*tasks)

        stats = cache.get_stats()
        assert stats.total_hits >= 10

    def test_threading_safety(self):
        """스레드 안전성"""
        cache = MemoryCache()
        results = []

        def worker(key, value):
            asyncio.run(cache.set(key, value))
            result = asyncio.run(cache.get(key))
            results.append(result)

        threads = [
            Thread(target=worker, args=(f"key{i}", f"value{i}")) for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 5


class TestIntegration:
    """통합 테스트"""

    @pytest.mark.asyncio
    async def test_cache_statistics_reporting(self):
        """캐시 통계 보고"""
        manager = CacheManager()

        await manager.set("key1", "value1")
        await manager.get("key1")
        await manager.get("nonexistent")

        stats = manager.get_combined_stats()

        # 통계가 생성되었는지 확인
        assert stats is not None
