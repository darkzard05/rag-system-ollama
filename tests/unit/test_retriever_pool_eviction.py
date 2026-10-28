import pytest
from unittest.mock import MagicMock, patch
from core.resource_manager import RetrieverPool


@pytest.mark.asyncio
async def test_retriever_pool_avoids_cascade_wipeout_under_high_memory_pressure():
    """
    [RED->GREEN] 외부 프로세스에 의해 시스템 메모리가 80% 이하로 떨어지지 않더라도
    check_memory_pressure()가 풀 내의 모든 리트리버를 전멸시키지 않고
    단 1개만 점진적으로 퇴출(Stepwise Single Eviction)하는지 검증한다.
    """
    # 1. 풀 설정 (최대 5개 유지 가능)
    pool = RetrieverPool("TestRetriever", item_limit=5, byte_limit=100 * 1024 * 1024)

    # 2. 3개 리트리버 등록 (가상 FAISS 인덱스 구조)
    for i in range(1, 4):
        fake_retriever = MagicMock()
        fake_retriever.index.ntotal = 100
        fake_retriever.index.d = 768
        await pool.put(f"hash_{i}", (fake_retriever, None))

    assert len(pool._pool) == 3

    # 3. psutil.virtual_memory()가 지속적으로 86.0%를 반환하도록 모킹 (Ollama 점유 시뮬레이션)
    fake_mem = MagicMock()
    fake_mem.percent = 86.0

    with (
        patch("psutil.virtual_memory", return_value=fake_mem),
        patch("core.resource_manager.eviction_allowed", return_value=True),
        patch("gc.collect") as mock_gc,
    ):
        # 4. 메모리 압박 체크 실행
        evicted = await pool.check_memory_pressure()

        # 5. 통과 조건 검증 (GREEN)
        assert evicted is True
        # As-Is에서는 루프가 돌아 1개만 남거나 0개가 됨 (RED: len == 1)
        # To-Be에서는 가장 오래된 hash_1 하나만 퇴출되어 2개가 유지되어야 함 (GREEN: len == 2)
        assert len(pool._pool) == 2, (
            f"Expected 2 retrievers remaining, but found {len(pool._pool)} (Cascade wipeout detected!)"
        )
        assert "hash_1" not in pool._pool
        assert "hash_2" in pool._pool
        assert "hash_3" in pool._pool
        mock_gc.assert_called_once()
