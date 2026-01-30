import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from cache.response_cache import ResponseCache
from core.session import SessionManager


async def test_cache_isolation_between_documents():
    """문서 변경 시 캐시가 격리되는지 테스트합니다."""

    # Mock CacheManager
    mock_manager = MagicMock()
    mock_manager.get = asyncio.Future()
    mock_manager.get.set_result(None)  # 초기에는 캐시 미스
    mock_manager.set = asyncio.Future()
    mock_manager.set.set_result(None)

    with patch("cache.response_cache.get_cache_manager", return_value=mock_manager):
        cache = ResponseCache()
        query = "이 문서 요약해줘"

        # 1. 문서 A 세션
        SessionManager.init_session("user1")
        SessionManager.set("pdf_file_path", "doc_A.pdf")

        key_A = cache._generate_key(query)
        await cache.set(query, "문서 A의 요약입니다.")

        # 2. 문서 B 세션 (동일 유저, 다른 문서)
        SessionManager.set("pdf_file_path", "doc_B.pdf")
        key_B = cache._generate_key(query)

        print(f"Key A: {key_A}")
        print(f"Key B: {key_B}")

        assert key_A != key_B, "문서가 다른데 캐시 키가 동일합니다! (결함 발생)"
        print("✅ 문서 간 캐시 격리 성공")


if __name__ == "__main__":
    asyncio.run(test_cache_isolation_between_documents())
