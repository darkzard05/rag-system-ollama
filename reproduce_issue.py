import asyncio
import unittest.mock as mock

import pytest
from src.core.model_loader import ModelManager


@pytest.mark.asyncio
async def test_reproduce_concurrency_issue():
    """재현 테스트: 여러 태스크가 동시에 모델을 요청할 때 load_llm이 여러 번 호출되는지 확인"""
    mock_llm = mock.MagicMock()

    async def slow_load(model_name):
        await asyncio.sleep(0.1)
        return mock_llm

    # core.model_loader.load_llm을 비동기 함수로 패치
    with mock.patch("core.model_loader.load_llm", side_effect=slow_load) as mock_load:
        tasks = [ModelManager.get_llm("test_model") for _ in range(5)]
        results = await asyncio.gather(*tasks)

        print(f"\nCall count: {mock_load.call_count}")
        print(f"Results: {results}")

        assert all(r == mock_llm for r in results)
        assert mock_load.call_count == 1


if __name__ == "__main__":
    asyncio.run(test_reproduce_concurrency_issue())
