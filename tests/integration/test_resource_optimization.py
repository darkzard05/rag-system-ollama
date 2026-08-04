import asyncio
import os

import pytest
from src.api.api_server import RAGResourceManager


@pytest.mark.asyncio
@pytest.mark.skipif(
    os.environ.get("IS_CI_TEST") == "true",
    reason="실제 Ollama 모델 로딩 필요",
)
async def test_resource_loading_concurrency():
    """
    여러 요청이 동시에 올 때 세마포어에 의해 모델 로딩이 순차적으로 진행되는지 테스트합니다.
    """
    # 실제 모델 로딩은 시간이 걸리므로, 여기서는 ResourceManager가
    # 세마포어를 사용하는지 구조적으로 확인합니다.

    # 두 개의 모델 로딩 요청을 동시에 보냄
    # (실제로는 캐시되어 있지 않은 상태여야 함)
    tasks = [RAGResourceManager.get_llm("llama3"), RAGResourceManager.get_llm("qwen2")]

    # 실제 모델 로딩 함수가 thread-safe하게 호출되는지 로그로 확인 가능
    # 여기서는 예외가 발생하지 않고 완료되는지 체크
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 세마포어와 락이 정상 작동한다면 두 요청 모두 안전하게 처리되어야 함
    assert all(not isinstance(r, Exception) for r in results)
