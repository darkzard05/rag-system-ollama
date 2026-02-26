import asyncio
import pytest
import unittest.mock as mock
from core.model_loader import ModelManager
from common.config import MAX_CONCURRENT_INFERENCE

@pytest.mark.asyncio
async def test_model_manager_concurrency():
    """여러 태스크가 동시에 모델을 요청할 때 락이 정상 작동하는지 테스트"""
    mock_llm = mock.MagicMock()
    with mock.patch("core.model_loader.load_llm", return_value=mock_llm) as mock_load:
        tasks = [ModelManager.get_llm("test_model") for _ in range(5)]
        results = await asyncio.gather(*tasks)
        assert all(r == mock_llm for r in results)
        assert mock_load.call_count == 1

@pytest.mark.asyncio
async def test_inference_semaphore_limit():
    """세마포어가 동시 추론 수를 제한하는지 테스트"""
    ModelManager._inference_semaphore = None 
    active_tasks = 0
    max_observed_tasks = 0

    async def simulated_inference():
        nonlocal active_tasks, max_observed_tasks
        await ModelManager.acquire_inference_lock()
        try:
            active_tasks += 1
            max_observed_tasks = max(max_observed_tasks, active_tasks)
            await asyncio.sleep(0.1)
        finally:
            active_tasks -= 1
            ModelManager.release_inference_lock()

    num_tasks = MAX_CONCURRENT_INFERENCE + 2
    await asyncio.gather(*(simulated_inference() for _ in range(num_tasks)))
    assert max_observed_tasks <= MAX_CONCURRENT_INFERENCE
    assert active_tasks == 0

@pytest.mark.asyncio
async def test_clear_vram_async():
    """비동기 clear_vram이 캐시를 올바르게 비우는지 테스트"""
    with mock.patch("core.model_loader.load_llm", return_value=mock.MagicMock()):
        await ModelManager.get_llm("to_be_cleared")
        assert any("to_be_cleared" in k for k in ModelManager._instances.keys())
        await ModelManager.clear_vram()
        assert len(ModelManager._instances) == 0

@pytest.mark.asyncio
async def test_async_client_singleton_per_loop():
    """비동기 클라이언트가 루프당 하나씩 관리되는지 테스트"""
    # 초기화
    ModelManager._async_client = None
    ModelManager._client_loop = None
    
    with mock.patch("ollama.AsyncClient") as mock_client_cls:
        # 모킹된 객체에 base_url 속성 설정 (구현부의 비교 로직 대응)
        host = "http://localhost:11434"
        mock_instance = mock.AsyncMock()
        mock_instance.base_url = host
        mock_client_cls.return_value = mock_instance
        
        client1 = await ModelManager.get_async_client(host)
        client2 = await ModelManager.get_async_client(host)
        
        # 동일한 루프에서는 같은 클라이언트여야 함
        assert client1 == client2
        assert mock_client_cls.call_count == 1
