import asyncio
import time
import pytest
from src.core.model_loader import ModelManager

@pytest.mark.asyncio
async def test_hotswap_performance():
    model_name = "nomic-embed-text"
    
    # 1. 최초 로드 (실제 모델 로드 비용 발생 가능하나 테스트 환경에서는 가짜 모델일 것)
    await ModelManager.get_embedder(model_name)
    
    # 2. 루프 변경 시뮬레이션 (강제로 _resource_loop_id를 틀리게 설정)
    # 현재는 0이 아니면서 현재 루프와 다르면 초기화가 트리거됨
    ModelManager._resource_loop_id = 999999 
    
    start = time.time()
    # 재호출 시 hotswap 로직 작동
    await ModelManager.get_embedder(model_name)
    duration = time.time() - start
    
    print(f"\nHotswap duration: {duration:.4f}s")
    
    # 목표: 0.5초 미만 (최적화 전에는 전체 모델을 다시 만드느라 오래 걸릴 수 있음)
    # 실제 테스트 환경이 FakeEmbeddings라면 이미 빠를 수 있지만 로직의 정확성을 검증
    assert duration < 0.5
