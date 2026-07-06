# tests/performance/test_inference_concurrency.py
import asyncio
import time
import pytest
from src.core.model_loader import ModelManager

@pytest.mark.asyncio
async def test_parallel_inference_semaphore():
    """세마포어가 병렬 요청을 허용하는지 확인합니다."""
from src.core.model_loader import ModelManager
    from common.config import MAX_CONCURRENT_INFERENCE
    import threading
    import time
    
    # 세마포어 강제 업데이트 (테스트 환경 보장)
    ModelManager.update_inference_limit(MAX_CONCURRENT_INFERENCE)
    semaphore = ModelManager._inference_semaphore
    
    def worker(id, results):
        if semaphore.acquire(timeout=2):
            try:
                time.sleep(1.0) # 실제 추론 시간 시뮬레이션
                results.append(id)
            finally:
                semaphore.release()
        else:
            results.append(f"timeout_{id}")

    results = []
    t1 = threading.Thread(target=worker, args=(1, results))
    t2 = threading.Thread(target=worker, args=(2, results))
    
    start = time.time()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.time() - start
    
    print(f"Elapsed: {elapsed:.2f}s, Results: {results}")
    
    # MAX_CONCURRENT_INFERENCE=1이면 최소 2초 소요 (직렬)
    # 0.8초 미만이면 통과 (현재 실패를 유도하기 위해 1.5초로 설정)
    assert elapsed < 1.5, f"Inference is serialized (Elapsed: {elapsed:.2f}s)"
