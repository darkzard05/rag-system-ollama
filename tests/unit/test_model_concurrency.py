import asyncio
import time
from core.model_loader import ModelManager

async def mock_inference_task(task_id, duration):
    print(f"[{task_id}] 추론 세션 진입 시도...")
    start_time = time.time()
    
    async with ModelManager.inference_session():
        entry_time = time.time()
        print(f"[{task_id}] 추론 세션 획득! (대기 시간: {entry_time - start_time:.2f}s)")
        # 실제 추론을 흉내내는 지연
        await asyncio.sleep(duration)
        print(f"[{task_id}] 추론 완료 및 세션 반환")
        
    return entry_time

async def test_concurrency_control():
    # 강제로 세마포어를 2로 설정 (테스트용)
    ModelManager._inference_semaphore = asyncio.Semaphore(2)
    print(f"Forced MAX_CONCURRENT_INFERENCE for test: 2")
    
    tasks = [
        mock_inference_task(1, 2),
        mock_inference_task(2, 2),
        mock_inference_task(3, 2)
    ]
    
    print("\n--- 동시 추론 테스트 시작 (Max: 2) ---")
    results = await asyncio.gather(*tasks)
    print("--- 동시 추론 테스트 종료 ---\n")
    
    # 분석: 시작 시간이 겹치는지 확인
    results.sort()
    for i in range(len(results) - 1):
        diff = results[i+1] - results[i]
        # 0.1초 미만 차이는 동시 실행으로 간주
        if diff < 0.1: 
            print(f"Task {i+1}와 {i+2}가 동시에 실행되었습니다. (차이: {diff:.2f}s)")
        else:
            print(f"Task {i+1}와 {i+2}가 순차적으로 실행되었습니다. (차이: {diff:.2f}s)")

if __name__ == "__main__":
    asyncio.run(test_concurrency_control())
