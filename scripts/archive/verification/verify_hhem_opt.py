import asyncio
import time
import torch
from ragas.metrics import FaithfulnesswithHHEM
from ragas.llms import LangchainLLMWrapper
from langchain_ollama import ChatOllama
from ragas.dataset_schema import SingleTurnSample

async def verify_hhem_speed():
    print(f"[*] CUDA Available: {torch.cuda.is_available()}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Using Device: {device}")

    # 가짜 데이터 준비
    sample = SingleTurnSample(
        user_input="CM3 모델의 특징이 뭐야?",
        response="CM3는 텍스트와 이미지를 동시에 처리하는 멀티모달 모델입니다.",
        retrieved_contexts=["CM3는 causally masked model로 텍스트와 이미지 토큰을 생성할 수 있습니다."]
    )

    # 1. 기본 설정 (CPU/순차)
    print("\n[1] Testing Standard Configuration...")
    metric_std = FaithfulnesswithHHEM()
    start = time.perf_counter()
    score = await metric_std.ascore(sample)
    print(f"    - Score: {score}")
    print(f"    - Duration: {time.perf_counter() - start:.4f}s")

    # 2. 최적화 설정 (GPU/Batch)
    print(f"\n[2] Testing Optimized Configuration (Device: {device}, Batch: 16)...")
    metric_opt = FaithfulnesswithHHEM(device=device, batch_size=16)
    start = time.perf_counter()
    score = await metric_opt.ascore(sample)
    print(f"    - Score: {score}")
    print(f"    - Duration: {time.perf_counter() - start:.4f}s")

if __name__ == "__main__":
    asyncio.run(verify_hhem_speed())
