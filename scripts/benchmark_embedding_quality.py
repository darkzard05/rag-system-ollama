import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from common.config import OLLAMA_BASE_URL

# 테스트 데이터: 한국어와 영어가 혼합된 실제 RAG 시나리오
TEST_CHUNKS = [
    "CM3 모델은 이미지와 텍스트를 동시에 학습하는 멀티모달 아키텍처를 가지고 있습니다.",
    "The model uses a causally masked approach for sequence generation.",
    "이 논문의 핵심 기여는 데이터 효율성을 높인 토큰화 방식에 있습니다.",
    "Ollama provides an easy way to run large language models locally.",
    "한국어 검색 성능을 높이기 위해서는 형태소 분석이나 정교한 임베딩이 필수적입니다.",
] * 20 # 총 100개 청크

MODELS_TO_TEST = [
    {"name": "nomic-embed-text:latest", "type": "ollama (Baseline)"},
    {"name": "BAAI/bge-m3", "type": "hf_gpu (SOTA)"},
    {"name": "intfloat/multilingual-e5-small", "type": "hf_gpu (Efficient)"},
    {"name": "jhgan/ko-sroberta-multitask", "type": "hf_gpu (KR-Specific)"},
]

def benchmark_model(model_info):
    name = model_info["name"]
    m_type = model_info["type"]
    
    print(f"\n[Bench] Testing: {name} ({m_type})")
    
    start_t = time.time()
    try:
        if "ollama" in m_type:
            from langchain_ollama import OllamaEmbeddings
            embedder = OllamaEmbeddings(model=name, base_url=OLLAMA_BASE_URL)
        else:
            from langchain_huggingface import HuggingFaceEmbeddings
            device = "cuda" if torch.cuda.is_available() else "cpu"
            # E5 모델 등은 'query: ', 'passage: ' 접두사가 필요한 경우가 있으나, 최신 인터페이스로 테스트
            embedder = HuggingFaceEmbeddings(
                model_name=name,
                model_kwargs={"device": device},
                encode_kwargs={"device": device, "batch_size": 32}
            )
        load_time = time.time() - start_t
    except Exception as e:
        print(f"   - Load Failed: {e}")
        return None

    # 웜업
    embedder.embed_query("테스트")

    # 속도 측정
    start_t = time.time()
    vectors = embedder.embed_documents(TEST_CHUNKS)
    total_time = time.time() - start_t
    
    avg_latency = (total_time / len(TEST_CHUNKS)) * 1000
    
    print(f"   - Load Time: {load_time:.2f}s")
    print(f"   - Total Time (100 chunks): {total_time:.2f}s")
    print(f"   - Avg Latency: {avg_latency:.2f}ms")
    print(f"   - Vector Dimension: {len(vectors[0])}")
    
    return {
        "name": name,
        "type": m_type,
        "total_time": total_time,
        "avg_latency": avg_latency,
        "dim": len(vectors[0])
    }

def main():
    print("="*80)
    print("Advanced Embedding Benchmark (KR + Multilingual)")
    print("="*80)
    
    results = []
    for m in MODELS_TO_TEST:
        res = benchmark_model(m)
        if res:
            results.append(res)

    print("\n" + "="*80)
    print(f"{'Model Name':<35} | {'Type':<15} | {'Total(s)':<8} | {'Lat(ms)':<8} | {'Dim':<5}")
    print("-" * 80)
    for r in results:
        print(f"{r['name'][:35]:<35} | {r['type']:<15} | {r['total_time']:<8.2f} | {r['avg_latency']:<8.2f} | {r['dim']:<5}")
    print("="*80)

if __name__ == "__main__":
    main()
