import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.model_loader import ModelManager
from common.config import AVAILABLE_EMBEDDING_MODELS

# 테스트 데이터 생성 (약 500개의 문장)
SAMPLE_TEXTS = [
    "This is a sample sentence for embedding performance benchmark.",
    "Artificial intelligence is transforming the way we process information.",
    "Efficient vector search requires high-quality embeddings and optimized indexes.",
    "NumPy and PyTorch are essential tools for data science and machine learning.",
    "The transformer architecture has become the standard for NLP tasks."
] * 100 

MODEL_NAME = AVAILABLE_EMBEDDING_MODELS[0]

def run_pure_embedding_benchmark(device_type):
    print("\n>>> Pure Embedding Test on: [%s]" % device_type.upper())
    
    # 모델 로드
    os.environ["EMBEDDING_DEVICE"] = device_type
    ModelManager._instances.clear()
    
    start = time.time()
    embedder = ModelManager.get_embedder(MODEL_NAME)
    load_time = time.time() - start
    print("   Model Load Time: %.2fs" % load_time)

    # 대량 문서 임베딩 (Indexing 시뮬레이션)
    start = time.time()
    _ = embedder.embed_documents(SAMPLE_TEXTS)
    total_time = time.time() - start
    
    avg_per_doc = (total_time / len(SAMPLE_TEXTS)) * 1000
    print("   Total Time (500 docs): %.4fs" % total_time)
    print("   Avg Latency per Doc: %.2fms" % avg_per_doc)

    # 단일 쿼리 임베딩 (Search/Router 시뮬레이션)
    start = time.time()
    for _ in range(50):
        _ = embedder.embed_query("What is the main contribution of this paper?")
    query_latency = (time.time() - start) / 50 * 1000
    print("   Single Query Latency: %.2fms" % query_latency)

    return {
        "device": device_type,
        "total": total_time,
        "query": query_latency
    }

def main():
    print("============================================================")
    print("⚡ Pure Embedding Performance: GPU vs CPU")
    print("============================================================")
    
    results = []
    
    if torch.cuda.is_available():
        res_gpu = run_pure_embedding_benchmark("cuda")
        results.append(res_gpu)
    
    res_cpu = run_pure_embedding_benchmark("cpu")
    results.append(res_cpu)

    if len(results) == 2:
        gpu, cpu = results[0], results[1]
        print("\n============================================================")
        print("📊 Performance Comparison Summary")
        print("------------------------------------------------------------")
        print("Massive Indexing (500 docs): CPU %.2fs | GPU %.2fs" % (cpu['total'], gpu['total']))
        print("Indexing Speedup: %.2fx" % (cpu['total']/gpu['total']))
        print("Single Query Latency: CPU %.2fms | GPU %.2fms" % (cpu['query'], gpu['query']))
        print("Query Speedup: %.2fx" % (cpu['query']/gpu['query']))
        print("============================================================")

if __name__ == "__main__":
    main()
