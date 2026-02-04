import time
import numpy as np
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.query_optimizer import RAGQueryOptimizer
from core.reranker import DiversityCalculator, RerankingResult

def benchmark_semantic_chunker():
    print("\n[1] Semantic Chunker Benchmark")
    # 1000개의 가상 유사도 데이터 생성
    sims = np.random.rand(1000).tolist()
    
    # 더미 임베더
    class DummyEmbedder:
        def embed_documents(self, texts): return [np.random.rand(384) for _ in texts]
    
    chunker = EmbeddingBasedSemanticChunker(embedder=DummyEmbedder())
    
    start = time.perf_counter()
    for _ in range(100):
        chunker._find_breakpoints(sims)
    end = time.perf_counter()
    
    avg_time = (end - start) / 100 * 1000
    print(f"Average execution time (1000 items): {avg_time:.4f}ms")

def benchmark_query_router():
    print("\n[2] Query Router Benchmark")
    
    class DummyEmbedder:
        def embed_documents(self, texts): return np.random.rand(len(texts), 384)
        def embed_query(self, text): return np.random.rand(384)
    
    # 캐시 초기화 및 샘플 생성
    RAGQueryOptimizer._sample_vectors_cache = {}
    embedder = DummyEmbedder()
    
    # 강제로 행렬 캐시 생성 유도 (내부 로직 모방)
    for intent, samples in RAGQueryOptimizer.INTENT_SAMPLES.items():
        RAGQueryOptimizer._sample_vectors_cache[intent] = np.random.rand(len(samples), 384).astype("float32")

    q_vec = np.random.rand(384).astype("float32")
    q_norm = np.linalg.norm(q_vec)
    
    start = time.perf_counter()
    for _ in range(1000):
        # 최적화된 행렬 연산 로직만 테스트
        max_sim = -1.0
        for intent, s_matrix in RAGQueryOptimizer._sample_vectors_cache.items():
            dot_products = s_matrix @ q_vec
            s_norms = np.linalg.norm(s_matrix, axis=1)
            sims = dot_products / (s_norms * q_norm)
            np.max(sims)
    end = time.perf_counter()
    
    avg_time = (end - start) / 1000 * 1000
    print(f"Average execution time (Matrix Ops): {avg_time:.4f}ms")

def benchmark_reranker():
    print("\n[3] Reranker Diversity Benchmark")
    calc = DiversityCalculator()
    
    # 100개의 가상 결과 생성
    results = []
    for i in range(100):
        results.append(RerankingResult(
            doc_id=f"doc_{i}",
            content=" ".join(["word"] * 50) + f" unique_{i}",
            original_score=0.9,
            reranked_score=0.9,
            original_rank=i,
            final_rank=i,
            metadata={"page": i, "source": "test.pdf"}
        ))
    
    # 첫 번째 호출 (캐시 생성 포함)
    calc.calculate_diversity_penalty(results[50], results[:50])
    
    start = time.perf_counter()
    for _ in range(100):
        # 51번째 결과에 대해 이전 50개와의 다양성 계산
        calc.calculate_diversity_penalty(results[50], results[:50])
    end = time.perf_counter()
    
    avg_time = (end - start) / 100 * 1000
    print(f"Average execution time (1 vs 50 docs): {avg_time:.4f}ms")

if __name__ == "__main__":
    benchmark_semantic_chunker()
    benchmark_query_router()
    benchmark_reranker()