
import time
import numpy as np
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.query_optimizer import RAGQueryOptimizer
from core.reranker import DiversityCalculator, RerankingResult

# --- [OLD LOGIC REPRODUCTION] ---

def old_find_breakpoints(similarities, window_size=3, threshold=0.5):
    breakpoints = []
    for i, sim in enumerate(similarities):
        is_global_break = sim < threshold
        is_local_break = False
        if i >= window_size:
            local_avg = np.mean(similarities[max(0, i - window_size) : i])
            if sim < local_avg * 0.8:
                is_local_break = True
        if is_global_break or is_local_break:
            breakpoints.append(i + 1)
    return breakpoints

def old_intent_logic(q_vec, sample_dict):
    best_intent = "FACTOID"
    max_sim = -1.0
    for intent, s_vecs in sample_dict.items():
        for sv in s_vecs:
            sim = np.dot(q_vec, sv) / (np.linalg.norm(q_vec) * np.linalg.norm(sv))
            if sim > max_sim:
                max_sim = sim
                best_intent = intent
    return best_intent

def old_diversity_logic(content, selected_contents):
    if not selected_contents: return 0.0
    similarities = []
    for sel_content in selected_contents:
        common = len(set(content) & set(sel_content))
        max_len = max(len(content), len(sel_content))
        similarities.append(common / max_len if max_len > 0 else 0.0)
    return sum(similarities) / len(similarities)

# --- [BENCHMARK RUNNER] ---

def run_comparison():
    print(f"{'Operation':<25} | {'Old (ms)':<12} | {'New (ms)':<12} | {'Speedup':<10}")
    print("-" * 70)

    # 1. Semantic Chunker (1,000 items)
    sims = np.random.rand(1000).tolist()
    chunker = EmbeddingBasedSemanticChunker(embedder=None)
    
    t0 = time.perf_counter()
    for _ in range(100): old_find_breakpoints(sims)
    t_old = (time.perf_counter() - t0) / 100 * 1000
    
    t1 = time.perf_counter()
    for _ in range(100): chunker._find_breakpoints(sims)
    t_new = (time.perf_counter() - t1) / 100 * 1000
    
    print(f"{'Semantic Chunking':<25} | {t_old:10.4f} | {t_new:10.4f} | {t_old/t_new:8.1f}x")

    # 2. Query Router (Matrix vs Loop)
    dim = 384
    q_vec = np.random.rand(dim).astype("float32")
    q_norm = np.linalg.norm(q_vec)
    
    # 3가지 의도, 각 10개의 샘플
    sample_dict = {f"intent_{i}": [np.random.rand(dim).astype("float32") for _ in range(10)] for i in range(3)}
    
    # New logic uses matrix cache
    RAGQueryOptimizer._sample_vectors_cache = {k: np.array(v) for k, v in sample_dict.items()}
    
    t0 = time.perf_counter()
    for _ in range(500): old_intent_logic(q_vec, sample_dict)
    t_old = (time.perf_counter() - t0) / 500 * 1000
    
    t1 = time.perf_counter()
    for _ in range(500):
        for intent, s_matrix in RAGQueryOptimizer._sample_vectors_cache.items():
            dot_products = s_matrix @ q_vec
            s_norms = np.linalg.norm(s_matrix, axis=1)
            sims = dot_products / (s_norms * q_norm)
            np.max(sims)
    t_new = (time.perf_counter() - t1) / 500 * 1000
    
    print(f"{'Query Routing':<25} | {t_old:10.4f} | {t_new:10.4f} | {t_old/t_new:8.1f}x")

    # 3. Reranker (Set Caching)
    content = "This is a sample content for reranking " * 20
    selected_contents = ["This is selected content number " + str(i) for i in range(50)]
    
    calc = DiversityCalculator()
    results = [RerankingResult(doc_id=f"doc_{i}", content=c, original_score=0.9, reranked_score=0.9, original_rank=i, final_rank=i) for i, c in enumerate(selected_contents)]
    target = RerankingResult(doc_id="target", content=content, original_score=0.9, reranked_score=0.9, original_rank=0, final_rank=0)

    t0 = time.perf_counter()
    for _ in range(100): old_diversity_logic(content, selected_contents)
    t_old = (time.perf_counter() - t0) / 100 * 1000
    
    # Warm up cache
    calc.calculate_diversity_penalty(target, results)
    
    t1 = time.perf_counter()
    for _ in range(100): calc.calculate_diversity_penalty(target, results)
    t_new = (time.perf_counter() - t1) / 100 * 1000
    
    print(f"{'Reranker Diversity':<25} | {t_old:10.4f} | {t_new:10.4f} | {t_old/t_new:8.1f}x")

if __name__ == "__main__":
    run_comparison()
