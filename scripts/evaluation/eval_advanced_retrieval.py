import os
import time
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import torch

# 1. 데이터 로드 및 전처리
def load_pdf_and_chunk(pdf_path, chunk_size=500):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # 간단한 청킹
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# 2. 검색 엔진 클래스
class EvalSearchEngine:
    def __init__(self, chunks):
        self.chunks = chunks
        # Vector Engine (Dense)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_embeddings = self.embedder.encode(chunks, convert_to_tensor=True)
        
        # BM25 Engine (Sparse)
        tokenized_corpus = [doc.lower().split() for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def vector_search(self, query, top_k=5):
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(self.chunks)))
        return top_results.indices.tolist(), top_results.values.tolist()

    def bm25_search(self, query, top_k=5):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_n_indices = np.argsort(scores)[::-1][:top_k]
        return top_n_indices.tolist(), [scores[i] for i in top_n_indices]

    def hybrid_search_rrf(self, query, top_k=5, k=60):
        # RRF (Reciprocal Rank Fusion) 알고리즘 적용
        v_indices, _ = self.vector_search(query, top_k=20)
        b_indices, _ = self.bm25_search(query, top_k=20)
        
        rrf_scores = {}
        for rank, idx in enumerate(v_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)
        for rank, idx in enumerate(b_indices):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)
            
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        return sorted_indices[:top_k], [rrf_scores[i] for i in sorted_indices[:top_k]]

# 3. 평가 실행
pdf_path = "tests/data/2201.07520v1.pdf"
print(f"Loading {pdf_path}...")
chunks = load_pdf_and_chunk(pdf_path)
engine = EvalSearchEngine(chunks)

test_queries = [
    "InstructGPT", # 키워드 쿼리
    "How does RLHF improve model safety?", # 문맥 쿼리
    "PPO algorithm details", # 기술 용어
    "What is the main objective of the paper?" # 일반적 질문
]

print("\n=== Search Comparison Results ===")
for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Vector Search
    start = time.time()
    v_idx, _ = engine.vector_search(query)
    v_time = time.time() - start
    
    # BM25 Search
    start = time.time()
    b_idx, _ = engine.bm25_search(query)
    b_time = time.time() - start
    
    # Hybrid Search
    start = time.time()
    h_idx, _ = engine.hybrid_search_rrf(query)
    h_time = time.time() - start
    
    print(f"Vector Top 1: {chunks[v_idx[0]][:100]}... (Time: {v_time:.4f}s)")
    print(f"BM25   Top 1: {chunks[b_idx[0]][:100]}... (Time: {b_time:.4f}s)")
    print(f"Hybrid Top 1: {chunks[h_idx[0]][:100]}... (Time: {h_time:.4f}s)")
    
    # 하이브리드가 벡터와 다른 결과(혹은 보완된 결과)를 냈는지 확인
    overlap = set(v_idx[:3]) & set(b_idx[:3])
    print(f"Overlap (Top 3): {len(overlap)} chunks common between Vector and BM25")
