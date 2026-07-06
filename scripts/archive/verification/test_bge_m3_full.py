import os
import time
import numpy as np
from pypdf import PdfReader
from FlagEmbedding import BGEM3FlagModel
import torch

# 1. 데이터 로드 및 전처리
def load_pdf_and_chunk(pdf_path, chunk_size=800):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + " "
    
    # BGE-M3는 긴 문맥을 잘 처리하므로 청크 사이즈를 조금 더 크게 설정
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks[:50] # 테스트를 위해 상위 50개 청크만 사용

# 2. BGE-M3 하이브리드 엔진
class BGEM3AdvancedEngine:
    def __init__(self, chunks):
        print("Loading BGE-M3 Model (BAAI/bge-m3)...")
        self.chunks = chunks
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=torch.cuda.is_available())
        
        print("Encoding chunks (Dense + Sparse + ColBERT)... This may take a while.")
        encoded_output = self.model.encode(
            chunks, 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=True
        )
        self.chunk_dense = encoded_output['dense_vecs']
        self.chunk_sparse = encoded_output['lexical_weights']
        self.chunk_colbert = encoded_output['colbert_vecs']

    def search(self, query, top_k=3, weights=[0.4, 0.2, 0.4]):
        # Query encoding
        query_output = self.model.encode(
            [query], 
            return_dense=True, 
            return_sparse=True, 
            return_colbert_vecs=True
        )
        q_dense = query_output['dense_vecs'][0]
        q_sparse = query_output['lexical_weights'][0]
        q_colbert = query_output['colbert_vecs'][0]

        results = []
        for i in range(len(self.chunks)):
            # 1. Dense Score (Cosine Similarity)
            d_score = np.dot(q_dense, self.chunk_dense[i])
            
            # 2. Sparse Score (Lexical Matching)
            s_score = self.model.compute_lexical_matching_score(q_sparse, self.chunk_sparse[i])
            
            # 3. ColBERT Score (Late Interaction)
            c_score = self.model.colbert_score(q_colbert, self.chunk_colbert[i])
            
            # Weighted Hybrid Score
            h_score = weights[0] * d_score + weights[1] * s_score + weights[2] * c_score
            
            results.append({
                'idx': i,
                'dense': d_score,
                'sparse': s_score,
                'colbert': c_score,
                'hybrid': h_score,
                'text': self.chunks[i][:150].replace('\n', ' ')
            })

        # Sort by hybrid score
        hybrid_sorted = sorted(results, key=lambda x: x['hybrid'], reverse=True)[:top_k]
        return hybrid_sorted

# 3. 실행 및 심화 분석
pdf_path = "tests/data/2201.07520v1.pdf"
chunks = load_pdf_and_chunk(pdf_path)
engine = BGEM3AdvancedEngine(chunks)

test_queries = [
    "Technical details of PPO reinforcement learning",
    "How does the model handle safety and bias?",
    "Summary of Table 1 results"
]

print("\n" + "="*80)
print(f"{'Query':<40} | {'Mode':<10} | {'Top Result Snippet'}")
print("-" * 80)

for query in test_queries:
    start_time = time.time()
    top_results = engine.search(query)
    elapsed = time.time() - start_time
    
    print(f"\nQUERY: {query} (Took: {elapsed:.2f}s)")
    for r in top_results:
        print(f"  [Score: {r['hybrid']:.4f}] (D:{r['dense']:.2f}, S:{r['sparse']:.2f}, C:{r['colbert']:.2f})")
        print(f"  Snippet: {r['text']}...")
print("="*80)
