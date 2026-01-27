
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from langchain_core.documents import Document
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.model_loader import load_embedding_model
from langchain_community.vectorstores import FAISS

def verify_integrity():
    print("--- 임베딩 재사용 무결성 및 정확도 검증 ---")
    
    # 1. 실제 경량 모델 로드
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder = load_embedding_model(model_name)
    
    test_text = """
    인공지능 기술은 빠르게 발전하고 있습니다. 
    특히 거대언어모델(LLM)은 자연어 처리 분야에서 혁신을 일으키고 있습니다. 
    RAG 시스템은 외부 지식을 활용하여 모델의 답변 정확도를 높입니다. 
    이 테스트는 임베딩 재사용 로직의 무결성을 검증하기 위한 것입니다.
    """
    docs = [Document(page_content=test_text, metadata={"source": "integrity_test.pdf"})]
    
    # 2. 최적화된 방식으로 청크 및 벡터 추출
    chunker = EmbeddingBasedSemanticChunker(embedder=embedder, max_chunk_size=200)
    split_docs, reused_vectors = chunker.split_documents(docs)
    
    print(f"생성된 청크 수: {len(split_docs)}")
    print(f"추출된 벡터 수: {len(reused_vectors)}")
    
    # 검증 1: 개수 및 차원 확인
    assert len(split_docs) == len(reused_vectors), "청크와 벡터의 개수가 일치하지 않습니다!"
    assert reused_vectors[0].shape[0] == 384, f"벡터 차원이 올바르지 않습니다: {reused_vectors[0].shape[0]}"
    print("✅ 검증 1: 데이터 개수 및 차원 일치 확인 완료")

    # 3. 두 가지 인덱스 구축
    # 인덱스 A: 재사용된 벡터 사용
    text_embeddings = zip([d.page_content for d in split_docs], reused_vectors)
    vector_store_reused = FAISS.from_embeddings(text_embeddings, embedder, metadatas=[d.metadata for d in split_docs])
    
    # 인덱스 B: 청크 텍스트를 새로 임베딩 (전통적 방식)
    vector_store_fresh = FAISS.from_documents(split_docs, embedder)
    
    # 4. 검색 결과 비교
    query = "RAG 시스템의 장점은 무엇인가요?"
    results_reused = vector_store_reused.similarity_search_with_score(query, k=1)
    results_fresh = vector_store_fresh.similarity_search_with_score(query, k=1)
    
    doc_reused, score_reused = results_reused[0]
    doc_fresh, score_fresh = results_fresh[0]
    
    print(f"\n[질의]: {query}")
    print(f"[재사용 방식 결과]: {doc_reused.page_content[:50]}... (점수: {score_reused:.4f})")
    print(f"[새로 임베딩 결과]: {doc_fresh.page_content[:50]}... (점수: {score_fresh:.4f})")
    
    # 검증 2: 결과 일치 여부
    # 점수는 미세하게 다를 수 있지만(Averaging vs Full Embedding), 가장 관련 있는 문서는 동일해야 함
    if doc_reused.page_content == doc_fresh.page_content:
        print("✅ 검증 2: 검색 결과 일관성 확인 완료 (동일 문서 추출)")
    else:
        print("⚠️ 경고: 검색 결과가 다릅니다. 의미론적 차이를 확인하세요.")
        
    # 검증 3: 코사인 유사도 거리 확인 (두 방식의 벡터가 얼마나 유사한지)
    fresh_vector = embedder.embed_query(doc_reused.page_content)
    cos_sim = np.dot(reused_vectors[0], fresh_vector) / (np.linalg.norm(reused_vectors[0]) * np.linalg.norm(fresh_vector))
    print(f"✅ 검증 3: 두 방식 간 벡터 유사도: {cos_sim:.4f}")
    
    if cos_sim > 0.9:
        print("결론: 최적화된 임베딩 재사용 방식은 원본 방식과 90% 이상 유사하며 무결합니다.")
    else:
        print("결론: 임베딩 방식에 따른 차이가 큽니다. 성능 평가가 필요합니다.")

if __name__ == "__main__":
    verify_integrity()
