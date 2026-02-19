import sys
import os

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import bm25_tokenizer

def test_tokenizer_improvement():
    print("=== BM25 토크나이저 개선 검증 ===")
    
    test_cases = [
        "자연어처리는 재미있다",  # 복합명사 + 형용사
        "데이터베이스에서 검색한다", # 명사 + 조사
        "RAG시스템 구축",        # 영어 + 한글 혼용
        "학교에 간다"            # 명사 + 조사
    ]
    
    print("-" * 80)
    
    for text in test_cases:
        tokens = bm25_tokenizer(text)
        print(f"입력: {text}")
        print(f"토큰: {tokens}")
        print("-" * 20)

    # 검색 시나리오 시뮬레이션
    print("\n[검색 재현율(Recall) 시뮬레이션]")
    doc_text = "우리는 고성능 데이터베이스 시스템을 구축했습니다."
    doc_tokens = set(bm25_tokenizer(doc_text))
    
    queries = ["데이터", "베이스", "시스템", "구축"]
    
    print(f"문서 원문: {doc_text}")
    print(f"문서 토큰: {doc_tokens}")
    print("-" * 50)
    
    for q in queries:
        # 쿼리도 동일하게 토크나이징됨
        q_tokens = set(bm25_tokenizer(q))
        # 교집합이 있으면 매칭된 것
        match = doc_tokens.intersection(q_tokens)
        status = "✅ 매칭 성공" if match else "❌ 매칭 실패"
        print(f"검색어 '{q}': {status} (매칭: {match})")

if __name__ == "__main__":
    test_tokenizer_improvement()