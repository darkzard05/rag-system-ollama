
import asyncio
import sys
import os
import fitz
import time
import re

# 프로젝트 루트 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.model_loader import ModelManager
from core.rag_core import preprocess_text
from langchain_core.documents import Document

def test_chunking_optimization():
    print("=== 의미론적 청킹 최적화 테스트 ===")
    
    pdf_path = "tests/data/2201.07520v1.pdf"
    if not os.path.exists(pdf_path):
        print(f"오류: 테스트 파일이 없습니다. {pdf_path}")
        return

    # 1. PDF 텍스트 추출
    print(f"파일 로딩: {pdf_path}")
    doc = fitz.open(pdf_path)
    text_content = ""
    for i in range(min(5, len(doc))):
        page_text = doc[i].get_text()
        if page_text:
            text_content += preprocess_text(page_text) + "\n"
    doc.close()
    
    if not text_content.strip():
        print("경고: 추출된 텍스트가 없습니다.")
        return

    # LangChain 문서 객체 생성
    input_docs = [Document(page_content=text_content)]
    
    # 2. 임베딩 모델 로드
    embedder = ModelManager.get_embedder()
    
    # 3. 청커 초기화
    chunker = EmbeddingBasedSemanticChunker(
        embedder=embedder,
        min_chunk_size=100,
        max_chunk_size=800,
        similarity_threshold=0.5
    )
    
    # 4. 비교 검증
    print("\n[병합 로직 검증]")
    
    # A. 병합 전 (Pure Split)
    # 정규식에 캡처 그룹()이 있으면 구분자도 리스트에 포함되므로 제거 필요
    split_pattern = re.compile(chunker.sentence_split_regex)
    raw_segments = [s for s in split_pattern.split(text_content) if s.strip()]
    raw_count = len(raw_segments)
    
    # B. 병합 후 (Optimized)
    optimized_sentences = chunker._split_sentences(text_content)
    optimized_count = len(optimized_sentences)
    
    # C. 감소율 계산
    reduction_rate = (raw_count - optimized_count) / raw_count * 100
    
    print(f"1. 병합 전 문장 수: {raw_count}개")
    print(f"2. 병합 후 문장 수: {optimized_count}개")
    print(f"-> 임베딩 호출 감소율: {reduction_rate:.1f}% ({raw_count - optimized_count}회 절약)")
    
    # 30자 미만 잔존율 확인
    short_sentences = [s for s in optimized_sentences if len(s["text"]) < 30]
    
    print(f"3. 30자 미만 잔존 수: {len(short_sentences)}개 (전체의 {len(short_sentences)/optimized_count*100:.1f}%)")
    
    # 5. 실제 처리
    print("\n[분할 및 임베딩 수행 중...]")
    start_t = time.time()
    chunks, _ = chunker.split_documents(input_docs)
    end_t = time.time()
    
    print(f"최종 청크 수: {len(chunks)}")
    print(f"처리 소요 시간: {end_t - start_t:.4f}초")

if __name__ == "__main__":
    try:
        test_chunking_optimization()
    except Exception as e:
        import traceback
        traceback.print_exc()
