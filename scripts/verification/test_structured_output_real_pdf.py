import asyncio
import logging
import os
import sys

# PYTHONPATH 설정 보정 (src 디렉토리 포함)
sys.path.append(os.path.abspath("src"))

from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from core.graph_builder import GradeResponse, RewriteResponse, grade_documents, rewrite_query
from core.document_processor import load_pdf_docs

# 로깅 설정 (내부 동작 확인용)
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("RAG_VERIFY")

async def test_real_pdf_structured_output():
    # 1. 모델 초기화 (사용자 요청 모델: qwen3:4b)
    # ollama list에서 확인된 정확한 이름: qwen3:4b-instruct-2507-q4_K_M
    model_name = "qwen3:4b-instruct-2507-q4_K_M"
    llm = ChatOllama(model=model_name, temperature=0)
    
    # 2. 실제 PDF 데이터 로드
    pdf_path = "tests/data/2201.07520v1.pdf"
    print(f"\n--- Loading PDF: {pdf_path} ---")
    try:
        docs = load_pdf_docs(pdf_path, "test_file.pdf")
        sample_doc = docs[2] if len(docs) > 2 else docs[0] # 본문 중간 페이지 선택
        print(f"Loaded {len(docs)} pages. Using page {sample_doc.metadata.get('page')} for test.")
    except Exception as e:
        print(f"PDF Loading Error: {e}")
        return

    # 3. GradeResponse 테스트 (실제 문서 내용 활용)
    print("\n--- Testing Structured Grade Node ---")
    state = {
        "input": "DeepSeek-R1의 성능에 대해 설명해줘", # PDF 내용과 무관한 질문 시도
        "relevant_docs": [sample_doc],
        "is_cached": False,
        "intent": "rag"
    }
    config = {"configurable": {"llm": llm}}
    writer = lambda x: print(f"  [STATUS] {x.get('status', '')}") if x.get('status') else None

    print(f"Question: {state['input']}")
    print(f"Document Preview: {sample_doc.page_content[:150]}...")
    
    grade_result = await grade_documents(state, config, writer)
    print(f"Resulting Intent: {grade_result.get('intent')}")

    # 4. RewriteResponse 테스트
    print("\n--- Testing Structured Rewrite Node ---")
    state_rewrite = {
        "input": "딥식 R1 성능 요약",
        "retry_count": 0,
        "search_queries": []
    }
    
    rewrite_result = await rewrite_query(state_rewrite, config, writer)
    print(f"Original Query: {state_rewrite['input']}")
    print(f"Optimized Query: {rewrite_result.get('search_queries', ['N/A'])[-1]}")
    print(f"Retry Count: {rewrite_result.get('retry_count')}")

if __name__ == "__main__":
    asyncio.run(test_real_pdf_structured_output())
