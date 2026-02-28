import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from common.config import DEFAULT_OLLAMA_MODEL
from common.logging_config import setup_logging

async def quick_verify():
    setup_logging(log_level="INFO")
    
    print("""
[Quick Verify] Starting RAG Pipeline...""")
    
    rag = RAGSystem(session_id="verify-session")
    
    print("1. Loading Models...")
    embedder = load_embedding_model()
    llm = load_llm(DEFAULT_OLLAMA_MODEL)

    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    print(f"2. Indexing Document: {os.path.basename(test_pdf)}")
    
    # 캐시 사용 설정 (빠른 재시험을 위해)
    msg, cache_used = await rag.load_document(test_pdf, "2201.07520v1.pdf", embedder)
    print(f"   Result: {msg} (Cache: {cache_used})")

    test_query = "CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야? 기존 DALL-E와는 어떤 차이가 있어?"
    
    print(f"""
3. Querying: '{test_query}'""")
    # [리팩토링 반영] 모델 이름만 전달
    result = await rag.aquery(test_query, model_name=DEFAULT_OLLAMA_MODEL)
    
    print("""
""" + "="*50)
    print("RESPONSE:")
    print(result.get("response", "No response found."))
    print("="*50)
    print(f"""
Source Nodes: {len(result.get('source_nodes', []))}""")

if __name__ == "__main__":
    asyncio.run(quick_verify())
