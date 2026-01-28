import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.model_loader import load_llm, load_embedding_model
from core.rag_core import build_rag_pipeline
from core.graph_builder import build_graph
from core.session import SessionManager
from common.config import DEFAULT_OLLAMA_MODEL, AVAILABLE_EMBEDDING_MODELS
from common.utils import apply_tooltips_to_response

async def test_full_flow():
    print("🚀 [최종 검증] RAG 풀 파이프라인 (LangGraph) 테스트 시작")
    pdf_path = "tests/2201.07520v1.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ 테스트용 PDF 파일을 찾을 수 없습니다: {pdf_path}")
        return

    # 1. 세션 초기화 및 모델 로드
    SessionManager.init_session()
    embedding_model_name = AVAILABLE_EMBEDDING_MODELS[0]
    embedder = load_embedding_model(embedding_model_name)
    llm = load_llm(DEFAULT_OLLAMA_MODEL)
    
    SessionManager.set("llm", llm)
    SessionManager.set("embedder", embedder)
    
    # 2. RAG 파이프라인 빌드
    print(f"⚙️ 문서 인덱싱 및 RAG 빌드 중: {pdf_path}")
    success_msg, cache_used = build_rag_pipeline(
        uploaded_file_name="2201.07520v1.pdf",
        file_path=pdf_path,
        embedder=embedder
    )
    print(f"✅ RAG 빌드 완료: {success_msg} (캐시 사용: {cache_used})")
    
    # 3. 그래프 생성
    app = build_graph()
    
    # 4. 질문 및 답변 생성 (스트리밍 시뮬레이션)
    question = "What are CM3-Medium and CM3-Large models?"
    print(f"🤔 질문: {question}")
    
    full_response = ""
    retrieved_docs = []
    
    print("📡 답변 스트리밍 중...")
    async for event in app.astream_events(
        {"input": question},
        config={"configurable": {"llm": llm}},
        version="v2"
    ):
        kind = event["event"]
        name = event.get("name", "Unknown")
        data = event.get("data", {})
        
        if kind == "on_custom_event" and name == "response_chunk":
            chunk = data.get("chunk", "")
            full_response += chunk
            if chunk: print(chunk, end="", flush=True)
            
        elif kind == "on_chain_end" and name == "retrieve":
            retrieved_docs = data.get("output", {}).get("documents", [])
            print(f"\n🔍 문서 {len(retrieved_docs)}개 검색 완료")

    print("\n" + "="*50)
    print("📋 [최종 답변 내용]")
    print(full_response)
    
    # 5. 포맷팅 검증 (툴팁 적용)
    final_content = apply_tooltips_to_response(full_response, retrieved_docs)
    print("\n📋 [툴팁 적용 결과 (일부)]")
    print(final_content[:500] + "...")
    print("="*50)
    
    if len(full_response) > 50:
        print("\n🎉 테스트 성공: RAG 파이프라인이 정상 작동합니다.")
    else:
        print("\nFAIL: 답변이 너무 짧거나 비어있습니다.")

if __name__ == "__main__":
    asyncio.run(test_full_flow())