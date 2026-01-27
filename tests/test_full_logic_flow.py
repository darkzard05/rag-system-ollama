import os
import sys
import asyncio
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# src 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

# Streamlit mock (SessionManager가 st.session_state를 사용함)
import streamlit as st
if "session_state" not in st.__dict__:
    st.session_state = {}

from core.rag_core import build_rag_pipeline
from core.model_loader import load_embedding_model, load_llm
from core.session import SessionManager
from common.config import AVAILABLE_EMBEDDING_MODELS, OLLAMA_MODEL_NAME

@pytest.mark.asyncio
async def test_rag_system_full_flow():
    """
    LangGraph 파이프라인을 직접 사용하여 PDF 로드부터 답변 생성까지의 전체 로직을 테스트합니다.
    """
    print("\n🚀 RAG 통합 테스트 시작...")
    
    # 0. 세션 초기화
    SessionManager.init_session()
    
    # 1. 모델 로드 (임베딩 및 LLM)
    embedding_model = AVAILABLE_EMBEDDING_MODELS[0]
    llm_model = OLLAMA_MODEL_NAME
    
    print(f"🧬 모델 로딩 중: {embedding_model}, {llm_model}")
    embedder = await asyncio.to_thread(load_embedding_model, embedding_model)
    llm = await asyncio.to_thread(load_llm, llm_model)
    print("✅ 모델 로딩 완료")

    # 2. PDF 문서 로드 및 파이프라인 구축
    pdf_path = os.path.join("tests", "2201.07520v1.pdf")
    if not os.path.exists(pdf_path):
        # 파일이 없을 경우 빈 파일이라도 생성하거나 스킵 (여기서는 기존 파일 활용 가정)
        pytest.skip(f"테스트 파일이 없습니다: {pdf_path}")
        
    print(f"📂 문서 인덱싱 및 파이프라인 구축 시작: {pdf_path}")
    # SessionManager에 LLM 미리 설정 (build_rag_pipeline이 session을 사용함)
    SessionManager.set("llm", llm)
    
    msg, cache_used = await asyncio.to_thread(
        build_rag_pipeline,
        uploaded_file_name="2201.07520v1.pdf",
        file_path=pdf_path,
        embedder=embedder
    )
    print(f"✨ {msg} (캐시 사용: {cache_used})")
    
    rag_engine = SessionManager.get("rag_engine")
    assert rag_engine is not None, "RAG 엔진(Graph)이 생성되지 않았습니다."

    # 3. 질의응답 테스트
    question = "What is the main contribution of this paper?"
    print(f"💬 질문 입력: {question}")
    
    # LangGraph 실행
    # config에 llm을 전달해야 함 (graph_builder.py의 generate_response가 이를 기대함)
    config = {"configurable": {"llm": llm}}
    
    # 스트리밍 방식이 아닌 일반 비동기 실행 (ainvoke)
    response = await rag_engine.ainvoke(
        {"input": question},
        config=config
    )
    
    print("\n" + "="*50)
    print("🤖 AI 답변:")
    print(response.get("response", "답변 실패"))
    
    if response.get("thought"):
        print("\n💭 사고 과정:")
        print(response.get("thought")[:200] + "...")
    print("-" * 50)
    
    # 4. 검증
    ans_text = response.get("response", "")
    assert len(ans_text) > 10, "답변이 너무 짧습니다."
    assert "documents" in response, "참조 문서가 포함되지 않았습니다."
    
    print(f"🔍 참조 문서 수: {len(response['documents'])}")
    print("="*50)
    print("\n✨ 모든 통합 테스트 과정이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    asyncio.run(test_rag_system_full_flow())
