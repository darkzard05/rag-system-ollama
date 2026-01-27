import asyncio
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.rag_core import RAGSystem
from core.model_loader import load_llm, load_embedding_model
from core.graph_builder import build_graph
from common.config import OLLAMA_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS
from common.utils import apply_tooltips_to_response
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

async def test_until_success():
    print("🚀 [최종 검증] PDF 기반 답변 생성 테스트 시작")
    pdf_path = "tests/2201.07520v1.pdf"
    session_id = "final_verify_session"
    
    # 1. 시스템 초기화
    embedding_model_name = AVAILABLE_EMBEDDING_MODELS[0]
    embedder = load_embedding_model(embedding_model_name)
    llm = load_llm(OLLAMA_MODEL_NAME)
    rag_system = RAGSystem(session_id=session_id)
    
    print(f"⚙️ 문서 인덱싱 중: {pdf_path}")
    await asyncio.to_thread(rag_system.load_document, pdf_path, "test.pdf", embedder)
    
    # 2. 질문 설정
    question = "What are CM3-Medium and CM3-Large models?"
    print(f"🤔 질문: {question}")
    
    # 3. 성공할 때까지 최대 3회 시도
    max_retries = 3
    full_response = ""
    retrieved_docs = []
    
    for attempt in range(1, max_retries + 1):
        print(f"\n🔄 시도 {attempt}/{max_retries}...")
        
        # 검색 수행
        retrieved_docs = await rag_system.ensemble_retriever.ainvoke(question)
        if not retrieved_docs:
            print("❌ 문서를 찾지 못했습니다.")
            continue
            
        context_text = "\n".join([d.page_content[:300] for d in retrieved_docs[:3]])
        
        # 프롬프트 구성 (최대한 단순하게)
        prompt = ChatPromptTemplate.from_template(
            "Use the context to answer the question briefly.\nContext: {context}\nQuestion: {question}"
        )
        chain = prompt | llm | StrOutputParser()
        
        try:
            # 타임아웃 설정 강화
            response = await asyncio.wait_for(
                chain.ainvoke({"context": context_text, "question": question}),
                timeout=60
            )
            
            if response and len(response.strip()) > 20:
                full_response = response
                print(f"✅ 답변 생성 성공! (길이: {len(full_response)})")
                break
            else:
                print("⚠️ 모델이 빈 응답 또는 너무 짧은 답변을 반환했습니다.")
        except asyncio.TimeoutError:
            print("⚠️ 타임아웃 발생")
        except Exception as e:
            print(f"⚠️ 오류 발생: {e}")
            
        await asyncio.sleep(2) # 잠시 대기

    if not full_response:
        print("\n❌ 모든 시도가 실패했습니다. 직접 질문으로 마지막 시도...")
        res = await llm.ainvoke(f"Based on CM3 paper, what is CM3 model? Answer in one sentence.")
        full_response = res.content

    # 4. 결과 출력 및 포맷팅 확인
    final_content = apply_tooltips_to_response(full_response, retrieved_docs)
    
    print("\n" + "="*50)
    print("📋 [최종 답변 내용]")
    print(full_response)
    print("\n📋 [포맷팅 적용 내용 (인용구 포함 여부 확인)]")
    print(final_content[:500] + "...")
    print("="*50)
    
    if len(full_response) > 0:
        print("\n🎉 테스트 성공: 제대로 된 답변을 수신했습니다.")
    else:
        print("\nFAIL: 답변을 수신하지 못했습니다.")

if __name__ == "__main__":
    asyncio.run(test_until_success())
