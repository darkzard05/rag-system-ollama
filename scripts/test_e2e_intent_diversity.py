import os
import sys
import asyncio
import time
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from core.rag_core import RAGSystem
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL
from core.session import SessionManager

async def run_e2e_diversity_test():
    print("\n======================================================================")
    print("🎯 RAG Pipeline E2E Intent Diversity & Quality Test")
    print("======================================================================")
    
    session_id = f"e2e-test-{int(time.time())}"
    rag = RAGSystem(session_id=session_id)
    
    # 1. 모델 및 문서 준비
    print("\n[STEP 1] Initializing Resources...")
    embedder = ModelManager.get_embedder()
    llm = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
    
    # 인덱싱 (최적화된 캐시 활용)
    await rag.load_document(test_pdf, "CM3_Paper.pdf", embedder)
    print("✅ System Ready with CM3 Paper.")

    # 2. 테스트 쿼리 정의
    test_cases = [
        {
            "type": "GREETING",
            "query": "안녕? 넌 어떤 도움을 줄 수 있는 인공지능이니?",
            "eval_point": "검색 노드를 타지 않고 즉시 친절하게 응답하는가?"
        },
        {
            "type": "FACTOID",
            "query": "CM3 훈련에 사용된 데이터셋의 규모가 어느 정도야?",
            "eval_point": "구체적인 수치(2.7B, 13B 등)와 페이지 번호를 정확히 추출하는가?"
        },
        {
            "type": "RESEARCH",
            "query": "CM3의 'Causally Masked' 방식이 기존의 'Masked Language Modeling'과 어떻게 다른지 기술적으로 비교해줘.",
            "eval_point": "두 방식의 차이점을 논리적으로 대조하고 깊이 있게 분석하는가?"
        },
        {
            "type": "SUMMARY",
            "query": "이 문서 전체를 바탕으로 핵심 내용을 질문과 답변(Q&A) 형식 3가지로 재구성해줘.",
            "eval_point": "문서 전체의 맥락을 파악하고 요청한 'Q&A 형식'을 완벽히 준수하는가?"
        }
    ]

    # 3. 순차적 테스트 실행
    for i, case in enumerate(test_cases, 1):
        print(f"\n[Test Case {i}] Type: {case['type']}")
        print(f"💬 Query: {case['query']}")
        
        start_time = time.time()
        # aquery 호출
        result = await rag.aquery(case['query'], llm=llm)
        latency = time.time() - start_time
        
        # 결과 분석
        intent = result.get("route_decision", "N/A")
        docs_count = len(result.get("documents", []))
        answer = result.get("response", "")
        
        print(f"📡 Detected Intent: {intent}")
        print(f"📚 Context Docs: {docs_count} blocks")
        print(f"⏱️ Latency: {latency:.2f}s")
        print(f"🎯 Evaluation Goal: {case['eval_point']}")
        print("\n[Preview Response]")
        print("----------------------------------------")
        print(answer[:400] + ("..." if len(answer) > 400 else ""))
        print("----------------------------------------")
        
        # 검증 로직 (인텐트 일치 확인)
        if intent == case['type']:
            print(f"✅ Intent Classification: MATCH")
        else:
            print(f"⚠️ Intent Classification: MISMATCH (Expected {case['type']}, Got {intent})")

    print("\n======================================================================")
    print("✅ E2E Diversity Test Completed")
    print("======================================================================")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_e2e_diversity_test())