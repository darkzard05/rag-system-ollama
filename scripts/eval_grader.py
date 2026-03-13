
import asyncio
import logging
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from langchain_core.documents import Document
from core.graph_builder import grade_documents
from core.model_loader import ModelManager
from langchain_core.runnables import RunnableConfig

# 로깅 설정 (결과만 보기 위해 ERROR로 설정)
logging.basicConfig(level=logging.ERROR)

class MockWriter:
    """StreamWriter 대용"""
    def __call__(self, data):
        pass

async def evaluate_grader():
    print("="*60)
    print(" [Grade Documents Node] 정밀도 및 신뢰성 평가")
    print("="*60)

    # 1. LLM 준비
    from common.config import DEFAULT_OLLAMA_MODEL
    llm = await ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    config = {"configurable": {"llm": llm}}
    writer = MockWriter()

    # 2. 테스트용 문서 샘플 (CM3 논문 내용 발췌)
    sample_doc = Document(
        page_content="CM3 is a causally masked objective for multi-modal modeling. It enables unified representation of text and images by masking spans and regenerating them at the end of the string.",
        metadata={"page": 1}
    )

    test_cases = [
        {
            "name": "Positive (직접적 관련)",
            "query": "CM3 모델의 원인 마스킹(causally masked) 목적이 무엇인가요?",
            "expected": "YES",
            "docs": [sample_doc]
        },
        {
            "name": "Negative (완전 무관)",
            "query": "맛있는 김치찌개를 끓이는 법을 알려줘.",
            "expected": "NO",
            "docs": [sample_doc]
        },
        {
            "name": "Ambiguous (주제는 비슷하나 근거 없음)",
            "query": "이 논문을 쓴 저자의 이메일 주소와 전화번호는 무엇인가요?",
            "expected": "NO",
            "docs": [sample_doc]
        },
        {
            "name": "Empty (문서 없음)",
            "query": "아무 질문이나 합니다.",
            "expected": "transform", # 노드에서 문서가 없으면 intent를 transform으로 반환함
            "docs": []
        }
    ]

    success_count = 0
    for case in test_cases:
        state = {
            "input": case["query"],
            "relevant_docs": case["docs"],
            "intent": "rag",
            "is_cached": False
        }
        
        print(f"\n[Test: {case['name']}]")
        print(f" - 질문: {case['query']}")
        
        # 노드 실행
        result = await grade_documents(state, config, writer)
        
        actual = result.get("intent", "N/A")
        # 예상 결과 매칭 확인
        is_success = False
        if case["expected"] == "YES" and actual == "generate":
            is_success = True
        elif case["expected"] == "NO" and actual == "transform":
            is_success = True
        elif case["expected"] == "transform" and actual == "transform":
            is_success = True
            
        status = "✅ PASS" if is_success else "❌ FAIL"
        if is_success: success_count += 1
        
        print(f" - 예상 결과: {case['expected']} (Intent mapping)")
        print(f" - 실제 결과: {actual}")
        print(f" - 판정: {status}")

    print("\n" + "="*60)
    print(f" 최종 결과: {success_count}/{len(test_cases)} 통과 ({success_count/len(test_cases)*100:.1f}%)")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(evaluate_grader())
