import asyncio
from langchain_ollama import ChatOllama
from core.graph_builder import GradeResponse, RewriteResponse

async def test_real_structured_output():
    # 1. 모델 초기화
    llm = ChatOllama(model="qwen3.5:4b", temperature=0)
    
    # 2. GradeResponse 테스트
    print("\n--- Testing GradeResponse ---")
    structured_grade_llm = llm.with_structured_output(GradeResponse)
    
    grade_prompt = """사용자의 질문에 대해 아래 문서가 답변을 제공하기 위한 실질적인 근거를 포함하고 있는지 판단하세요.
질문: DeepSeek-R1의 성능은 어때?
문서:
DOC 1: DeepSeek-R1은 오픈 소스 모델 중 가장 뛰어난 추론 성능을 보이며, 특히 수학과 코딩 분야에서 GPT-4o에 필적하는 결과를 보여줍니다."""
    
    try:
        grade_result = await structured_grade_llm.ainvoke(grade_prompt)
        print(f"Is Relevant: {grade_result.is_relevant}")
        print(f"Reason: {grade_result.reason}")
    except Exception as e:
        print(f"GradeResponse Error: {e}")

    # 3. RewriteResponse 테스트
    print("\n--- Testing RewriteResponse ---")
    structured_rewrite_llm = llm.with_structured_output(RewriteResponse)
    
    rewrite_prompt = """사용자의 질문을 분석하여 문서 저장소(RAG)에서 더 정확한 정보를 찾을 수 있도록 구체적인 검색어로 재구성하세요.
불필요한 수식어는 제거하고, 핵심 키워드와 맥락 위주로 작성하세요.
원본 질문: 딥식 R1이랑 GPT4 비교해줘"""
    
    try:
        rewrite_result = await structured_rewrite_llm.ainvoke(rewrite_prompt)
        print(f"Optimized Query: {rewrite_result.optimized_query}")
    except Exception as e:
        print(f"RewriteResponse Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_real_structured_output())
