import asyncio
import os
import sys
import io
from pathlib import Path

# Windows 인코딩 대응
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 프로젝트 루트를 path에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from common.config import QA_SYSTEM_PROMPT, OLLAMA_MODEL_NAME
from core.model_loader import load_llm
from core.graph_builder import build_graph
from langchain_core.documents import Document

async def test_new_prompt():
    print("=== QA 프롬프트 V2 테스트 시작 ===")
    
    # 1. 모델 로드
    try:
        llm = load_llm(OLLAMA_MODEL_NAME)
        print(f"모델: {llm.model}")
    except Exception as e:
        print(f"오류: {e}")
        return

    # 2. 컨텍스트 생성
    mock_docs = [
        Document(page_content="DeepSeek-R1은 671B 파라미터를 가진 모델입니다.", metadata={"page": 1, "source": "test.pdf"}),
        Document(page_content="이 모델은 오픈 소스이며 MIT 라이선스를 따릅니다.", metadata={"page": 2, "source": "test.pdf"})
    ]

    # 3. 그래프 실행
    class MockRetriever:
        async def ainvoke(self, query): return mock_docs
        def invoke(self, query): return mock_docs
        
    app = build_graph(retriever=MockRetriever())
    
    question = "DeepSeek-R1의 파라미터 수와 라이선스를 알려줘."
    print(f"질문: {question}")
    
    config = {"configurable": {"llm": llm}}
    inputs = {"input": question}
    
    result = await app.ainvoke(inputs, config=config)
    response = result["response"]
    
    # 결과를 파일로 저장 (검토용)
    with open("tests/test_result.txt", "w", encoding="utf-8") as f:
        f.write(response)
    
    print("\n" + "="*50)
    print("응답 내용:")
    print(response)
    print("="*50)

    # 4. 검증
    has_p1 = "[p.1]" in response
    has_p2 = "[p.2]" in response
    is_direct = not response.startswith(("질문에", "문서에", "제공된"))
    
    print(f"\n검증 결과:")
    print(f"- [p.1] 인용 포함: {'✅' if has_p1 else '❌'}")
    print(f"- [p.2] 인용 포함: {'✅' if has_p2 else '❌'}")
    print(f"- 직접 답변 (서두 없음): {'✅' if is_direct else '❌'}")

if __name__ == "__main__":
    asyncio.run(test_new_prompt())