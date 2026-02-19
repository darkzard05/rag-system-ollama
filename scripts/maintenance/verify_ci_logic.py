import os
import asyncio
import sys

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.model_loader import ModelManager

async def verify_fake_llm():
    # CI 테스트 환경 변수 설정
    os.environ["IS_CI_TEST"] = "true"
    print(f"[VERIFY] IS_CI_TEST: {os.environ.get('IS_CI_TEST')}")

    try:
        # LLM 로드 (실제 모델명이 무엇이든 Fake가 나와야 함)
        llm = ModelManager.get_llm(model_name="any-model")
        print(f"[VERIFY] LLM Type: {type(llm)}")
        
        # 응답 테스트 (동기 호출)
        response = llm.invoke("Hello CI Test")
        print(f"""[VERIFY] Response: {response.content}""")
        
        if "RAG 시스템 테스트 응답" in response.content:
            print("""\n[SUCCESS] Fake LLM is working correctly for CI!""")
        else:
            print("""\n[FAILURE] Fake LLM returned unexpected content.""")
            
    except Exception as e:
        print(f"""\n[ERROR] Verification failed: {e}""")

if __name__ == "__main__":
    asyncio.run(verify_fake_llm())
