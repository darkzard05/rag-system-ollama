import sys
import os
from pathlib import Path

# 프로젝트 루트 경로 추가
ROOT_DIR = Path(os.getcwd())
sys.path.append(str(ROOT_DIR / "src"))

import asyncio
from core.model_loader import ModelManager
from common.config import DEFAULT_OLLAMA_MODEL

async def test_llm_modernization():
    print(f"[*] Testing LLM Modernization with model: {DEFAULT_OLLAMA_MODEL}")
    
    # 1. ModelManager를 통한 LLM 획득 (keep_alive 확인)
    llm = ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
    print(f"[1] LLM Instance: {type(llm)}")
    print(f"    - keep_alive: {getattr(llm, 'keep_alive', 'N/A')}")
    
    # 2. Thinking 추출 테스트
    print("\n[2] Testing Thinking/Content Extraction...")
    messages = [("user", "Hello! Who are you?")]
    
    # 동기 호출 테스트
    print("    - Running sync invoke...")
    try:
        response = llm.invoke(messages)
        print(f"    - Response type: {type(response)}")
        print(f"    - Content: {response.content[:50]}...")
        if "thinking" in response.additional_kwargs:
            print(f"    - Thinking found! (len: {len(response.additional_kwargs['thinking'])})")
        else:
            print("    - Thinking not found (Expected if model doesn't support thinking)")
    except Exception as e:
        print(f"    - Invoke failed: {e}")

    # 비동기 스트리밍 테스트
    print("\n[3] Testing Async Streaming...")
    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if "thinking" in chunk.additional_kwargs:
                print(f"\n[Thought Chunk]: {chunk.additional_kwargs['thinking'][:30]}...")
        print("\n    - Async Streaming Done.")
    except Exception as e:
        print(f"\n    - Async streaming failed: {e}")

    # 4. with_structured_output 테스트
    print("\n[4] Testing with_structured_output(include_raw=True)...")
    from pydantic import BaseModel, Field
    
    class UserProfile(BaseModel):
        name: str = Field(..., description="The name of the user")
        age: int = Field(..., description="The age of the user")
        hobby: str = Field(..., description="The user's hobby")

    try:
        # qwen3가 tool calling/structured output을 잘 지원한다고 가정
        structured_llm = llm.with_structured_output(UserProfile, include_raw=True)
        print(f"    - Structured LLM type: {type(structured_llm)}")
        
        prompt = "Create a user profile for a 25-year-old developer named Alice who likes hiking."
        response = structured_llm.invoke(prompt)
        
        print(f"    - Raw output keys: {response.keys()}")
        print(f"    - Parsed object: {response['parsed']}")
        
        if response['raw'].additional_kwargs:
            print(f"    - Additional kwargs in raw: {response['raw'].additional_kwargs}")
    except Exception as e:
        print(f"    - Structured output test failed (model might not support it): {e}")

if __name__ == "__main__":
    asyncio.run(test_llm_modernization())
