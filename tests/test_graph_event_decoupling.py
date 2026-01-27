import asyncio
import pytest
from typing import Any, AsyncIterator, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from unittest.mock import MagicMock, AsyncMock
from src.core.graph_builder import build_graph

class FakeLLM(BaseChatModel):
    """테스트용 가짜 LLM"""
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        gen = ChatGeneration(message=AIMessage(content="확장된 질문 1\n확장된 질문 2"))
        return ChatResult(generations=[gen])

    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        gen = ChatGeneration(message=AIMessage(content="확장된 질문 1\n확장된 질문 2"))
        return ChatResult(generations=[gen])

    async def _astream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="이것은 "))
        yield ChatGenerationChunk(message=AIMessageChunk(content="테스트 "))
        yield ChatGenerationChunk(message=AIMessageChunk(content="답변입니다."))

    @property
    def _llm_type(self) -> str:
        return "fake"

@pytest.mark.asyncio
async def test_graph_emits_custom_status_events():
    # 1. Mock Retriever
    mock_retriever = MagicMock()
    mock_retriever.ainvoke = AsyncMock(return_value=[])
    
    # 2. Fake LLM
    fake_llm = FakeLLM()
    
    # 3. 그래프 빌드
    app = build_graph(retriever=mock_retriever)
    
    # 4. 실행 설정
    config = {"configurable": {"llm": fake_llm}}
    received_events = []
    
    print("\n[Test] 그래프 실행 시작...")
    
    try:
        async for event in app.astream_events(
            {"input": "테스트 질문입니다."}, 
            config=config, 
            version="v2"
        ):
            kind = event["event"]
            name = event.get("name")
            received_events.append(f"{kind}:{name}")
            
            if kind == "on_custom_event" and name == "status_update":
                print(f"  >> [Status Update] {event['data'].get('message')}")
            elif kind == "on_chain_start":
                print(f"  >> [Start] {name}")
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {e}")
        raise e

    # 5. 검증
    status_updates = [e for e in received_events if "on_custom_event" in e]
    print(f"\n[Test] 총 이벤트 개수: {len(received_events)}")
    print(f"[Test] 상태 업데이트 개수: {len(status_updates)}")
    
    assert len(status_updates) > 0, "커스텀 이벤트가 수집되지 않았습니다."
    print("✅ 이벤트 디커플링 검증 완료!")

if __name__ == "__main__":
    pytest.main([__file__, "-s"])
