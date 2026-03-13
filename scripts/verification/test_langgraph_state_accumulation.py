import asyncio
import os
import sys

# PYTHONPATH 보정
sys.path.append(os.path.abspath("src"))

from unittest.mock import AsyncMock, MagicMock
from langgraph.graph import StateGraph, START, END
from core.graph_builder import preprocess, rewrite_query, GradeResponse, RewriteResponse
from api.schemas import GraphState

async def test_langgraph_state_accumulation():
    # 1. 그래프 구성 (테스트용 간소화 버전)
    workflow = StateGraph(GraphState)
    
    # 노드 추가
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("rewrite", rewrite_query)
    
    # 엣지 추가
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "rewrite")
    workflow.add_edge("rewrite", END)
    
    graph = workflow.compile()
    
    # 2. Mock LLM 설정
    llm = MagicMock()
    structured_llm = AsyncMock()
    structured_llm.ainvoke.return_value = RewriteResponse(optimized_query="optimized result")
    llm.with_structured_output.return_value = structured_llm
    
    config = {"configurable": {"llm": llm}}
    
    # 3. 그래프 실행
    print("\n--- Running LangGraph State Accumulation Test ---")
    initial_state = {
        "input": "test query",
        "chat_history": [],
        "search_queries": ["initial query"],
        "retry_count": 0
    }
    
    final_state = await graph.ainvoke(initial_state, config=config)
    
    # 4. 검증
    print(f"Final Search Queries: {final_state.get('search_queries')}")
    print(f"Final Retry Count: {final_state.get('retry_count')}")
    
    # Reducer 동작 확인: 
    # initial(1) + rewrite_update(1) = 2개여야 함
    assert len(final_state.get("search_queries")) == 2
    assert "optimized result" in final_state.get("search_queries")
    
    # retry_count: initial(0) + rewrite_update(1) = 1이어야 함
    assert final_state.get("retry_count") == 1
    
    print("\n[ACCUMULATION TEST SUCCESS]")

if __name__ == "__main__":
    asyncio.run(test_langgraph_state_accumulation())
