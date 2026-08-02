import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

# PYTHONPATH 보정
sys.path.append(os.path.abspath("src"))

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from api.schemas import GraphState
from core.graph_builder import grade_documents, preprocess, rewrite_query


async def test_langgraph_state_accumulation():
    # 1. 그래프 구성 (실제 워크플로우 순서: preprocess -> grade_documents -> rewrite_query)
    workflow = StateGraph(GraphState)

    # 노드 추가
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)

    # 엣지 추가
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "grade")
    workflow.add_edge("grade", "rewrite")
    workflow.add_edge("rewrite", END)

    graph = workflow.compile()

    # 2. 통합 응답(UnifiedGradeRewriteResponse JSON)을 반환하는 Mock LLM 설정
    #    새 API: with_structured_output 대신 llm.bind(response_format=json) + JSON 파싱
    llm = MagicMock()
    json_llm = AsyncMock()
    json_llm.ainvoke.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "action": "rewrite",
                "is_relevant": False,
                "relevant_entities": [],
                "reason": "검색된 문서가 질문에 대한 충분한 근거를 제공하지 못함",
                "optimized_query": "optimized result",
            },
            ensure_ascii=False,
        )
    )
    llm.bind.return_value = json_llm

    config = {"configurable": {"llm": llm}}

    # 3. 그래프 실행
    print("\n--- Running LangGraph State Accumulation Test ---")
    initial_state = {
        "input": "test query",
        "chat_history": [],
        "search_queries": ["initial query"],
        "retry_count": 0,
        "relevant_docs": [
            Document(
                page_content="테스트 문서 내용입니다.",
                metadata={"page": 1, "source": "test.pdf"},
            )
        ],
    }

    final_state = await graph.ainvoke(initial_state, config=config)

    # 4. 검증
    print(f"Final Search Queries: {final_state.get('search_queries')}")
    print(f"Final Retry Count: {final_state.get('retry_count')}")

    # Reducer 동작 확인 (search_queries: operator.add)
    # initial(1) + grade_documents가 추가한 재작성 쿼리(1) = 2개여야 함
    assert len(final_state.get("search_queries")) == 2
    assert "optimized result" in final_state.get("search_queries")

    # Reducer 동작 확인 (retry_count: operator.add)
    # initial(0) + grade_documents 갱신(1) = 1
    # (rewrite_query는 search_queries 존재 시 순수 passthrough — retry 예산 이중 소진 방지)
    assert final_state.get("retry_count") == 1

    print("\n[ACCUMULATION TEST SUCCESS]")


if __name__ == "__main__":
    asyncio.run(test_langgraph_state_accumulation())
