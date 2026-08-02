# LLM Grading Short-circuit 로직 검증을 위한 단위 테스트
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
from langchain_core.documents import Document
from core.graph_builder import grade_documents


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event", new_callable=AsyncMock)
async def test_grade_documents_short_circuit_high_score(mock_dispatch):
    """리랭킹 점수가 높을 때 LLM 호출 없이 generate로 전이되는지 확인합니다."""
    # 1. 고득점 문서 준비
    docs = [
        Document(page_content="신뢰도 높은 정보", metadata={"rerank_score": 0.95}),
        Document(page_content="보조 정보", metadata={"rerank_score": 0.4}),
    ]

    # 2. 상태(State) 설정
    state = {
        "relevant_docs": docs,
        "input": "테스트 질문",
        "intent": "rag",
        "retry_count": 0,
    }

    # 3. Config 및 Writer 모킹
    config = {"configurable": {"llm": MagicMock()}}
    writer = MagicMock()

    # 4. 함수 실행 (수정된 로직에 의해 LLM 호출 없이 generate 반환 예상)
    result = await grade_documents(state, config, writer=writer)

    # 5. 검증
    assert result["intent"] == "generate"
    # LLM의 어떠한 메서드(ainvoke, bind 등)도 호출되지 않았어야 함
    assert config["configurable"]["llm"].mock_calls == []


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event", new_callable=AsyncMock)
async def test_grade_documents_proceeds_to_llm_on_low_score(mock_dispatch):
    """점수가 낮을 때 기존처럼 LLM 평가를 수행하는지 확인합니다."""
    # 1. 저득점 문서 준비
    docs = [Document(page_content="모호한 정보", metadata={"rerank_score": 0.6})]

    state = {
        "relevant_docs": docs,
        "input": "테스트 질문",
        "intent": "rag",
        "retry_count": 0,
    }

    # 2. LLM 모킹 (bind + JSON 모드 지원 포함)
    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm

    # LLM 응답 시뮬레이션 (JSON 모드)
    json_llm.ainvoke.return_value = SimpleNamespace(
        content=json.dumps(
            {
                "action": "generate",
                "is_relevant": True,
                "relevant_entities": [],
                "reason": "테스트 근거",
                "optimized_query": None,
            }
        )
    )

    config = {"configurable": {"llm": mock_llm}}
    writer = MagicMock()

    # 3. 함수 실행
    result = await grade_documents(state, config, writer=writer)

    # 4. 검증
    assert result["intent"] == "generate"
    # 점수가 낮으므로 LLM 평가가 호출되어야 함
    json_llm.ainvoke.assert_awaited_once()
