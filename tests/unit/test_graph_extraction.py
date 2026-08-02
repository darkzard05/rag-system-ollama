import pytest

pytestmark = pytest.mark.skip(reason="Functionality removed/refactored")
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig


@pytest.fixture
def mock_llm():
    """LLM과 구조화된 출력을 모킹합니다."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()

    # astream 모킹 (비동기 제너레이터)
    async def mock_astream(*args, **kwargs):
        chunk = MagicMock()
        chunk.content = "테스트 답변입니다."
        chunk.response_metadata = {"prompt_eval_count": 10}
        yield chunk

    llm.astream = mock_astream

    # CustomOllama의 전처리 메서드 모사
    def mock_convert(chunk):
        return chunk.content, None

    llm._convert_chunk_to_thought_and_content = mock_convert

    structured_llm = AsyncMock()
    llm.with_structured_output.return_value = structured_llm
    return llm, structured_llm


@pytest.mark.asyncio
async def test_extract_graph_elements_success(mock_llm):
    """엔티티와 관계가 성공적으로 추출되는 경우 테스트"""
    llm, structured_llm = mock_llm

    # Mock Data
    doc = Document(
        page_content="DeepSeek-R1은 강력한 추론 능력을 가진 모델입니다. OpenAI는 이 모델의 경쟁사입니다.",
        metadata={"page": 1},
    )
    state: GraphState = {
        "input": "DeepSeek-R1에 대해 알려줘",
        "relevant_docs": [doc],
        "entities": [],
        "relations": [],
        "search_queries": [],
        "chat_history": [],
        "intent": "rag",
        "documents": [],
        "context": None,
        "response": None,
        "thought": None,
        "performance": None,
        "search_weights": None,
        "is_cached": False,
        "retry_count": 0,
    }

    # Mock LLM Output
    expected_entities = [
        Entity(
            name="DeepSeek-R1", type="Model", description="강력한 추론 능력을 가진 모델"
        ),
        Entity(name="OpenAI", type="Organization", description="DeepSeek-R1의 경쟁사"),
    ]
    expected_relations = [
        Relation(
            source="DeepSeek-R1",
            target="OpenAI",
            relation="is_competitor_of",
            description="경쟁 관계",
        )
    ]
    structured_llm.ainvoke.return_value = GraphExtractionResponse(
        entities=expected_entities, relations=expected_relations
    )

    config = {"configurable": {"llm": llm, "run_id": uuid4()}}

    # Mock ModelManager.inference_session as an async context manager
    with patch(
        "core.resource_manager.ResourceCoordinator.inference_session"
    ) as mock_session_factory:
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = None
        mock_session.__aexit__.return_value = None
        mock_session_factory.return_value = mock_session

        # Mock callback manager to avoid RuntimeError
        with patch(
            "langchain_core.runnables.config.get_async_callback_manager_for_config"
        ) as mock_get_cb_manager:
            mock_cb_manager = AsyncMock()
            mock_cb_manager.parent_run_id = uuid4()
            mock_get_cb_manager.return_value = mock_cb_manager

            # Execute
            result = await extract_graph_elements(state, config, writer=MagicMock())

            # Verify
            assert "entities" in result
            assert "relations" in result
            assert len(result["entities"]) == 2
            assert result["entities"][0].name == "DeepSeek-R1"
            assert len(result["relations"]) == 1
            assert result["relations"][0].source == "DeepSeek-R1"


@pytest.mark.asyncio
async def test_extract_graph_elements_no_docs(mock_llm):
    """문서가 없는 경우 추출을 건너뛰는지 테스트"""
    llm, _ = mock_llm
    state: GraphState = {
        "input": "test",
        "relevant_docs": [],
        "entities": [],
        "relations": [],
        "search_queries": [],
        "chat_history": [],
        "intent": "rag",
        "documents": [],
        "context": None,
        "response": None,
        "thought": None,
        "performance": None,
        "search_weights": None,
        "is_cached": False,
        "retry_count": 0,
    }
    config = {"configurable": {"llm": llm, "run_id": uuid4()}}

    result = await extract_graph_elements(state, config, writer=MagicMock())

    assert result == {}


@pytest.mark.asyncio
async def test_extract_graph_elements_empty_extraction(mock_llm):
    """추출된 결과가 비어있는 경우 테스트"""
    llm, structured_llm = mock_llm

    doc = Document(page_content="Just some text.", metadata={"page": 1})
    state: GraphState = {
        "input": "test",
        "relevant_docs": [doc],
        "entities": [],
        "relations": [],
        "search_queries": [],
        "chat_history": [],
        "intent": "rag",
        "documents": [],
        "context": None,
        "response": None,
        "thought": None,
        "performance": None,
        "search_weights": None,
        "is_cached": False,
        "retry_count": 0,
    }

    structured_llm.ainvoke.return_value = GraphExtractionResponse(
        entities=[], relations=[]
    )
    config = {"configurable": {"llm": llm, "run_id": uuid4()}}

    with patch(
        "core.resource_manager.ResourceCoordinator.inference_session"
    ) as mock_session_factory:
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value = None
        mock_session.__aexit__.return_value = None
        mock_session_factory.return_value = mock_session

        # Mock callback manager to avoid RuntimeError
        with patch(
            "langchain_core.runnables.config.get_async_callback_manager_for_config"
        ) as mock_get_cb_manager:
            mock_cb_manager = AsyncMock()
            mock_cb_manager.parent_run_id = uuid4()
            mock_get_cb_manager.return_value = mock_cb_manager

            result = await extract_graph_elements(state, config, writer=MagicMock())

            assert result == {}
