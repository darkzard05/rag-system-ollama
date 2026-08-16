import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

from common.config import GRADING_CONFIG
from core.graph_builder import grade_documents


class MockWriter:
    """StreamWriter 대용으로 grade_documents의 writer 인자에 전달합니다."""

    def __call__(self, data: object) -> None:
        pass


def json_response(**fields):
    """LLM JSON 모드 응답을 모사하는 SimpleNamespace 객체를 생성합니다."""
    return SimpleNamespace(content=json.dumps(fields))


@pytest.fixture
def mock_state():
    return {
        "input": "DeepSeek-R1의 성능은 어떠한가요?",
        "relevant_docs": [
            Document(
                page_content="DeepSeek-R1은 매우 높은 성능을 가진 모델입니다.",
                metadata={"rerank_score": 0.5},
            ),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }


@pytest.fixture
def mock_config():
    """bind+JSON 모드로 모킹된 LLM을 포함한 config를 생성합니다."""
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm
    return {"configurable": {"llm": llm}}


@pytest.mark.asyncio
async def test_grade_documents_short_circuit():
    """리랭킹 점수가 매우 높을 때 Short-circuit이 작동하여 즉시 generate를 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 최고", metadata={"rerank_score": 0.9}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    llm = MagicMock()
    config = {"configurable": {"llm": llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {"intent": "generate", "route": "generate"}
    # Short-circuit에서는 LLM을 호출하지 않아야 합니다.
    llm.ainvoke.assert_not_called()
    llm.bind.assert_not_called()


@pytest.mark.asyncio
async def test_grade_documents_rubric_pass():
    """Unified 응답(action=generate)이면 LLM 검증을 통과하여 generate를 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 보통", metadata={"rerank_score": 0.5}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="generate",
        is_relevant=True,
        relevant_entities=["DeepSeek-R1"],
        reason="충분한 정보 포함",
        optimized_query=None,
    )

    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {"intent": "generate", "route": "generate"}


@pytest.mark.asyncio
async def test_grade_documents_rubric_fail():
    """Unified 응답(action=rewrite)이면 재작성 쿼리와 함께 transform을 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 낮음", metadata={"rerank_score": 0.3}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="rewrite",
        is_relevant=False,
        relevant_entities=["DeepSeek-R1"],
        reason="정보는 있으나 부족함",
        optimized_query="DeepSeek-R1 성능 벤치마크",
    )

    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {
        "intent": "transform",
        "route": "transform",
        "search_queries": ["DeepSeek-R1 성능 벤치마크"],
        "retry_count": 1,
    }


@pytest.mark.asyncio
async def test_grade_documents_is_relevant_false():
    """is_relevant가 False일 때 점수와 상관없이 transform을 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="전혀 상관없는 내용", metadata={"rerank_score": 0.5}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="rewrite",
        is_relevant=False,
        relevant_entities=[],
        reason="관련 없음",
        optimized_query="DeepSeek-R1 리뷰",
    )

    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=None)
    assert result == {
        "intent": "transform",
        "route": "transform",
        "search_queries": ["DeepSeek-R1 리뷰"],
        "retry_count": 1,
    }


def test_min_score_to_skip_loaded_from_config():
    """config.yml prompts.grading.min_score_to_skip(=0.85)가 GRADING_CONFIG에 로드되는지 검증"""
    assert GRADING_CONFIG.get("min_score_to_skip") == 0.85


@pytest.mark.asyncio
async def test_grade_documents_short_circuit_above_threshold():
    """rerank_score가 min_score_to_skip 이상이면 LLM 검증 없이 즉시 generate를 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 최고", metadata={"rerank_score": 0.9}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }
    llm = MagicMock()
    config = {"configurable": {"llm": llm}}

    result = await grade_documents(state, config, writer=MockWriter())
    assert result == {"intent": "generate", "route": "generate"}
    # Short-circuit에서는 LLM을 호출하지 않아야 합니다.
    llm.ainvoke.assert_not_called()
    llm.bind.assert_not_called()


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event", new_callable=AsyncMock)
async def test_grade_documents_calls_llm_below_threshold(
    mock_adispatch: AsyncMock,
):
    """rerank_score가 min_score_to_skip 미만이면 LLM 검증 경로로 진행해 generate를 반환하는지 검증"""
    state = {
        "input": "DeepSeek-R1 성능",
        "relevant_docs": [
            Document(page_content="성능 보통", metadata={"rerank_score": 0.5}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = json_response(
        action="generate",
        is_relevant=True,
        relevant_entities=["DeepSeek-R1"],
        reason="충분한 정보 포함",
        optimized_query=None,
    )
    config = {"configurable": {"llm": mock_llm}}

    result = await grade_documents(state, config, writer=MockWriter())
    assert result == {"intent": "generate", "route": "generate"}
    # 임계값 미만이므로 LLM 검증 경로가 실제로 실행되어야 합니다.
    mock_llm.bind.assert_called_once()
    json_llm.ainvoke.assert_awaited_once()


def test_structured_output_prompt_drops_inline_citation_mandate():
    """structured_output 프롬프트가 인라인 `[doc:N]` 강제 삽입 명령을 포함하지 않는지,
    그리고 citations[] 배열이 스키마에 남아 있는지 검증 (P1: MED)."""
    from common.config import PROMPT_TEMPLATES_CONFIG

    prompt: str = PROMPT_TEMPLATES_CONFIG.get("structured_output", "")
    assert prompt, "structured_output 프롬프트가 비어 있습니다."

    # (a) 인라인 강제 인용 명령이 제거되었는지 확인
    assert "반드시 `[doc:N]` 형태로 인용" not in prompt
    # (b) citations가 여전히 스키마/프롬프트의 유일한 인용 출처로 유지되는지 확인
    assert "citations" in prompt
    assert "citations[]" in prompt


def test_structured_output_schema_retains_citations_key():
    """로드된 structured_output 프롬프트 본문에 citations 키 정의가 존재하는지 검증."""
    from common.config import PROMPT_TEMPLATES_CONFIG

    prompt = PROMPT_TEMPLATES_CONFIG.get("structured_output", "")
    assert '"citations"' in prompt
