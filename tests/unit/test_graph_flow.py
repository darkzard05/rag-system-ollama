import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from core.graph_builder import _merge_adjacent_chunks

from core.graph_builder import (
    preprocess,
    retrieve_and_rerank,
    grade_documents,
    rewrite_query,
    generate,
    GraphState,
)


def json_response(**fields):
    """LLM JSON 모드 응답을 모사하는 SimpleNamespace 객체를 생성합니다."""
    return SimpleNamespace(content=json.dumps(fields))


@pytest.fixture
def mock_llm():
    """LLM과 JSON 구조화 응답(json_llm)을 모킹합니다."""
    llm = MagicMock()
    json_llm = AsyncMock()
    llm.bind.return_value = json_llm

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

    return llm, json_llm


@pytest.fixture
def mock_reranker():
    """get_async_reranker()를 대체해 환경 의존 Ollama 호출을 차단합니다.

    retrieve_and_rerank가 함수 내부에서 `from core.async_reranker import
    get_async_reranker`로 바인딩하므로 소스 모듈 경로를 패치합니다.
    """
    with patch(
        "core.async_reranker.get_async_reranker", new_callable=AsyncMock
    ) as mock_get:
        reranker = AsyncMock()
        # 전달받은 문서를 그대로 최종 선별 결과로 돌려줍니다.
        reranker.rerank.side_effect = lambda docs, **kwargs: (docs, None)
        mock_get.return_value = reranker
        yield reranker


@pytest.fixture
def mock_retrievers():
    """BM25 및 FAISS 리트리버를 모킹합니다."""
    bm25 = AsyncMock()
    faiss = AsyncMock()
    return bm25, faiss


@pytest_asyncio.fixture
async def compiled_workflow():
    """실제 graph_builder.build_graph()와 동일한 구조의 테스트용 그래프를 생성합니다."""
    from core.graph_builder import invalidate_graph_cache, build_graph

    invalidate_graph_cache()
    with patch("aiosqlite.connect", side_effect=Exception("force InMemorySaver")):
        graph = await build_graph()
    invalidate_graph_cache()
    return graph


@pytest.mark.asyncio
async def test_workflow_general_intent(compiled_workflow, mock_llm):
    """일상 대화(General) 흐름 테스트: preprocess -> generate -> END"""
    llm, _json_llm = mock_llm

    config = {"configurable": {"llm": llm, "thread_id": "test_thread"}}
    inputs = {
        "input": "안녕",
        "chat_history": [],
        "retry_count": 0,
        "search_queries": [],
        "relevant_docs": [],
    }

    # Execute
    result = await compiled_workflow.ainvoke(inputs, config=config)

    # Verify
    assert result["intent"] == "general"
    assert "테스트 답변" in result["response"]


@pytest.mark.asyncio
async def test_workflow_rag_success_path(
    compiled_workflow, mock_llm, mock_retrievers, mock_reranker
):
    """RAG 성공 흐름 테스트: preprocess -> retrieve -> grade(YES) -> generate -> END"""
    llm, json_llm = mock_llm
    bm25, faiss = mock_retrievers

    # Mock Retrieve
    doc = Document(page_content="RAG 지식", metadata={"page": 1})
    bm25.ainvoke.return_value = [doc]
    faiss.ainvoke.return_value = []

    # Mock Grade (JSON 모드)
    json_llm.ainvoke.return_value = json_response(
        action="generate",
        is_relevant=True,
        relevant_entities=["RAG"],
        reason="Matched",
        optimized_query=None,
    )

    config = {
        "configurable": {
            "llm": llm,
            "bm25_retriever": bm25,
            "faiss_retriever": faiss,
            "thread_id": "test_thread_rag",
        }
    }
    inputs = {"input": "RAG가 뭐야?", "chat_history": []}

    # Execute
    result = await compiled_workflow.ainvoke(inputs, config=config)

    # Verify
    assert result["intent"] == "generate"  # grade 이후 전이됨
    assert len(result["relevant_docs"]) > 0
    assert "테스트 답변" in result["response"]


@pytest.mark.asyncio
async def test_workflow_rag_retry_path(
    compiled_workflow, mock_llm, mock_retrievers, mock_reranker
):
    """RAG 재시도 흐름 테스트"""
    llm, json_llm = mock_llm
    bm25, faiss = mock_retrievers

    bm25.ainvoke.side_effect = [
        [Document(page_content="무관한 정보")],
        [Document(page_content="정확한 정보")],
    ]
    faiss.ainvoke.return_value = []

    json_llm.ainvoke.side_effect = [
        json_response(
            action="rewrite",
            is_relevant=False,
            relevant_entities=[],
            reason="Irrelevant",
            optimized_query="재구성 쿼리",
        ),  # 1차 Grade → rewrite
        json_response(
            action="generate",
            is_relevant=True,
            relevant_entities=["정보"],
            reason="Matched",
            optimized_query=None,
        ),  # 2차 Grade → generate
    ]

    config = {
        "configurable": {
            "llm": llm,
            "bm25_retriever": bm25,
            "faiss_retriever": faiss,
            "thread_id": "test_thread_rag",
        }
    }
    inputs = {"input": "정보를 알려줘", "chat_history": []}

    # Execute
    result = await compiled_workflow.ainvoke(inputs, config=config)

    # Verify
    assert result["retry_count"] == 1
    assert "재구성 쿼리" in result["search_queries"]
    assert "테스트 답변" in result["response"]


@pytest.mark.asyncio
async def test_retrieve_and_rerank_uses_smaller_candidate_pool(
    mock_llm, mock_retrievers
):
    """긴 질의에서는 리랭크를 수행해 후보 수를 줄여 리랭크 비용을 낮춘다."""
    llm, _json_llm = mock_llm
    bm25, faiss = mock_retrievers

    docs = [
        Document(page_content=f"doc {i}", metadata={"page": i + 1}) for i in range(20)
    ]
    bm25.ainvoke.return_value = docs
    faiss.ainvoke.return_value = []

    with patch(
        "core.async_reranker.get_async_reranker", new_callable=AsyncMock
    ) as mock_get_reranker:
        mock_reranker = AsyncMock()
        mock_reranker.rerank.return_value = (docs[:5], None)
        mock_get_reranker.return_value = mock_reranker

        state = {"input": "짧은 질문입니다", "search_queries": [], "retry_count": 0}
        config = {
            "configurable": {
                "llm": llm,
                "bm25_retriever": bm25,
                "faiss_retriever": faiss,
            }
        }

        result = await retrieve_and_rerank(state, config, writer=None)

        assert result["relevant_docs"]
        assert mock_reranker.rerank.call_count == 1


@pytest.mark.asyncio
async def test_merge_adjacent_chunks_merges_consecutive_sections():
    """연속된 청크는 문맥 병합되어 더 작은 문서 세트로 정리된다."""
    docs = [
        Document(
            page_content="A",
            metadata={
                "source": "s1",
                "page": 1,
                "chunk_index": 0,
                "current_section": "sec",
            },
        ),
        Document(
            page_content="B",
            metadata={
                "source": "s1",
                "page": 1,
                "chunk_index": 1,
                "current_section": "sec",
            },
        ),
        Document(
            page_content="C",
            metadata={
                "source": "s1",
                "page": 1,
                "chunk_index": 2,
                "current_section": "other",
            },
        ),
    ]

    merged = _merge_adjacent_chunks(docs, max_tokens=100)

    assert len(merged) == 2
    assert "A\n\nB" in merged[0].page_content


@pytest.mark.asyncio
async def test_workflow_cache_hit_path(mock_llm):
    """캐시 적중 시 즉시 종료 테스트"""
    from core.graph_builder import invalidate_graph_cache, build_graph

    llm, _json_llm = mock_llm

    async def _mock_preprocess(state, config, *, writer=None):
        return {"intent": "general", "is_cached": True}

    async def _mock_generate(state, config, *, writer=None):
        return {
            "response": "캐시된 결과",
            "thought": "캐시된 생각",
            "is_cached": True,
        }

    invalidate_graph_cache()
    try:
        with (
            patch("core.graph_builder.preprocess", new=_mock_preprocess),
            patch("core.graph_builder.generate", new=_mock_generate),
            patch("aiosqlite.connect", side_effect=Exception("force InMemorySaver")),
        ):
            compiled_workflow = await build_graph()

            config = {"configurable": {"llm": llm, "thread_id": "test_thread_cache"}}
            inputs = {"input": "테스트", "chat_history": []}

            # Execute
            result = await compiled_workflow.ainvoke(inputs, config=config)

            # Verify
            assert result["is_cached"] is True
            assert result["response"] == "캐시된 결과"
    finally:
        invalidate_graph_cache()
