import pytest
from langchain_core.documents import Document

from core.graph._grade_verify import grade_documents


@pytest.mark.asyncio
async def test_grade_documents_with_none_rerank_score():
    # 문서 메타데이터에 rerank_score가 None, 문자열, NaN, 정상값 등으로 섞여 있을 때
    # grade_documents가 500 크래시(TypeError)를 발생시키지 않고 안전하게 처리하는지 검증
    docs = [
        Document(page_content="doc 1 content", metadata={"rerank_score": None}),
        Document(page_content="doc 2 content", metadata={"rerank_score": "0.85"}),
        Document(page_content="doc 3 content", metadata={"rerank_score": float("nan")}),
        Document(page_content="doc 4 content", metadata={"rerank_score": -0.1}),
        Document(page_content="doc 5 content", metadata={"rerank_score": 1}),
    ]
    state = {
        "input": "What is RAG?",
        "relevant_docs": docs,
        "is_cached": False,
        "intent": "rag",
    }
    config = {"configurable": {}}

    # 크래시(TypeError 등) 없이 실행 완료되어야 함
    try:
        res = await grade_documents(state, config, writer=None)
        assert isinstance(res, dict)
    except TypeError as e:
        pytest.fail(
            f"grade_documents crashed with TypeError due to None/invalid rerank_score: {e}"
        )
