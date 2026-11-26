"""Reference/appendix chunks must not crowd out body candidates at retrieval.

The CM3 paper repro (`cm3`, `vqvae-gan`) showed 9 of 10 retrieved chunks coming
from the bibliography (p14-18) and the appendix (p19), leaving the body passage
that defines VQVAE-GAN (p3) out of the context entirely.
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document

from core.graph.graph_builder import retrieve_and_rerank


def _doc(doc_id: str, page: int, reference: bool) -> Document:
    section = "REFERENCE" if reference else "2 DATA"
    return Document(
        page_content=f"{doc_id} 본문 문맥 보강 문장 " * 4,
        metadata={
            "doc_id": doc_id,
            "score": 10.0 if reference else 1.0,
            "source": "2201.07520v1.pdf",
            "page": page,
            "chunk_index": page,
            "start_index": page * 100,
            "end_index": page * 100 + 99,
            "section_path": section,
            "is_reference": reference,
        },
    )


@pytest.mark.asyncio
async def test_reference_chunks_are_excluded_when_body_candidates_exist():
    bm25 = AsyncMock()
    faiss = AsyncMock()
    bm25.ainvoke.return_value = [
        _doc(f"ref_{i}", 14 + i, reference=True) for i in range(4)
    ] + [_doc(f"body_{i}", i + 1, reference=False) for i in range(5)]
    faiss.ainvoke.return_value = [_doc("body_9", 9, reference=False)]

    state = {
        "input": "vqvae-gan",
        "search_queries": [],
        "search_weights": {"bm25": 0.5, "faiss": 0.5},
        "retry_count": 0,
    }
    config = {"configurable": {"bm25_retriever": bm25, "faiss_retriever": faiss}}

    with patch(
        "core.async_reranker.get_async_reranker", new_callable=AsyncMock
    ) as mock_get:
        reranker = AsyncMock()
        reranker.rerank.side_effect = lambda docs, **kwargs: (docs[:5], None)
        mock_get.return_value = reranker
        result = await retrieve_and_rerank(state, config, writer=None)

    ids = [doc.metadata.get("doc_id") for doc in result["relevant_docs"]]
    assert "body_3" in ids
    assert "body_1" in ids
    assert not [doc_id for doc_id in ids if str(doc_id).startswith("ref_")]


@pytest.mark.asyncio
async def test_reference_chunks_kept_when_no_body_candidate_exists():
    """Appendix-only questions must still retrieve something."""
    bm25 = AsyncMock()
    faiss = AsyncMock()
    bm25.ainvoke.return_value = [
        _doc("appendix_19", 19, reference=True),
        _doc("appendix_20", 20, reference=True),
    ]
    faiss.ainvoke.return_value = []

    state = {
        "input": "GLUE 하이퍼파라미터",
        "search_queries": [],
        "search_weights": {"bm25": 1.0, "faiss": 0.0},
        "retry_count": 0,
    }
    config = {"configurable": {"bm25_retriever": bm25, "faiss_retriever": faiss}}

    with patch(
        "core.async_reranker.get_async_reranker", new_callable=AsyncMock
    ) as mock_get:
        reranker = AsyncMock()
        reranker.rerank.side_effect = lambda docs, **kwargs: (docs[:5], None)
        mock_get.return_value = reranker
        result = await retrieve_and_rerank(state, config, writer=None)

    ids = [doc.metadata.get("doc_id") for doc in result["relevant_docs"]]
    assert "appendix_19" in ids
