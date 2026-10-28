import asyncio
import pytest
from core.model_loader import MemoizingEmbedding
from langchain_core.embeddings import FakeEmbeddings


@pytest.mark.asyncio
async def test_memoizing_embedding_async_methods():
    inner = FakeEmbeddings(size=1536)
    memo = MemoizingEmbedding(inner)

    vec1 = await memo.aembed_query("hello world")
    assert isinstance(vec1, list)
    assert len(vec1) == 1536

    # Test cache hit
    vec2 = await memo.aembed_query("hello world")
    assert vec1 == vec2
    assert memo.stats()["hits"] >= 1

    docs_vecs = await memo.aembed_documents(["doc1", "doc2"])
    assert len(docs_vecs) == 2
    assert len(docs_vecs[0]) == 1536
