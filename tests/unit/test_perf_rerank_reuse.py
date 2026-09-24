"""리랭커 metadata 재사용 bound (TDD).

- 전원히트시 embed_documents 스킵 (t_batch→0), recall@5 불변.
- 정규화(" ".join(text.split()), chunker:329 동일) 해시 동등성.
- model_name 키 + dim 불일치시 재계산.
"""

import pytest
from langchain_core.documents import Document

import core.async_reranker as ar
from core.async_reranker import _text_hash


@pytest.fixture(autouse=True)
def _reset_doc_cache():
    ar._doc_emb_cache.clear()
    yield
    ar._doc_emb_cache.clear()


class _FakeEmbedder:
    def __init__(self, dim: int = 4, model: str = "test-model") -> None:
        self.dim = dim
        self.model = model
        self.doc_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.doc_calls += 1
        return [[float(len(t) % 7 + 1)] * self.dim for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query) % 7 + 1)] * self.dim


def _docs_with_vectors(n: int = 5, dim: int = 4) -> list[Document]:
    return [
        Document(
            page_content=f"doc content {i}",
            metadata={"embedding_vector": [float(i + 1)] * dim},
        )
        for i in range(n)
    ]


def test_text_hash_normalizes_like_chunker() -> None:
    """정규화 해시 동등성: 공백 변형이 동일 해시 (chunker:329 동일)."""
    assert _text_hash("a  b\n c") == _text_hash("a b c")


@pytest.mark.asyncio
async def test_full_metadata_hit_skips_embed_documents() -> None:
    """전원히트시 embed_documents 스킵 + recall@5(순서) 불변."""
    from core.async_reranker import AsyncSemanticReranker

    emb = _FakeEmbedder()
    r = AsyncSemanticReranker(emb)  # type: ignore[arg-type]
    docs = _docs_with_vectors(5)
    ranked, scores = await r.rerank(docs, query="q", top_k=5)
    assert emb.doc_calls == 0
    assert len(ranked) == 5
    assert len(scores) == 5
    assert scores == sorted(scores, reverse=True)


@pytest.mark.asyncio
async def test_dim_mismatch_recomputes() -> None:
    """dim 불일치 stale 벡터는 무시하고 재계산한다."""
    from core.async_reranker import AsyncSemanticReranker

    emb = _FakeEmbedder(dim=4)
    r = AsyncSemanticReranker(emb)  # type: ignore[arg-type]
    docs = [Document(page_content="hello", metadata={"embedding_vector": [1.0]})]
    await r.rerank(docs, query="q", top_k=1)
    assert emb.doc_calls == 1


@pytest.mark.asyncio
async def test_model_key_isolates_cache() -> None:
    """model_name이 다르면 모듈 캐시를 재사용하지 않는다."""
    from core.async_reranker import AsyncSemanticReranker

    emb_a = _FakeEmbedder(model="model-a")
    emb_b = _FakeEmbedder(model="model-b")
    docs = [Document(page_content="shared text", metadata={})]
    await AsyncSemanticReranker(emb_a).rerank(  # type: ignore[arg-type]
        docs, query="q", top_k=1
    )
    assert emb_a.doc_calls == 1
    docs2 = [Document(page_content="shared text", metadata={})]
    await AsyncSemanticReranker(emb_b).rerank(  # type: ignore[arg-type]
        docs2, query="q", top_k=1
    )
    assert emb_b.doc_calls == 1
