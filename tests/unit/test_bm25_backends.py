"""BM25 backend gradual-wrapping contracts (P0-1)."""

from langchain_core.documents import Document

from core.bm25_backends import BACKEND_BM25S, BACKEND_RANK_BM25, is_bm25s_available
from core.retriever_factory import _bm25_results_with_scores, create_bm25_retriever


def _docs() -> list[Document]:
    return [
        Document(page_content="alpha attention mechanism paper", metadata={}),
        Document(page_content="beta cooking recipe soup", metadata={}),
        Document(page_content="gamma alpha mixture model", metadata={}),
    ]


def test_default_backend_is_rank_bm25():
    retriever = create_bm25_retriever(_docs())
    assert retriever is not None
    # rank-bm25 path exposes BM25Retriever type; bm25s adapter only on opt-in.
    assert type(retriever).__name__ != "Bm25sAdapter"


def test_bm25s_requested_without_install_falls_back():
    # bm25s is optional; when missing we must fall back, never crash.
    if is_bm25s_available():
        return  # environment has bm25s; fallback path not applicable here.
    retriever = create_bm25_retriever(_docs(), backend=BACKEND_BM25S)
    assert retriever is not None
    assert type(retriever).__name__ != "Bm25sAdapter"


def test_duck_typed_adapter_scoring():
    """_bm25_results_with_scores works on any vectorizer/docs/preprocess_func."""

    class _Vec:
        def get_scores(self, tokens):  # noqa: ANN001, ANN202
            return [3.0, 1.0, 2.0]

    class _Fake:
        docs = _docs()
        k = 2

        @staticmethod
        def preprocess_func(q):  # noqa: ANN001, ANN205
            return q.split()

        vectorizer = _Vec()

    out = _bm25_results_with_scores(_Fake(), "alpha", 2)  # type: ignore[arg-type]
    assert [d.page_content for d in out][0].startswith("alpha attention")
    assert out[0].metadata["score"] == 3.0
    # Original pool docs must not be mutated (D2 invariant).
    assert "score" not in _Fake.docs[0].metadata


def test_backend_constants():
    assert BACKEND_RANK_BM25 == "rank-bm25"
    assert BACKEND_BM25S == "bm25s"


async def test_bm25s_adapter_uses_scored_path():
    """Bm25sAdapter duck-type passes the Mock guard and scores via thread path."""
    from core.bm25_backends import Bm25sAdapter
    from core.retriever_factory import search_bm25_with_scores

    class _Idx:
        def get_scores(self, tokens):  # noqa: ANN001, ANN202
            assert tokens  # tokenizer produced non-empty tokens
            return [2.0, 5.0, 1.0]

    adapter = Bm25sAdapter(docs=_docs(), bm25_index=_Idx(), k=2)
    out = await search_bm25_with_scores(adapter, "alpha", 2)
    assert len(out) == 2
    assert out[0].metadata["score"] == 5.0
    assert "score" not in adapter.docs[1].metadata
