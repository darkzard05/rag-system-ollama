"""BM25 backend adapters (gradual wrapping).

Default backend is the existing ``rank-bm25`` path via
``langchain_community.retrievers.BM25Retriever``. The optional ``bm25s``
backend (``pip install bm25s``) exposes the same duck-typed surface so
``retriever_factory._bm25_results_with_scores`` works unchanged:

- ``preprocess_func``: query -> tokens
- ``docs``: list[Document]
- ``k``: int
- ``vectorizer.get_scores(tokenized_query)``: -> array-like scores
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document

from common.text_utils import bm25_tokenizer

logger = logging.getLogger(__name__)

BACKEND_RANK_BM25 = "rank-bm25"
BACKEND_BM25S = "bm25s"


class Bm25sVectorizerShim:
    """Expose ``get_scores`` over a ``bm25s.BM25`` index."""

    def __init__(self, bm25_index: Any) -> None:
        self._index = bm25_index

    def get_scores(self, tokenized_query: list[str]) -> Any:
        # bm25s versions differ: prefer get_scores, fall back to retrieve paths.
        get_scores = getattr(self._index, "get_scores", None)
        if callable(get_scores):
            return get_scores(tokenized_query)
        retrieve = getattr(self._index, "retrieve", None)
        if callable(retrieve):
            # retrieve may return (docs, scores) or ranked objects; last resort.
            res = retrieve([tokenized_query])
            if isinstance(res, tuple) and len(res) == 2:
                return res[1][0] if res[1] is not None else []
            raise AttributeError("bm25s retrieve() shape not supported for scoring")
        raise AttributeError("bm25s index has neither get_scores nor retrieve")


class Bm25sAdapter:
    """Duck-typed BM25 retriever backed by ``bm25s`` (optional dependency)."""

    def __init__(self, docs: list[Document], bm25_index: Any, k: int) -> None:
        self.docs = docs
        self.preprocess_func = bm25_tokenizer
        self.k = k
        self.vectorizer = Bm25sVectorizerShim(bm25_index)
        self._bm25_index = bm25_index

    async def ainvoke(self, query: str) -> list[Document]:
        """Fallback path for generic callers (returns top-k without scores)."""
        from core.retriever_factory import _bm25_results_with_scores

        return _bm25_results_with_scores(self, query, self.k)


def build_bm25s_retriever(docs: list[Document], k: int) -> Bm25sAdapter:
    """Build a bm25s-backed retriever reusing the project Korean tokenizer."""
    try:
        import bm25s  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "bm25s backend requested but 'bm25s' is not installed. "
            "Install with: pip install -r requirements/optional-bm25s.txt"
        ) from e

    tokenized = [bm25_tokenizer(d.page_content) for d in docs]
    # Guard against empty-token docs (bm25s handles them, but keep parity
    # with rank-bm25 empty-corpus check done by the caller).
    index = bm25s.BM25()
    try:
        index.index(tokenized)
    except TypeError:
        # Older bm25s accepted tokenize-on-init; retry with corpus passthrough.
        index = bm25s.BM25(corpus=docs)  # type: ignore[call-arg]
    return Bm25sAdapter(docs=docs, bm25_index=index, k=k)


def is_bm25s_available() -> bool:
    """Return True when the optional ``bm25s`` dependency is importable."""
    try:
        import bm25s  # noqa: F401  # type: ignore[import-not-found]
    except ImportError:
        return False
    return True
