"""Cited sources stay reachable in the one-line reference list.

The pipeline treats ``citations[]`` as the single source of truth. When
reranked relevance order puts a cited chunk past the fifth document, the UI
must still show that chunk as a clickable source row.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from ui.components import chat_references as refs


def _doc(doc_id: str, page: int) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"doc_id": doc_id, "page": page}
    doc.page_content = f"본문 {page} " + "내용 " * 20
    return doc


def _cited_rows(
    msg_id: str, documents: list[Any], citations: list[dict[str, Any]]
) -> list[str]:
    with patch.object(refs, "st") as mock_st:
        assert refs.render_reference_list(msg_id, documents, citations) is True
    return [call.kwargs["key"] for call in mock_st.button.call_args_list]


def test_cited_chunk_past_fifth_document_is_rendered() -> None:
    """A citation at rank 6 replaces an uncited top-five row."""
    documents = [_doc(f"doc-{page}", page) for page in range(1, 10)]
    citations = [
        {
            "doc_id": "doc-6",
            "text_span": "인용 근거",
            "section": "3 CM3",
            "page": 6,
        },
        {"doc_id": "doc-1", "text_span": "첫 인용", "section": "1 서론", "page": 1},
    ]

    keys = _cited_rows("cited-past-fifth", documents, citations)

    pages = {int(key.rsplit("_", 1)[1]) for key in keys}
    assert 6 in pages
    assert 1 in pages
    assert 2 not in pages
    assert len(keys) == 2


def test_cited_selection_keeps_five_row_cap() -> None:
    """With more than five cited documents the first five cited rows render."""
    documents = [_doc(f"doc-{page}", page) for page in range(1, 10)]
    citations = [
        {"doc_id": f"doc-{page}", "text_span": f"인용 {page}", "page": page}
        for page in (9, 8, 7, 6, 5, 4, 3)
    ]

    keys = _cited_rows("cited-cap", documents, citations)

    pages = [int(key.rsplit("_", 1)[1]) for key in keys]
    assert len(keys) == 5
    assert set(pages) == {3, 4, 5, 6, 7}


def test_no_citations_keeps_relevance_order() -> None:
    """Without citations the top-five document order is unchanged."""
    documents = [_doc(f"doc-{page}", page) for page in range(1, 7)]

    keys = _cited_rows("no-citations", documents, [])

    pages = [int(key.rsplit("_", 1)[1]) for key in keys]
    assert pages == [1, 2, 3, 4, 5]
