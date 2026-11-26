"""Reference rows and inline badge markers share one citation order.

The pipeline emits ``citations[]`` once. The inline markers number the
page-sorted entries from ``ordered_reference_entries``; the one-line source
list must number those same entries, or ``[1]`` points at a different PDF
page than row ``1``.
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


def test_reference_rows_use_badge_page_order() -> None:
    """Row numbers match the page order of the inline [n] markers."""
    documents = [
        _doc("doc-1", 50),
        _doc("doc-2", 40),
        _doc("doc-3", 30),
        _doc("doc-4", 20),
        _doc("doc-5", 10),
        _doc("doc-6", 60),
    ]
    citations: list[dict[str, Any]] = [
        {"doc_id": "doc-5", "text_span": "다섯 번째 인용", "page": 10},
        {"doc_id": "doc-2", "text_span": "두 번째 인용", "page": 40},
        {"doc_id": "doc-6", "text_span": "여섯 번째 인용", "page": 60},
        {"doc_id": "doc-1", "text_span": "첫 번째 인용", "page": 50},
    ]

    with patch.object(refs, "st") as mock_st:
        assert refs.render_reference_list("badge-order", documents, citations) is True
        keys = [call.kwargs["key"] for call in mock_st.button.call_args_list]
    row_pages = [int(key.rsplit("_", 1)[1]) for key in keys]

    with patch.object(refs, "st") as mock_st:
        assert refs.render_inline_citation_badges(citations, documents) is True
        markers = str(mock_st.markdown.call_args.args[0])

    assert markers == "[1] [2] [3] [4]"
    assert row_pages == [10, 40, 50, 60]


def test_badge_order_survives_rerun_with_same_citation_count() -> None:
    """A citation upsert with the same length must not reuse stale rows."""
    documents = [_doc("doc-1", 10), _doc("doc-2", 20), _doc("doc-3", 30)]
    first = [{"doc_id": "doc-1", "text_span": "예전 인용", "page": 10}]
    second = [{"doc_id": "doc-3", "text_span": "새 인용", "page": 30}]

    with patch.object(refs, "st") as mock_st:
        refs.render_reference_list("upsert-order", documents, first)
    with patch.object(refs, "st") as mock_st:
        refs.render_reference_list("upsert-order", documents, second)
        labels = [str(call.args[0]) for call in mock_st.button.call_args_list]

    assert len(labels) == 1
    assert "새 인용" in labels[0]
    assert "예전 인용" not in labels[0]
