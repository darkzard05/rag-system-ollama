"""show_caption flag gates the expander excerpt caption (failing-first).

Spec:
- ``_render_references_content`` accepts ``show_caption`` (default True).
- ``True`` keeps today's behavior: 1-line ``st.caption`` excerpt + button
  ``help=`` full excerpt.
- ``False`` (generation expander path) suppresses the caption but keeps
  the button ``help=`` tooltip unaffected.
- Empty excerpt renders no caption even with ``True`` (vacuous-pass guard).
"""

from typing import Any
from unittest.mock import MagicMock, patch

LONG_SPAN = "인용 원문 전체 문장입니다. " * 30  # ~450 chars, multi-word


def _doc(page: int, text: str, doc_id: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page, "doc_id": doc_id}
    doc.page_content = text
    return doc


def test_show_caption_true_renders_caption(session_context: str) -> None:
    """Explicit True -> caption called once with the excerpt text."""
    from ui.components.chat_references import _render_references_content

    documents: list[Any] = [_doc(3, LONG_SPAN, "chunk_A")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_A", "text_span": LONG_SPAN, "section": "제3장", "page": 3},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content(
            "msg_cap", documents, citations, show_caption=True
        )
    assert rendered is True
    captions = mock_st.caption.call_args_list
    assert len(captions) == 1
    assert "인용 원문 전체 문장입니다." in str(captions[0].args[0])


def test_show_caption_false_suppresses_caption_keeps_help(
    session_context: str,
) -> None:
    """False -> no caption, but button + help= excerpt stay intact."""
    from ui.components.chat_references import _render_references_content

    documents: list[Any] = [_doc(3, LONG_SPAN, "chunk_A")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_A", "text_span": LONG_SPAN, "section": "제3장", "page": 3},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content(
            "msg_cap", documents, citations, show_caption=False
        )
    assert rendered is True
    mock_st.caption.assert_not_called()
    buttons = mock_st.button.call_args_list
    assert len(buttons) == 1
    assert "인용 원문 전체 문장입니다." in str(buttons[0].kwargs.get("help", ""))


def test_empty_excerpt_renders_no_caption(session_context: str) -> None:
    """No text_span and no doc text -> no caption even with True."""
    from ui.components.chat_references import _render_references_content

    documents: list[Any] = [_doc(3, "", "chunk_A")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_A", "section": "제3장", "page": 3},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content(
            "msg_cap", documents, citations, show_caption=True
        )
    assert rendered is True
    mock_st.caption.assert_not_called()
