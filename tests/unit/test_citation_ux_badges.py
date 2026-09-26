"""Citation UX overhaul: inline marker line + hover excerpts (failing-first).

Spec (v2: single horizontal small-grey text line replaces the badge stack):
- Answer end shows one horizontal ``[1] [2] ...`` line via a single
  ``st.markdown`` call (no badge widgets, no inline help/tooltips).
- Each expander jump button carries the excerpt on hover via ``help=``
  plus a 1-line ``st.caption`` excerpt (Streamlit help tooltips break on
  ``\\n\\n`` per issue #13339, so excerpts are single-line via
  :func:`normalize_excerpt`).
- :func:`normalize_excerpt` (strip + collapse whitespace, truncate ~500 chars)
  is shared by both render paths.
"""

from typing import Any
from unittest.mock import MagicMock, patch

LONG_SPAN = "인용 원문 전체 문장입니다. " * 30  # ~450 chars, multi-word


def _doc(page: int, text: str, doc_id: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page, "doc_id": doc_id}
    doc.page_content = text
    return doc


def test_normalize_excerpt_collapses_newlines() -> None:
    """Double newlines/tabs collapse to single spaces (tooltip-safe)."""
    from ui.components.chat_references import normalize_excerpt

    out = normalize_excerpt("앞 문장.\n\n뒷 문장.\n\t탭 들여쓰기")
    assert "\n" not in out
    assert "\t" not in out
    assert out == "앞 문장. 뒷 문장. 탭 들여쓰기"


def test_normalize_excerpt_strips_and_truncates() -> None:
    """Strip edges; ~500 chars max with ellipsis suffix."""
    from ui.components.chat_references import normalize_excerpt

    out = normalize_excerpt("   " + LONG_SPAN * 2 + "   ")
    assert out == out.strip()
    assert len(out) <= 503
    assert out.endswith("...")
    assert "\n" not in out


def test_normalize_excerpt_short_text_untouched() -> None:
    """Short single-line input passes through (no ellipsis)."""
    from ui.components.chat_references import normalize_excerpt

    assert normalize_excerpt("짧은 발췌.") == "짧은 발췌."


def test_normalize_excerpt_non_string_empty() -> None:
    """Non-string/empty inputs degrade to empty string."""
    from ui.components.chat_references import normalize_excerpt

    assert normalize_excerpt(None) == ""
    assert normalize_excerpt(123) == ""
    assert normalize_excerpt("") == ""


def test_badge_render_has_no_full_text_but_help_excerpt() -> None:
    """Answer-end markers are one horizontal line; excerpts live in expander."""
    from ui.components.chat_references import render_inline_citation_badges

    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_A", "text_span": LONG_SPAN, "section": "제3장", "page": 3},
        {"doc_id": "chunk_B", "text_span": LONG_SPAN, "section": "제4장", "page": 7},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = render_inline_citation_badges(citations)
    assert rendered is True
    mock_st.badge.assert_not_called()
    assert mock_st.markdown.call_count == 1
    line = str(mock_st.markdown.call_args.args[0])
    assert "[1]" in line and "[2]" in line
    assert "\n" not in line
    assert "인용 원문 전체 문장입니다." not in line
    assert mock_st.markdown.call_args.kwargs.get("help") is None


def test_badge_render_empty_without_citations() -> None:
    """No citations -> no markers, returns False."""
    from ui.components.chat_references import render_inline_citation_badges

    with patch("ui.components.chat_references.st") as mock_st:
        assert render_inline_citation_badges(None) is False
        assert render_inline_citation_badges([]) is False
    mock_st.badge.assert_not_called()
    mock_st.markdown.assert_not_called()


def test_jump_button_help_carries_excerpt() -> None:
    """Expander jump buttons keep label/key but gain help= excerpt."""
    from ui.components.chat_references import _render_references_content

    documents: list[Any] = [_doc(3, LONG_SPAN, "chunk_A")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_A", "text_span": LONG_SPAN, "section": "제3장", "page": 3},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content("msg_help", documents, citations)
    assert rendered is True
    buttons = mock_st.button.call_args_list
    assert len(buttons) == 1
    kwargs = buttons[0].kwargs
    assert kwargs["key"] == "pop_doc_msg_help_3_0"
    assert "p.3" in str(buttons[0].args[0])
    assert "인용 원문 전체 문장입니다." in str(kwargs.get("help", ""))
    assert "\n" not in str(kwargs.get("help", ""))


def test_jump_behavior_intact_branch_d() -> None:
    """Branch-D dedup intact: same text on 4 pages -> one button, min page kept."""
    from ui.components.chat_references import (
        _handle_page_jump,
        _render_references_content,
    )

    body = "동일한 인용 본문 텍스트 " * 20
    sections = ["제3장 실험 결과 상세 분석", "제4장 고찰", "일반 본문", "문서 본문"]
    pages = [3, 10, 11, 14]
    documents: list[Any] = [
        _doc(p, body, f"chunk_{p}") for p, s in zip(pages, sections, strict=True)
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content("msg_branch_d", documents)
    assert rendered is True
    buttons = mock_st.button.call_args_list
    assert len(buttons) == 1
    assert "p.3" in str(buttons[0].args[0])
    assert buttons[0].kwargs["on_click"] is _handle_page_jump
    assert buttons[0].kwargs["args"] == (3,)


def test_no_blue_full_text_block_after_answer() -> None:
    """apply_tooltips output has no full-sentence blue citation-sources block."""
    from common.utils import apply_tooltips_to_response

    documents: list[Any] = [_doc(7, "Deep content about topic X.", "doc_abc123")]
    citations: list[dict[str, Any]] = [
        {
            "doc_id": "doc_abc123",
            "text_span": LONG_SPAN,
            "section": "§3",
            "page": 7,
            "score": 0.91,
        }
    ]
    out = apply_tooltips_to_response(
        "The model says topic X applies.", documents, citations=citations
    )
    assert "citation-sources" not in out
    assert "citation-source" not in out
    assert LONG_SPAN.strip() not in out
