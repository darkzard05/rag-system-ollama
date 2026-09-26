"""Citation UX v2 refinements (failing-first).

Spec (follow-up to grey-badge overhaul):
- (a) Citation jump syncs the viewer Page number_input: ``navigate_to_page``
  writes the widget-bound key (``PDF_NAV_INPUT_KEY``) in callback phase so the
  input displays the jumped page instead of staying stale.
- (b) Answer-end citations render as ONE horizontal small grey text line
  (``[1] [2] ...``) via a single ``st.markdown`` call — no badge widgets,
  no inline help/tooltips (hover lives only in the expander).
- (c) Each expander row keeps its jump button byte-identical
  (label/key/on_click/args) with ``help=`` full excerpt, plus a 1-line
  excerpt line. NOTE: installed Streamlit 1.60 exposes NO ``wrap`` param on
  ``st.markdown``/``st.caption``/``st.text`` (signature-introspected), so the
  line falls back to a single-line ``st.caption`` (newline-stripped) with the
  full excerpt on the button ``help=`` — no hand-rolled CSS ellipsis.
- (d) ``normalize_excerpt`` keeps newline-strip + whitespace collapse but
  relaxes the cut from ~200 to ~500 chars (clipping is CSS-driven; hover
  previews carry real excerpts).
"""

from typing import Any
from unittest.mock import MagicMock, patch

from ui.widget_keys import PDF_NAV_INPUT_KEY

LONG_SPAN = "인용 원문 전체 문장입니다. " * 30  # ~450 chars, multi-word
SPAN_600 = "발췌 문장입니다. " * 60  # ~600 chars


def _doc(page: int, text: str, doc_id: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page, "doc_id": doc_id}
    doc.page_content = text
    return doc


def test_normalize_excerpt_relaxes_cut_to_500(session_context: str) -> None:
    """~450-char span survives intact (was cut at ~200); ~600-char is cut."""
    from ui.components.chat_references import normalize_excerpt

    kept = normalize_excerpt(LONG_SPAN)
    assert kept == LONG_SPAN.strip()
    assert "\n" not in kept
    assert not kept.endswith("...")

    cut = normalize_excerpt(SPAN_600)
    assert cut.endswith("...")
    assert len(cut) <= 503
    assert "\n" not in cut


def test_normalize_excerpt_still_collapses_newlines(session_context: str) -> None:
    """Newline-strip + whitespace collapse preserved under the new limit."""
    from ui.components.chat_references import normalize_excerpt

    out = normalize_excerpt("앞 문장.\n\n뒷 문장.\n\t탭 들여쓰기")
    assert out == "앞 문장. 뒷 문장. 탭 들여쓰기"


def test_inline_markers_single_horizontal_line(session_context: str) -> None:
    """Five citations -> exactly one st.markdown call, one horizontal line."""
    from ui.components.chat_references import render_inline_citation_badges

    citations: list[dict[str, Any]] = [
        {"doc_id": f"chunk_{i}", "text_span": f"발췌 {i}", "page": i + 1}
        for i in range(5)
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = render_inline_citation_badges(citations)
    assert rendered is True
    mock_st.badge.assert_not_called()
    assert mock_st.markdown.call_count == 1
    body = str(mock_st.markdown.call_args.args[0])
    for n in range(1, 6):
        assert f"[{n}]" in body
    assert "\n" not in body
    assert mock_st.markdown.call_args.kwargs.get("help") is None


def test_inline_markers_empty_renders_nothing(session_context: str) -> None:
    """No citations -> False, no markdown/badge calls."""
    from ui.components.chat_references import render_inline_citation_badges

    with patch("ui.components.chat_references.st") as mock_st:
        assert render_inline_citation_badges(None) is False
        assert render_inline_citation_badges([]) is False
    mock_st.markdown.assert_not_called()
    mock_st.badge.assert_not_called()


def test_expander_row_keeps_button_and_adds_excerpt_line(
    session_context: str,
) -> None:
    """Jump button byte-identical + help carries full excerpt; 1-line caption."""
    from ui.components.chat_references import (
        _handle_page_jump,
        _render_references_content,
    )

    documents: list[Any] = [_doc(3, LONG_SPAN, "chunk_A")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_A", "text_span": LONG_SPAN, "section": "제3장", "page": 3},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content("msg_v2", documents, citations)
    assert rendered is True
    buttons = mock_st.button.call_args_list
    assert len(buttons) == 1
    kwargs = buttons[0].kwargs
    assert kwargs["key"] == "pop_doc_msg_v2_3_0"
    assert "p.3" in str(buttons[0].args[0])
    assert kwargs["on_click"] is _handle_page_jump
    assert kwargs["args"] == (3,)
    assert "인용 원문 전체 문장입니다." in str(kwargs.get("help", ""))
    assert "\n" not in str(kwargs.get("help", ""))
    captions = mock_st.caption.call_args_list
    assert len(captions) == 1
    line = str(captions[0].args[0])
    assert "인용 원문 전체 문장입니다." in line
    assert "\n" not in line


def test_navigate_to_page_syncs_nav_input(session_context: str) -> None:
    """Regression: jumped page lands on the widget-bound nav input key."""
    from core.session import SessionManager
    from ui.components.common import navigate_to_page

    fake_state: dict[str, Any] = {}
    with patch("ui.components.common.st") as mock_st:
        mock_st.session_state = fake_state  # type: ignore[attr-defined]
        navigate_to_page(3)
    assert SessionManager.get("current_page") == 3
    assert fake_state.get(PDF_NAV_INPUT_KEY) == 3


def test_handle_page_jump_syncs_nav_input(session_context: str) -> None:
    """End-to-end jump: citation click sets manager page AND nav input value."""
    from core.session import SessionManager
    from ui.components.chat_references import _handle_page_jump

    fake_state: dict[str, Any] = {}
    with (
        patch("ui.components.chat_references.st") as mock_ref_st,
        patch("ui.components.common.st") as mock_common_st,
    ):
        mock_ref_st.session_state = fake_state  # type: ignore[attr-defined]
        mock_common_st.session_state = fake_state  # type: ignore[attr-defined]
        _handle_page_jump(3)
    assert SessionManager.get("current_page") == 3
    assert fake_state.get(PDF_NAV_INPUT_KEY) == 3
