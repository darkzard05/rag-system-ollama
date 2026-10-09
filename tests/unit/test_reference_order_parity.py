"""Phase 2: badge<->expander ordering single-source parity.

- ``ordered_reference_entries`` is the shared ordering (page sort + Branch-D
  dedup) used by both the expander rows and the ``[1] [2]`` badge line.
- Badge marker count always equals the expander button count, including the
  duplicate-page merge case (previously badges used raw input order).
- Ghost skeleton text goes through ``t("ref_ghost_searching")`` (ko/en).
- Citations-only latest turns auto-expand the generation expander.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import ui.strings as strings_mod
from ui.strings import set_lang, t

LONG_SPAN = "인용 원문 전체 문장입니다. " * 30


def _doc(page: int, text: str, doc_id: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page, "doc_id": doc_id}
    doc.page_content = text
    return doc


def test_ordered_entries_sorted_and_deduped() -> None:
    """Unsorted input with duplicate pages collapses to sorted unique pages."""
    from ui.components.chat_references import ordered_reference_entries

    documents: list[Any] = [
        _doc(7, "본문 B", "chunk_B"),
        _doc(3, "본문 A", "chunk_A"),
    ]
    citations: list[dict[str, Any]] = [
        {"doc_id": "chunk_B", "section": "제4장", "page": 7},
        {"doc_id": "chunk_A", "section": "제3장", "page": 3},
        # duplicate page 7 with a different section -> merged into one row
        {"doc_id": "chunk_B", "section": "제4장 부록", "page": 7},
    ]
    entries = ordered_reference_entries(documents, citations)
    assert [e["page"] for e in entries] == [3, 7]
    assert entries[0]["section"] == "제3장"
    assert all({"page", "section", "sid", "excerpt"} <= set(e) for e in entries)


def test_badge_count_matches_expander_buttons() -> None:
    """Badge markers == expander rows for the same merged input."""
    from ui.components.chat_references import (
        _render_references_content,
        render_inline_citation_badges,
    )

    body = "동일한 인용 본문 텍스트 " * 20
    documents: list[Any] = [_doc(3, body, "c3"), _doc(10, body, "c10")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "c3", "section": "제3장", "page": 3, "text_span": body},
        {"doc_id": "c10", "section": "제4장", "page": 10, "text_span": body},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        assert _render_references_content("msg_par", documents, citations) is True
        n_buttons = mock_st.button.call_count
    with patch("ui.components.chat_references.st") as mock_st2:
        assert render_inline_citation_badges(citations, documents) is True
        line = str(mock_st2.markdown.call_args.args[0])
    # Branch-D merges identical text -> 1 row / 1 marker (previously 2 markers).
    assert n_buttons == 1
    assert line.strip() == "[1]"


def test_badge_distinct_pages_match_expander() -> None:
    """Distinct pages: N buttons -> '[1] ... [N]' in the same order."""
    from ui.components.chat_references import (
        _render_references_content,
        render_inline_citation_badges,
    )

    documents: list[Any] = [_doc(7, "본문 B", "cB"), _doc(3, "본문 A", "cA")]
    citations: list[dict[str, Any]] = [
        {"doc_id": "cB", "section": "제4장", "page": 7},
        {"doc_id": "cA", "section": "제3장", "page": 3},
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        assert _render_references_content("msg_ord", documents, citations) is True
        n_buttons = mock_st.button.call_count
    with patch("ui.components.chat_references.st") as mock_st2:
        assert render_inline_citation_badges(citations, documents) is True
        line = str(mock_st2.markdown.call_args.args[0])
    assert n_buttons == 2
    assert line.strip() == "[1] [2]"


def test_ghost_text_uses_i18n_key() -> None:
    """Ghost skeleton renders t('ref_ghost_searching') in the active locale."""
    from ui.components.chat_references import _render_references_content

    documents: list[Any] = [_doc(5, "본문", "c5")]
    original = strings_mod.LANG
    set_lang("ko")
    try:
        expected = t("ref_ghost_searching").format(page=5)
        with patch("ui.components.chat_references.st") as mock_st:
            assert (
                _render_references_content(
                    "msg_ghost", documents, None, generating=True
                )
                is True
            )
            body = str(mock_st.markdown.call_args.args[0])
    finally:
        set_lang(original)
    assert expected in body
    assert "searching p." not in body


def test_citations_only_latest_auto_expands() -> None:
    """Citations-only 턴은 익스팬더 대신 참조 목록을 렌더한다."""
    import ui.components.chat as chat_mod

    msg: dict[str, Any] = {
        "role": "assistant",
        "content": "답변 본문",
        "msg_type": "general",
        "msg_id": "cit-only-1",
        "thought": "",
        "documents": [],
        "citations": [{"doc_id": "c1", "section": "제1장", "page": 1}],
    }
    with (
        patch.object(chat_mod.SessionManager, "get_messages", return_value=[msg]),
        patch.object(chat_mod, "render_reference_list") as mock_refs,
        patch.object(chat_mod.st, "chat_message") as mock_cm,
        patch.object(chat_mod.st, "markdown"),
        patch.object(chat_mod.st, "caption"),
    ):
        mock_cm.return_value.__enter__.return_value = MagicMock()
        mock_cm.return_value.__exit__.return_value = False
        chat_mod._render_unified_timeline("sid-cit-only")
    assert mock_refs.called
    assert mock_refs.call_args.args[0] == "cit-only-1"
