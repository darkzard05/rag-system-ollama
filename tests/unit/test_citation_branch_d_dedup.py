"""Branch-D regression: same-text-different-page chunks -> exactly 1 citation.

Wave-0 evidence (reports/measure_citation_20260926_142832.json): branches H
(identical bytes) and E (extraction dup) ruled out; the renderer
(chat_references.py grouping keyed only on page) emits p.3/p.10/p.11/p.14
near-identical refs by construction.

UX choice (ONE): exactly ONE citation keeping the minimum page, with the
most-specific section preserved in the label (no silent collapse to "").
"""

from typing import Any
from unittest.mock import MagicMock, patch

from ui.components.chat_references import _render_references_content


def _doc(page: int, section: str, text: str, doc_id: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"page": page, "current_section": section, "doc_id": doc_id}
    doc.page_content = text
    return doc


def test_same_text_different_pages_collapses_to_single_citation() -> None:
    """Identical chunk text on pages 3/10/11/14 must render one citation."""
    body = "동일한 인용 본문 텍스트 " * 20
    sections = ["제3장 실험 결과 상세 분석", "제4장 고찰", "일반 본문", "문서 본문"]
    pages = [3, 10, 11, 14]
    documents: list[Any] = [
        _doc(p, s, body, f"chunk_{p}") for p, s in zip(pages, sections, strict=True)
    ]
    with patch("ui.components.chat_references.st") as mock_st:
        rendered = _render_references_content("msg_branch_d", documents)
    assert rendered is True
    buttons = mock_st.button.call_args_list
    assert len(buttons) == 1
    label = str(buttons[0].args[0])
    assert "p.3" in label
    assert "제3장 실험 결과 상세 분석" in label
