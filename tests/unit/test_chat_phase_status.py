"""The final chat status line must show every completed answer phase."""

from ui.components.chat import _final_status_line

_SEARCH_AND_GENERATE = [
    "관련 지식 검색 중...",
    "문서 관련성 검증 중...",
    "답변 설계 및 생성 중...",
]


def test_final_status_marks_evidence_phase_when_citations_exist() -> None:
    line = _final_status_line(
        _SEARCH_AND_GENERATE,
        [{"page": 6, "doc_id": "doc-1"}],
        {"total_time": 31.6},
    )

    assert line.count("✓") == 3, line


def test_final_status_leaves_evidence_phase_unmarked_without_citations() -> None:
    line = _final_status_line(_SEARCH_AND_GENERATE, [], {"total_time": 31.6})

    assert line.count("✓") == 2, line
