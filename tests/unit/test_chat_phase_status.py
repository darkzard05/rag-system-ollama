import ui.strings as strings_module
from ui.components.chat import _final_status_line, _live_status_line
from ui.strings import set_lang

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


def test_live_status_line_uses_one_phase_and_short_fallback() -> None:
    original = strings_module.LANG
    try:
        set_lang("en")
        unknown = _live_status_line(
            "Analyzing the question and searching related knowledge... ▍", 5.0, 3
        )
        set_lang("ko")
        known = _live_status_line(
            "질문을 분석하고 관련 지식을 검색하는 중입니다... ▍", 5.0, 3
        )
    finally:
        set_lang(original)

    assert unknown == "▸ Working... · Citations 3 · 5s ▍"
    assert known == "▸ 문서 검색 · 인용 3개 · 5s ▍"


def test_final_status_line_uses_short_phase_labels() -> None:
    original = strings_module.LANG
    try:
        set_lang("ko")
        line = _final_status_line(
            _SEARCH_AND_GENERATE,
            [{"page": 1}] * 3,
            {"total_time": 12.3},
        )
    finally:
        set_lang(original)

    assert line == "✓ 검색 · ✓ 증거 · ✓ 생성 · 인용 3개 · 12.3s"
