import ui.strings as strings_module
from ui.components.chat import _final_status_line, _live_status_line
from ui.strings import set_lang

_SEARCH_AND_GENERATE = [
    "관련 지식 검색 중...",
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
        unknown = _live_status_line("zzz-no-such-phase-qqq", 5.0, 3)
        set_lang("ko")
        known = _live_status_line(
            "질문을 분석하고 관련 지식을 검색하는 중입니다... ▍", 5.0, 3
        )
    finally:
        set_lang(original)

    assert unknown == "▸ Working... · Citations 3 · 5s ▍"
    assert known == "▸ 문서 검색 · 인용 3개 · 5s ▍"


def test_live_status_uses_matching_phase_for_each_status() -> None:
    """상태 문구가 실제 위상에 맞는 라벨로 이어진다."""
    original = strings_module.LANG
    cases = {
        "관련 지식 검색 중...": "문서 검색",
        "관련 문서 3개를 찾았습니다.": "문서 검색",
        "문서 관련성 검증 중...": "증거 수집",
        "답변 설계 및 생성 중...": "답변 생성",
        "인용 정리 중...": "답변 생성",
    }
    try:
        set_lang("ko")
        lines = {status: _live_status_line(status, 3.0, 0) for status in cases}
    finally:
        set_lang(original)

    for status, label in cases.items():
        assert f"▸ {label}" in lines[status], (status, lines[status])


def test_live_status_recognizes_initial_search_in_english() -> None:
    """영문 초기 상태도 검색 위상으로 표시된다."""
    original = strings_module.LANG
    try:
        set_lang("en")
        line = _live_status_line(
            "Analyzing the question and searching related knowledge... ▍", 3.0, 0
        )
    finally:
        set_lang(original)

    assert "▸ Document Search" in line


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
