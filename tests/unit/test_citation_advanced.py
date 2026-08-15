from langchain_core.documents import Document

from common.utils import apply_tooltips_to_response


def test_citation_with_section_name():
    """섹션명이 포함된 인용구 처리 및 툴팁 매칭 테스트"""
    # Setup
    response = "CM3 모델은 새로운 목적 함수를 사용합니다 [섹션: 3 CM3, p.3]."
    docs = [
        Document(
            page_content="이것은 1페이지 내용입니다.",
            metadata={"page": 1, "current_section": "ABSTRACT"},
        ),
        Document(
            page_content="CM3의 핵심 원리는 마스킹입니다.",
            metadata={"page": 3, "current_section": "3 CM3"},
        ),
        Document(
            page_content="3페이지의 다른 섹션 내용입니다.",
            metadata={"page": 3, "current_section": "4 EXPERIMENTS"},
        ),
    ]

    # Execute
    result = apply_tooltips_to_response(response, docs)

    # Verify
    # 1. 인용구가 span으로 변환되었는지 확인
    assert '<span class="citation-highlight"' in result
    # 2. 섹션명이 일치하는 문서의 내용이 툴팁에 들어갔는지 확인
    assert 'title="CM3의 핵심 원리는 마스킹입니다...."' in result
    # 3. data-page 속성이 올바른지 확인
    assert 'data-page="3"' in result
    # 4. 원본 텍스트가 유지되는지 확인
    assert "[섹션: 3 CM3, p.3]" in result


def test_citation_normalization():
    """다양한 인용 패턴(DOC, page, 괄호 등) 지원 여부 테스트"""
    docs = [Document(page_content="내용", metadata={"page": 5})]

    # 유형 1: [DOC 1, p.5]
    res1 = apply_tooltips_to_response("참조 [DOC 1, p.5]", docs)
    assert 'data-page="5"' in res1

    # 유형 2: (p.5)
    res2 = apply_tooltips_to_response("참조 (p.5)", docs)
    assert 'data-page="5"' in res2

    # 유형 3: [5]
    res3 = apply_tooltips_to_response("참조 [5]", docs)
    assert 'data-page="5"' in res3


def test_tooltip_escaping():
    """툴팁 내용의 HTML 이스케이프 및 줄바꿈 처리 테스트"""
    docs = [Document(page_content='그는 "안녕"이라고\n말했다.', metadata={"page": 10})]

    result = apply_tooltips_to_response("참조 [p.10]", docs)

    # 따옴표(&quot;)와 줄바꿈(공백) 처리 확인
    assert "&quot;안녕&quot;" in result
    assert "\n" not in result  # 줄바꿈이 제거되어야 함


def test_citation_no_false_positive_done():
    """'[done]' (no digit) must NOT become a citation span."""
    result = apply_tooltips_to_response(
        "Task [done] completed",
        [Document(page_content="x", metadata={"page": 1})],
    )
    assert "citation-highlight" not in result


def test_citation_no_false_positive_section():
    """'[Section A]' (no digit) must NOT become a citation span."""
    result = apply_tooltips_to_response(
        "See [Section A] now",
        [Document(page_content="y", metadata={"page": 2})],
    )
    assert "citation-highlight" not in result


def test_citation_no_false_positive_note():
    """'[Note]' (no digit) must NOT become a citation span."""
    result = apply_tooltips_to_response(
        "Note [Note] here",
        [Document(page_content="z", metadata={"page": 3})],
    )
    assert "citation-highlight" not in result


def test_citation_bare_page_still_spans():
    """'[5]' (digit) MUST still become a citation span with data-page=5."""
    result = apply_tooltips_to_response(
        "Ref [5] ok",
        [Document(page_content="w", metadata={"page": 5})],
    )
    assert "citation-highlight" in result
    assert 'data-page="5"' in result


def test_citation_date_like_does_not_span():
    """'[2024-01-01]' is NOT treated as a citation span.

    Although it leads with a digit, the `_RE_CITATION_BLOCK` pattern requires the
    digits to be followed by specific citation trailing syntax; an ISO date like
    '2024-01-01' does not match, so the text is left untouched. This documents the
    OBSERVED behavior (verified empirically) so future regex regressions are caught.
    """
    result = apply_tooltips_to_response(
        "On [2024-01-01] we met",
        [Document(page_content="d", metadata={"page": 9})],
    )
    assert "citation-highlight" not in result
