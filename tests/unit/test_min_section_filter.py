"""T11 — `_filter_min_section_len` 초단문 섹션 필터 단위 테스트.

대상: `src/core/graph_builder.py`의 `retrieve_and_rerank`가 `_merge_adjacent_chunks`
이후 50자 미만 섹션을 최종 컨텍스트에서 제거하는 동작.

케이스:
- 43자 문서 드롭 (실측: `[GENERATE] 문서 0 길이: 43`)
- 경계: 49자 드롭 / 50자 유지
- 정상(50자 이상) 문서 유지
- 빈 리스트 안전
- 전체가 50자 미만인 극단 케이스: 정확히 1개 유지 (가드)
"""

from langchain_core.documents import Document

from core.graph_builder import _filter_min_section_len


def _doc(text: str) -> Document:
    return Document(page_content=text, metadata={"source": "s.pdf", "page": 1})


def test_drops_43_char_section():
    """43자 문서는 제거된다 (실측 사례: [GENERATE] 문서 0 길이: 43)."""
    long_doc = _doc("x" * 50)
    short_doc = _doc("x" * 43)

    result = _filter_min_section_len([long_doc, short_doc])

    assert len(result) == 1
    assert result[0] is long_doc


def test_boundary_49_dropped_50_kept():
    """경계: 49자는 드롭, 50자는 유지된다."""
    result = _filter_min_section_len([_doc("x" * 49), _doc("y" * 50)])

    assert [d.page_content for d in result] == ["y" * 50]


def test_keeps_normal_sections():
    """50자 이상 정상 문서들은 모두 유지된다."""
    docs = [_doc("a" * 50), _doc("b" * 120), _doc("c" * 51)]

    result = _filter_min_section_len(docs)

    assert [d.page_content for d in result] == [d.page_content for d in docs]


def test_empty_list_is_safe():
    """빈 리스트는 그대로 빈 리스트를 반환한다."""
    assert _filter_min_section_len([]) == []


def test_all_short_keeps_exactly_one():
    """전체가 50자 미만이면 가드가 정확히 1개만 유지한다."""
    docs = [_doc("s" * 10), _doc("t" * 20), _doc("u" * 30)]

    result = _filter_min_section_len(docs)

    assert len(result) == 1
    assert result[0] is docs[0]
