"""T11 — `_filter_min_section_len` 초단문 섹션 필터 단위 테스트.

정책:
- 50자 이상 문서는 항상 유지한다.
- 50자 미만이라도 숫자/단답("671B", 표 셀)이나 의미 있는 단어("CM3")를
  포함하면 유지한다 (정답 단편 보존).
- 헤더("# ..."), 번호("3.1"), 기호만 있는 저정보 파편은 드롭한다.
- 전부 저정보 파편이면 최고 rerank_score 1건을 유지한다.
"""

from langchain_core.documents import Document

from core.graph.graph_builder import _filter_min_section_len


def _doc(text: str, score: float = 0.0) -> Document:
    return Document(
        page_content=text,
        metadata={"source": "s.pdf", "page": 1, "rerank_score": score},
    )


def test_drops_low_info_header_fragment():
    """마크다운 헤더 파편은 컨텍스트에서 제거된다."""
    long_doc = _doc("x" * 50)
    header_doc = _doc("## 3. 실험")

    result = _filter_min_section_len([long_doc, header_doc])

    assert result == [long_doc]


def test_drops_numbering_only_fragment():
    """번호/불릿만 있는 파편은 제거된다."""
    result = _filter_min_section_len([_doc("3.1"), _doc("---"), _doc("B" * 50)])

    assert [d.page_content for d in result] == ["B" * 50]


def test_keeps_short_numeric_answer():
    """수치 단답("671B")은 50자 미만이어도 유지된다 (정답 소실 방지)."""
    numeric = _doc("671B")
    long_doc = _doc("x" * 60)

    result = _filter_min_section_len([long_doc, numeric])

    assert numeric in result
    assert long_doc in result


def test_keeps_short_table_cell():
    """표 셀처럼 수치가 든 짧은 청크는 유지된다."""
    cell = _doc("| 671B | 1.76T |")

    result = _filter_min_section_len([cell, _doc("y" * 50)])

    assert cell in result


def test_keeps_short_model_name():
    """숫자가 없어도 의미 있는 단어(CM3, MoE)는 유지된다."""
    name = _doc("CM3 모델")

    result = _filter_min_section_len([name, _doc("z" * 50)])

    assert name in result


def test_boundary_49_low_info_dropped_50_kept():
    """경계: 저정보 49자는 드롭, 50자는 유지된다."""
    result = _filter_min_section_len([_doc("-" * 49), _doc("y" * 50)])

    assert [d.page_content for d in result] == ["y" * 50]


def test_keeps_normal_sections():
    """50자 이상 정상 문서들은 모두 유지된다."""
    docs = [_doc("a" * 50), _doc("b" * 120), _doc("c" * 51)]

    result = _filter_min_section_len(docs)

    assert [d.page_content for d in result] == [d.page_content for d in docs]


def test_empty_list_is_safe():
    """빈 리스트는 그대로 빈 리스트를 반환한다."""
    assert _filter_min_section_len([]) == []


def test_all_low_info_keeps_highest_score_one():
    """전부 저정보 파편이면 최고 rerank_score 1건만 유지한다."""
    docs = [_doc("## 1", score=0.2), _doc("---", score=0.9), _doc("2.", score=0.5)]

    result = _filter_min_section_len(docs)

    assert result == [docs[1]]
