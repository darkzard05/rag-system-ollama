"""
인용 툴팁 회귀 테스트 (P0 + P2: citation-display-improvements).

- BASELINE: 기존 [p.3] 페이지 인용 동작 보존 (data-page=3).
- P2 BEHAVIOR: [doc:<stable_id>]는 문서의 doc_id 메타데이터(또는 content 해시)
  로 매핑되어야 하며, 위치 인덱스/페이지 번호로 오인되면 안 됨.
  (P0의 위치 기반 [doc:1] 동작은 P2에서 안정 ID 매핑으로 교체됨.)
"""

from dataclasses import dataclass
from typing import Any

from common.utils import apply_tooltips_to_response


@dataclass
class FakeDoc:
    """경량 가짜 문서 — real Document 미사용."""

    page_content: str
    metadata: dict[str, Any]


def _build_docs() -> list[FakeDoc]:
    # doc_id 메타데이터로 stable id를 명시. page는 오인 검증용으로
    # 인덱스/해시와 무관하게 섞어둔다 (documents[1].page == 7).
    return [
        FakeDoc(
            page_content="doc0 content", metadata={"page": 10, "doc_id": "chunk_A"}
        ),
        FakeDoc(
            page_content="doc1 content page7",
            metadata={"page": 7, "doc_id": "chunk_B"},
        ),
        FakeDoc(
            page_content="doc2 content page3",
            metadata={"page": 3, "doc_id": "chunk_C"},
        ),
    ]


def test_baseline_page_citation_preserved():
    """[p.3] 페이지 인용이 기존 동작대로 data-page=3 스팬으로 변환되는지."""
    docs = _build_docs()
    result = apply_tooltips_to_response("참조 [p.3] 입니다.", docs)

    assert '<span class="citation-highlight"' in result
    assert 'data-page="3"' in result
    assert "[p.3]" in result


def test_doc_citation_maps_by_stable_id_not_by_page():
    """[doc:chunk_B] 는 doc_id=='chunk_B' 문서로 매핑 (page 7 로 오인 금지)."""
    docs = _build_docs()
    result = apply_tooltips_to_response("이것은 [doc:chunk_B] 에 나옵니다.", docs)

    # 1. doc-id는 stable id 'chunk_B' 이어야 함.
    assert 'data-doc-id="chunk_B"' in result
    # 2. 페이지 번호(7 등)로 오인되지 않아야 함.
    assert 'data-page="7"' not in result
    # 3. doc_id=='chunk_B' 내용(page7)이 툴팁에 들어가야 함.
    assert "doc1 content page7" in result
    # 4. 원본 [doc:chunk_B] 태그 텍스트가 유지되어야 함.
    assert "[doc:chunk_B]" in result


def test_doc_citation_unknown_id_returns_raw():
    """알 수 없는 [doc:unknown] 은 원본 그대로 반환 (죽은 인용 방지)."""
    docs = _build_docs()
    result = apply_tooltips_to_response("없는 인용 [doc:unknown] 입니다.", docs)

    assert "[doc:unknown]" in result
    assert 'data-doc-id="unknown"' not in result
    assert '<span class="citation-highlight"' not in result


def test_page_citation_still_resolves_by_page_with_doc_present():
    """[p.3] 은 여전히 page==3 인 문서(documents[2])로 해석되어야 함."""
    docs = _build_docs()
    result = apply_tooltips_to_response("페이지 인용 [p.3] 과 [doc:chunk_B].", docs)

    assert 'data-doc-id="chunk_B"' in result
    # page 3 인 documents[2] 의 내용이 툴팁에 들어가야 함.
    assert "doc2 content page3" in result
    assert 'data-page="3"' in result
