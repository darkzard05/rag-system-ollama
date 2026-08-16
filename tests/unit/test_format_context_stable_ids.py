"""
[P2] format_context 안정 ID(stable id) 인용 회귀 테스트.

- format_context가 [doc:<stable_id>] 토큰을 내보내는지.
- 토큰 <-> chunk_id 매핑이 rerank 재정렬(문서 순서 변경) 하에서 불변인지.
- verify 노드의 _validate_cited_doc_ids 가 안정 id를 멤버십 검사하여
  알려진 id는 통과, 모르는(환각) id는 거부하는지.
"""

from langchain_core.documents import Document

from core.graph_builder import _doc_stable_id, _validate_cited_doc_ids, format_context


def _make_doc(stable_id: str, content: str, page: int = 1) -> Document:
    return Document(page_content=content, metadata={"doc_id": stable_id, "page": page})


def test_format_context_emits_stable_id_tokens():
    """format_context가 [doc:<stable_id>] 토큰을 내보내야 함."""
    a = _make_doc("chunk_A", "alpha body")
    b = _make_doc("chunk_B", "beta body")
    c = _make_doc("chunk_C", "gamma body")

    ctx = format_context([a, b, c])

    assert "[doc:chunk_A]" in ctx
    assert "[doc:chunk_B]" in ctx
    assert "[doc:chunk_C]" in ctx


def test_token_mapping_invariant_under_reorder():
    """문서 순서를 바꿔도 각 chunk id는 동일 토큰에 매핑되어야 함 (불변)."""
    a = _make_doc("chunk_A", "alpha body")
    b = _make_doc("chunk_B", "beta body")
    c = _make_doc("chunk_C", "gamma body")

    ctx_abc = format_context([a, b, c])
    # rerank 재정렬: B, A, C
    ctx_bac = format_context([b, a, c])

    # 각 청크 내용이 여전히 자신의 stable_id 토큰 바로 뒤에 와야 함.
    def _token_for(ctx: str, content: str) -> str:
        idx = ctx.index(content)
        head = ctx[:idx]
        # 마지막 [doc:...] 토큰 추출
        start = head.rfind("[doc:")
        end = head.index("]", start)
        return head[start + len("[doc:") : end]

    assert (
        _token_for(ctx_abc, "alpha body")
        == _token_for(ctx_bac, "alpha body")
        == "chunk_A"
    )
    assert (
        _token_for(ctx_abc, "beta body")
        == _token_for(ctx_bac, "beta body")
        == "chunk_B"
    )
    assert (
        _token_for(ctx_abc, "gamma body")
        == _token_for(ctx_bac, "gamma body")
        == "chunk_C"
    )


def test_validate_accepts_known_stable_ids():
    """답변이 알려진 stable id를 인용하면 검증 통과(빈 리스트)."""
    a = _make_doc("chunk_A", "alpha body")
    b = _make_doc("chunk_B", "beta body")

    answer = "alpha는 [doc:chunk_A] 이고 beta는 [doc:chunk_B] 다."
    invalid = _validate_cited_doc_ids(answer, [a, b])

    assert invalid == []


def test_validate_rejects_unknown_stable_id():
    """답변이 모르는(환각) stable id를 인용하면 거부."""
    a = _make_doc("chunk_A", "alpha body")
    b = _make_doc("chunk_B", "beta body")

    answer = "이건 [doc:chunk_Z] 에 있다고 해요."  # docs에 없는 id
    invalid = _validate_cited_doc_ids(answer, [a, b])

    assert invalid == ["chunk_Z"]


def test_validate_rejects_when_no_docs_but_citation_present():
    """docs가 없는데 인용이 남아있으면 무효."""
    answer = "그건 [doc:chunk_A] 다."
    invalid = _validate_cited_doc_ids(answer, [])
    assert invalid == ["chunk_A"]


def test_doc_stable_id_helper_is_canonical():
    """_doc_stable_id가 doc_id 메타데이터를 우선 사용."""
    d = _make_doc("chunk_X", "body")
    assert _doc_stable_id(d) == "chunk_X"
