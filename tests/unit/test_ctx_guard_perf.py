"""CTX guard 성능·불변식 단위 테스트.

- (a) 대용량 문서 20개 → 트리밍 발생 시 format_context 호출 ≤ 6회 (O(n^2) 재포맷 제거)
- (b) 트리밍 후 최소 2문서 보존 (min-2-docs invariant)
- (c) 예산 내 문서 20개 → 트리밍 없음, format_context 호출 = 1(사전만), len == 20
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import core.graph_builder as gb
from core.graph_builder import _apply_ctx_guard


def _make_docs(n: int, content_size: int) -> list[Document]:
    payload = "가" * content_size
    docs = []
    for i in range(n):
        # rerank_score 내림차순 정렬 검증을 위해 점수 분산 부여
        docs.append(
            Document(
                page_content=f"{payload}-{i}",
                metadata={"rerank_score": float(n - i), "page": i + 1},
            )
        )
    return docs


@pytest.fixture
def patch_tokens(monkeypatch):
    """count_tokens_rough를 입력 길이 기반 결정론 상수로 교체 (len//4 + 1)."""

    def fake_count(text: str) -> int:
        return len(text) // 4 + 1

    monkeypatch.setattr(gb, "count_tokens_rough", fake_count)
    return fake_count


def test_ctx_guard_trims_with_few_format_calls(patch_tokens, monkeypatch):
    """예산 초과 시 트리밍되나 format_context 호출은 최대 6회."""
    fmt = MagicMock(side_effect=lambda docs: f"<ctx:{len(docs)}>")
    monkeypatch.setattr(gb, "format_context", fmt)

    docs = _make_docs(20, content_size=10_000)
    query = "질문입니다"

    trimmed, context, removed = _apply_ctx_guard(docs, query)

    # 예산 초과로 트리밍이 발생했다
    assert removed > 0
    # min-2-docs 보존
    assert len(trimmed) >= 2
    # O(n^2) 재포맷 제거: 사전 1회 + 사후 1회 ≈ 2회 (허용 상한 6)
    assert fmt.call_count <= 6
    # 내림차순 정렬 보존: 첫 문서가 가장 높은 rerank_score
    assert trimmed[0].metadata["rerank_score"] >= trimmed[-1].metadata["rerank_score"]
    assert context == f"<ctx:{len(trimmed)}>"


def test_ctx_guard_no_trim_when_fits(monkeypatch):
    """예산 내 문서는 트리밍되지 않고 format_context 호출 = 2회."""
    fmt = MagicMock(side_effect=lambda docs: f"<ctx:{len(docs)}>")
    monkeypatch.setattr(gb, "format_context", fmt)

    # 작은 문서: 예산 내에 들어옴
    docs = _make_docs(20, content_size=10)

    trimmed, context, removed = _apply_ctx_guard(docs, "q")

    assert removed == 0
    assert len(trimmed) == 20
    # 트리밍 분기 미진입 → 사전 포맷 1회만 호출 (사후 재포맷은 트림 시점에만)
    assert fmt.call_count == 1
    assert context == "<ctx:20>"


def test_ctx_guard_min_two_docs_always(monkeypatch):
    """매우 큰 문서라도 최소 2문서는 유지된다."""
    fmt = MagicMock(side_effect=lambda docs: f"<ctx:{len(docs)}>")
    monkeypatch.setattr(gb, "format_context", fmt)
    monkeypatch.setattr(gb, "count_tokens_rough", lambda t: 1_000_000)
    docs = _make_docs(20, content_size=50_000)

    trimmed, _, removed = _apply_ctx_guard(docs, "q")

    assert removed == 18
    assert len(trimmed) == 2
