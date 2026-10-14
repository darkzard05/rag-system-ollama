"""CTX guard 성능·불변식 단위 테스트.

- (a) 대용량 문서 20개 → 트리밍 발생 시 _render_doc_block 호출은 정확히 초기 문서 수(20회)만큼만 수행 (O(D) 렌더)
- (b) 트리밍 후 최소 2문서 보존 (min-2-docs invariant)
- (c) 예산 내 문서 20개 → 트리밍 없음, _render_doc_block 호출 = 20회, len == 20
"""

from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

import core.graph._generate as ggen
from core.graph.graph_builder import _apply_ctx_guard


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

    monkeypatch.setattr(ggen, "count_tokens_rough", fake_count)
    return fake_count


def test_ctx_guard_trims_with_few_format_calls(patch_tokens, monkeypatch):
    """예산 초과 시 트리밍되나 _render_doc_block 호출은 초기 문서 수(20회)를 초과하지 않음."""
    render_count = 0
    real_render = ggen._render_doc_block

    def counting_render(d: Document) -> str:
        nonlocal render_count
        render_count += 1
        return real_render(d)

    monkeypatch.setattr(ggen, "_render_doc_block", counting_render)

    docs = _make_docs(20, content_size=10_000)
    query = "질문입니다"

    trimmed, context, removed = _apply_ctx_guard(docs, query)

    # 예산 초과로 트리밍이 발생했다
    assert removed > 0
    # min-2-docs 보존
    assert len(trimmed) >= 2
    # O(D) 렌더 계약: 트리밍 제거마다 재렌더하지 않고 초기 20회만 렌더
    assert render_count == 20
    # 내림차순 정렬 보존: 첫 문서가 가장 높은 rerank_score
    assert trimmed[0].metadata["rerank_score"] >= trimmed[-1].metadata["rerank_score"]
    assert context == "".join(real_render(d) for d in trimmed)


def test_ctx_guard_no_trim_when_fits(monkeypatch):
    """예산 내 문서는 트리밍되지 않고 _render_doc_block 호출 = 20회."""
    render_count = 0
    real_render = ggen._render_doc_block

    def counting_render(d: Document) -> str:
        nonlocal render_count
        render_count += 1
        return real_render(d)

    monkeypatch.setattr(ggen, "_render_doc_block", counting_render)

    # 작은 문서: 예산 내에 들어옴
    docs = _make_docs(20, content_size=10)

    trimmed, context, removed = _apply_ctx_guard(docs, "q")

    assert removed == 0
    assert len(trimmed) == 20
    assert render_count == 20
    assert context == "".join(real_render(d) for d in trimmed)


def test_ctx_guard_min_two_docs_always(monkeypatch):
    """매우 큰 문서라도 최소 2문서는 유지된다."""
    monkeypatch.setattr(ggen, "count_tokens_rough", lambda t: 1_000_000)
    docs = _make_docs(20, content_size=50_000)

    trimmed, _, removed = _apply_ctx_guard(docs, "q")

    assert removed == 18
    assert len(trimmed) == 2
