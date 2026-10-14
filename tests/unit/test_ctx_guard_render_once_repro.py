"""TDD Repro: _apply_ctx_guard renders doc blocks exactly once and avoids format_context duplication.

- Defect: _apply_ctx_guard calls format_context (which re-renders + re-hashes all docs)
  and also builds local `blocks`, resulting in redundant O(2D) or O(3D) renders per call.
- Contract: _render_doc_block must be called exactly D times (once per unique doc),
  and _deduplicate_docs exactly once, on both trim and no-trim paths.
"""

from unittest.mock import patch
import pytest
from langchain_core.documents import Document

import core.graph._generate as ggen
from core.graph._generate import _apply_ctx_guard, _render_doc_block


def _make_test_docs(n: int = 20) -> list[Document]:
    return [
        Document(
            page_content=f"Document content number {i} for testing render-once guard.",
            metadata={
                "rerank_score": float(i) / n,
                "page": i + 1,
                "current_section": "테스트",
            },
        )
        for i in range(n)
    ]


def test_apply_ctx_guard_renders_docs_exactly_once() -> None:
    """예산 내 문서 20개 경로에서 _render_doc_block이 정확히 20회(D회)만 호출되는지 검증 (중복 렌더 방지)."""
    docs = _make_test_docs(20)

    render_calls = 0
    real_render = _render_doc_block

    def counting_render(d: Document) -> str:
        nonlocal render_calls
        render_calls += 1
        return real_render(d)

    dedup_calls = 0
    real_dedup = ggen._deduplicate_docs

    def counting_dedup(d_list: list[Document]) -> list[Document]:
        nonlocal dedup_calls
        dedup_calls += 1
        return real_dedup(d_list)

    with (
        patch("core.graph._generate._render_doc_block", side_effect=counting_render),
        patch("core.graph._generate._deduplicate_docs", side_effect=counting_dedup),
    ):
        trimmed, context, removed = _apply_ctx_guard(docs, "질문입니다")

    assert len(trimmed) == 20
    assert removed == 0
    assert dedup_calls == 1, f"_deduplicate_docs 호출 횟수 부적절: {dedup_calls}"
    assert render_calls == 20, (
        f"중복 렌더 결함: _render_doc_block이 {render_calls}회 호출됨 (기대: 문서 수 20회)"
    )
    # 컨텍스트가 정상 조인되었는지 검증
    expected_context = "".join(real_render(d) for d in trimmed)
    assert context == expected_context
