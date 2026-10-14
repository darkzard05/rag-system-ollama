"""CTX guard 재현 테스트 — 제거 1건당 전체 재포맷·재해시 결함 (red → green).

결함 위치: ``src/core/graph/_generate.py::_apply_ctx_guard``

- 예산 초과 시 while 루프가 문서를 1건 제거할 때마다 ``format_context``를 다시
  호출한다 → stable_id(MD5) 재해시 + 메타 포맷이 제거 횟수만큼 반복
  (O(제거횟수 × 문서수) 중복 연산).
- 정상(예산 내) 경로에서도 ``format_context``(사전)와 ``_estimate_ctx_tokens``
  (인라인 조립)로 동일 컨텍스트를 2회 조립한다.

기대 계약 (수정 후):
- ``format_context`` 호출 ≤ 2회 (사전 1회 + 트림 시 사후 1회).
- ``_doc_stable_id`` 해시 ≤ 4 × 문서수 (조각 1회 렌더 + 사전/사후 렌더).

이 테스트는 수정 전에는 실패(red)하고 수정 후 통과(green)해야 한다.
"""

from langchain_core.documents import Document

import core.graph._generate as ggen
from core.graph.graph_builder import _apply_ctx_guard

_NUM_CTX = 8192
_NUM_PREDICT = 2048
_N_DOCS = 20
_CONTENT_SIZE = 10_000


def _make_docs(n: int, content_size: int) -> list[Document]:
    """예산 초과를 확실히 유발하는 대용량 문서 세트 (회귀 테스트와 동일 레시피)."""
    payload = "가" * content_size
    return [
        Document(
            page_content=f"{payload}-{i}",
            metadata={"rerank_score": float(n - i), "page": i + 1},
        )
        for i in range(n)
    ]


def _install_budget(monkeypatch) -> None:
    monkeypatch.setattr(ggen, "OLLAMA_NUM_CTX", _NUM_CTX)
    monkeypatch.setattr(ggen, "OLLAMA_NUM_PREDICT", _NUM_PREDICT)


def test_repro_ctx_guard_format_context_call_bounded(monkeypatch) -> None:
    """트리밍 경로에서 format_context 호출이 제거 횟수에 비례하면 결함."""
    _install_budget(monkeypatch)

    real_format_context = ggen.format_context
    calls: list[int] = []

    def counting_format_context(docs: list[Document]) -> str:
        calls.append(len(docs))
        return real_format_context(docs)

    monkeypatch.setattr(ggen, "format_context", counting_format_context)

    docs = _make_docs(_N_DOCS, _CONTENT_SIZE)
    trimmed, _context, removed = _apply_ctx_guard(docs, "질문입니다")

    assert removed > 0, "재현 조건: 예산 초과로 트리밍이 발생해야 한다"
    assert len(trimmed) >= 2, "min-2-docs 불변식 유지"
    assert len(calls) <= 2, (
        f"format_context가 {len(calls)}회 호출됨 (제거 {removed}건). "
        f"결함: 제거 1건당 전체 재포맷. 기대: ≤2회 (사전 1 + 사후 1)"
    )


def test_repro_ctx_guard_stable_id_rehash_bounded(monkeypatch) -> None:
    """stable_id(MD5) 해시가 문서수×반복으로 재계산되면 결함."""
    _install_budget(monkeypatch)

    real_doc_stable_id = ggen._doc_stable_id
    hash_calls: list[int] = []

    def counting_doc_stable_id(doc) -> str:
        hash_calls.append(1)
        return real_doc_stable_id(doc)

    monkeypatch.setattr(ggen, "_doc_stable_id", counting_doc_stable_id)

    docs = _make_docs(_N_DOCS, _CONTENT_SIZE)
    trimmed, _context, removed = _apply_ctx_guard(docs, "질문입니다")

    assert removed > 0, "재현 조건: 예산 초과로 트리밍이 발생해야 한다"
    budget = 4 * _N_DOCS
    assert len(hash_calls) <= budget, (
        f"_doc_stable_id가 {len(hash_calls)}회 호출됨 (제거 {removed}건, 문서 {_N_DOCS}개). "
        f"결함: 재포맷마다 전체 재해시. 기대: ≤{budget}회 (조각 1 + 사전/사후 렌더)"
    )
