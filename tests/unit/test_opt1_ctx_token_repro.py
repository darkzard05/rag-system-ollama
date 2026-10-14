"""OPT-1 재현 테스트: CTX 가드 토큰 추정 최적화.

실패 조건 (수정 전):
- ``common.text_utils`` 에 사전 컴파일된 ASCII 패턴이 없음.
- ``_apply_ctx_guard`` 가 제거 1회마다 전체 프롬프트를 재조립+재카운트해
  ``count_tokens_rough`` 호출 횟수/스캔 문자량이 제거 횟수에 비례 (O(RxP)).

성공 조건 (수정 후): 위가 해소되고 트림 결과(경계·순서·context)는 동일.
"""

from langchain_core.documents import Document

import common.text_utils as text_utils
import core.graph._generate as ggen
from core.graph.graph_builder import _apply_ctx_guard


def _make_docs(n: int, content_size: int) -> list[Document]:
    payload = "가" * content_size
    return [
        Document(
            page_content=f"{payload}-{i}",
            metadata={"rerank_score": float(n - i), "page": i + 1},
        )
        for i in range(n)
    ]


def test_ascii_pattern_is_precompiled() -> None:
    """ASCII 판별 정규식이 모듈 상수로 사전 컴파일되어 있어야 한다."""
    assert hasattr(text_utils, "_RE_ASCII"), (
        "text_utils._RE_ASCII precompiled pattern missing: "
        "count_tokens_rough still recompiles the regex per call"
    )
    assert text_utils.count_tokens_rough("hello world") >= 1
    assert text_utils.count_tokens_rough("") == 0


def test_trim_avoids_repeated_full_prompt_recounts(monkeypatch) -> None:
    """트리밍이 전체 프롬프트 재카운트를 반복하지 않아야 한다.

    20개 대용량 문서 → 18개 제거. 수정 전에는 초기 1회 + 제거당 1회 = 19회
    전체 스캔이 발생하므로 호출 횟수 상한(8회)을 초과해 실패한다.
    """
    calls: list[int] = []
    real_count = ggen.count_tokens_rough

    def recording_count(text: str) -> int:
        calls.append(len(text))
        return real_count(text)

    monkeypatch.setattr(ggen, "count_tokens_rough", recording_count)

    docs = _make_docs(20, content_size=10_000)
    query = "질문입니다"
    full_len = len(
        ggen._ctx_prompt("".join(ggen._render_doc_block(d) for d in docs), query)
    )

    trimmed, context, removed = _apply_ctx_guard(docs, query)

    # 트림 결과 계약 (최적화 전후 동일해야 함)
    assert removed == 18
    assert len(trimmed) == 2
    assert context == "".join(ggen._render_doc_block(d) for d in trimmed)

    # 재카운트 예산: 전체 길이 스캔은 상수 회 이하로 제한
    full_scans = sum(1 for n in calls if n >= full_len)
    assert full_scans <= 2, (
        f"full-prompt recounts={full_scans} (calls={len(calls)}): "
        "trim loop still recounts the whole prompt per removal"
    )
    assert len(calls) <= 8, (
        f"count_tokens_rough calls={len(calls)}: expected binary-search "
        "probe budget, got linear per-removal recounts"
    )
