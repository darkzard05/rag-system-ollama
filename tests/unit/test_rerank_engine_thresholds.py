"""리랭킹 엔진별 임계값·폴백·회로 차단기 유닛 테스트 (R3b-02 / R3b-03).

- (a) semantic(bi-encoder, 코사인) 엔진에서 별도 임계값 `min_score_to_skip_semantic`으로
      short-circuit 발동 — FlashRank용 0.85와 분리됨
- (b) FlashRank(시그모이드) 엔진은 semantic 임계값이 아니라 0.85로 판정 (분기 누수 방지)
- (c) 6자 미만 쿼리는 리랭킹을 생략하지만 `rerank_score`(RRF 집계 점수)를 기록
- (d) get_or_build 실패 네거티브 캐시 — build_fn 연속 실패 N회 후 회로 차단(즉시 실패),
      TTL 경과 후 재시도 허용
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

import core.async_reranker as ar
from common.config import GRADING_CONFIG
from core.graph_builder import grade_documents, retrieve_and_rerank


@pytest.fixture(autouse=True)
def _reset_reranker_state():
    """엔진 추적 전역 상태를 테스트 간 초기화합니다."""
    ar._rerank_engine_active = "flashrank"
    ar._async_reranker = None
    ar._semantic_fallback_reranker = None
    yield
    ar._rerank_engine_active = "flashrank"
    ar._async_reranker = None
    ar._semantic_fallback_reranker = None


def _semantic_threshold() -> float:
    return float(GRADING_CONFIG.get("min_score_to_skip_semantic", 0.60))


def _flashrank_threshold() -> float:
    return float(GRADING_CONFIG.get("min_score_to_skip", 0.85))


def _grade_state(score: float) -> dict:
    return {
        "input": "DeepSeek-R1 성능은 어떠한가요?",
        "relevant_docs": [
            Document(page_content="관련 문서 내용", metadata={"rerank_score": score}),
        ],
        "retry_count": 0,
        "is_cached": False,
        "intent": "rag",
    }


@pytest.mark.asyncio
async def test_semantic_engine_fires_short_circuit_at_cosine_threshold():
    """(a) semantic 엔진+코사인 임계값(0.60)에 도달하면 LLM grade 생략(short-circuit)."""
    cosine_threshold = _semantic_threshold()
    # 코사인 스케일(실측 0.32~0.57) 상한을 넘는 점수 → FlashRank 0.85엔 못 미치지만
    # semantic 전용 임계값으론 충분해야 한다.
    score = cosine_threshold + 0.02
    assert score < _flashrank_threshold()  # 두 임계값 사이의 점수임을 보장

    llm = MagicMock()
    config = {"configurable": {"llm": llm}}

    with patch.object(ar, "_rerank_engine_active", "semantic"):
        result = await grade_documents(_grade_state(score), config, writer=None)

    assert result == {"intent": "generate", "route": "generate"}
    llm.ainvoke.assert_not_called()
    llm.bind.assert_not_called()


@pytest.mark.asyncio
async def test_flashrank_engine_does_not_fire_below_sigmoid_threshold():
    """(b) 동일한 점수라도 FlashRank 엔진(0.85)에서는 short-circuit이 발동하지 않는다."""
    score = _semantic_threshold() + 0.02  # semantic 임계값 이상이지만 0.85 미만
    assert score < _flashrank_threshold()

    mock_llm = MagicMock()
    json_llm = AsyncMock()
    mock_llm.bind.return_value = json_llm
    json_llm.ainvoke.return_value = MagicMock(
        content='{"action": "generate", "is_relevant": true, '
        '"relevant_entities": ["A"], "reason": "ok", "optimized_query": null}'
    )
    config = {"configurable": {"llm": mock_llm}}

    with (
        patch.object(ar, "_rerank_engine_active", "flashrank"),
        patch("core.graph_builder.adispatch_custom_event", new_callable=AsyncMock),
    ):
        result = await grade_documents(_grade_state(score), config, writer=None)

    assert result == {"intent": "generate", "route": "generate"}
    # FlashRank 임계값(0.85) 미만이므로 LLM 검증 경로가 실제 실행되어야 합니다.
    mock_llm.bind.assert_called_once()


@pytest.mark.asyncio
async def test_short_query_path_records_rrf_rerank_score():
    """(c) 6자 미만 쿼리는 리랭킹을 생략하지만 rerank_score(RRF)를 기록해야 한다."""
    docs = [
        Document(
            page_content=f"문서 {i}",
            metadata={
                "score": 1.0,
                "source": "s1",
                "page": 1,
                "chunk_index": i,
                "current_section": "sec",
            },
        )
        for i in range(3)
    ]
    bm25 = AsyncMock()
    bm25.ainvoke.return_value = docs
    faiss = AsyncMock()
    faiss.ainvoke.return_value = []
    llm = MagicMock()

    state = {"input": "짧은", "search_queries": [], "retry_count": 0}
    config = {
        "configurable": {
            "llm": llm,
            "bm25_retriever": bm25,
            "faiss_retriever": faiss,
            "session_id": "short-query-rrf",
        }
    }

    reranker = AsyncMock()
    reranker.rerank = AsyncMock(return_value=([], []))
    with patch(
        "core.async_reranker.get_async_reranker",
        new_callable=AsyncMock,
        return_value=reranker,
    ):
        result = await retrieve_and_rerank(state, config, writer=None)

    reranker.rerank.assert_not_awaited()  # 리랭킹은 생략되어야 함
    merged = result["relevant_docs"]
    assert merged
    # 리랭킹 생략 경로에도 rerank_score가 기록되어야 short-circuit이 0.0으로만 평가되지 않는다.
    assert all("rerank_score" in d.metadata for d in merged)
    assert max(float(d.metadata["rerank_score"]) for d in merged) > 0.0


def test_get_or_build_circuit_breaks_after_consecutive_failures():
    """(d) build_fn 연속 실패 N회 후 회로 차단 — 추가 호출은 build_fn 없이 즉시 실패."""
    from core.resource_manager import ResourceCoordinator

    coordinator = ResourceCoordinator()
    coordinator.reset()

    class _FakePool:
        name = "fake"

        def __init__(self):
            self._items: dict[str, object] = {}

        def get(self, key: str):
            return self._items.get(key)

        async def put(self, key: str, value: object) -> None:
            self._items[key] = value

    pool = _FakePool()
    calls = {"n": 0}

    def _failing_build():
        calls["n"] += 1
        raise RuntimeError("ONNX 모델 다운로드 실패")

    # 첫 3회는 build_fn이 호출되며 예외가 전파된다.
    for _ in range(3):
        with pytest.raises(RuntimeError):
            _asyncio_run(coordinator.get_or_build(pool, "flashrank_x", _failing_build))
    assert calls["n"] == 3

    # 4회째: 회로 차단 — build_fn을 호출하지 않고 즉시 ResourceBuildError(circuit_open).
    from common.exceptions import ResourceBuildError

    with pytest.raises(ResourceBuildError) as exc_info:
        _asyncio_run(coordinator.get_or_build(pool, "flashrank_x", _failing_build))
    assert exc_info.value.details.get("reason") == "circuit_open"
    assert calls["n"] == 3  # build_fn 재호출 없음


def test_get_or_build_circuit_expires_after_ttl(monkeypatch):
    """(d-2) TTL 경과 후에는 회로가 풀려 재시도가 허용된다 (성공 시 카운터 리셋)."""
    from core.resource_manager import ResourceCoordinator

    monkeypatch.setattr("core.resource_manager._BUILD_CIRCUIT_TTL_SECONDS", 0.05)

    coordinator = ResourceCoordinator()
    coordinator.reset()

    class _FakePool:
        name = "fake"

        def __init__(self):
            self._items: dict[str, object] = {}

        def get(self, key: str):
            return self._items.get(key)

        async def put(self, key: str, value: object) -> None:
            self._items[key] = value

    pool = _FakePool()
    calls = {"n": 0}

    def _flaky_build():
        calls["n"] += 1
        if calls["n"] <= 3:
            raise RuntimeError("일시적 다운로드 실패")
        return "loaded"

    for _ in range(3):
        with pytest.raises(RuntimeError):
            _asyncio_run(coordinator.get_or_build(pool, "flashrank_y", _flaky_build))

    # 회로 차단 상태 예외 확인
    from common.exceptions import ResourceBuildError

    with pytest.raises(ResourceBuildError):
        _asyncio_run(coordinator.get_or_build(pool, "flashrank_y", _flaky_build))

    # TTL 경과 후 재시도 → 성공 시 리소스 캐시에 저장되고 카운터 리셋.
    import time

    time.sleep(0.06)
    result = _asyncio_run(coordinator.get_or_build(pool, "flashrank_y", _flaky_build))
    assert result == "loaded"
    assert calls["n"] == 4


def _asyncio_run(coro):
    """단일 테스트용 헬퍼 — 임시 이벤트 루프에서 코루틴을 실행합니다."""
    import asyncio

    return asyncio.run(coro)
