"""F2 dead-range + silent dual-match fall-through — TDD RED repro.

Ledger: reports/_plan_evidence_ledger.md section F2 (F2-1..F2-13).
Stage-1 cap: src/core/graph/_retrieval.py:221 reads
RETRIEVER_CONFIG["search_kwargs"]["k"] (F2-1); config.yml:88 pins k=15
(F2-10). Aggregation: _retrieval.py:336 aggregate_results with
WEIGHTED_RRF and top_k=search_k (F2-2); dynamic_top_k :345-356 picks
min/max 12/18 on the RRF-scale gap (F2-3..F2-7, F2-12) then slices
aggregated[:dynamic_top_k] (:368). Guards :77/:81 (F2-8/F2-9) make the
keyword/semantic branches mutually exclusive, so a dual-match query
falls through to the static default [0.4, 0.6] (F2-11) with no log.
Config values :94-103 (F2-12/F2-13). Task 4/5 blend-vs-3-way decision
is PENDING — no blend branch is assumed here.

Tests (a) and (b) FAIL on the current tree (RED); controls PASS.
"""

import logging
import re
from typing import Any

import pytest
from api.schemas import AggregatedSearchResult
from common.config import (
    DYNAMIC_TOP_K_CONFIG,
    DYNAMIC_WEIGHTING_CONFIG,
    ENSEMBLE_WEIGHTS,
    RETRIEVER_CONFIG,
)
from core.graph.graph_builder import preprocess
from core.search_aggregator import (
    AggregatedResult,
    AggregationStrategy,
    SearchResultAggregator,
)


class MockWriter:
    """preprocess의 StreamWriter 키워드 인자를 대체하는 모의 객체."""

    def __call__(self, data: object, *, step: int | None = None) -> None:
        return None


DUAL_MATCH_QUERIES: list[str] = [
    "CM3 모델 v2.1과 v1.0의 파라미터 수 차이는?",
    "HTTP404 에러 코드의 발생 조건은 무엇인가?",
    "qwen3 모델의 num_ctx 설정 차이가 성능에 미치는 영향은?",
]

PURE_KEYWORD_QUERY: str = "BERTBase123 체크포인트의 학습 데이터 규모는?"
PURE_SEMANTIC_QUERY: str = "두 모델의 설계 철학 차이는 무엇인가요?"


async def _run_real_preprocess(query: str) -> dict[str, Any]:
    """Drive the real preprocess weight path (F2-8/F2-9 guards)."""
    state = {"input": query}
    config = {"configurable": {}}
    result: dict[str, Any] = await preprocess(state, config, writer=MockWriter())
    return result


def _real_dual_flags(query: str) -> tuple[bool, bool]:
    """Dual-match check with the REAL config lists (F2-13)."""
    keyword_patterns = DYNAMIC_WEIGHTING_CONFIG.get("keyword_patterns", [])
    semantic_keywords = DYNAMIC_WEIGHTING_CONFIG.get("semantic_keywords", [])
    is_keyword_heavy = any(re.search(p, query) for p in keyword_patterns)
    is_semantic_heavy = any(k in query for k in semantic_keywords)
    return bool(is_keyword_heavy), bool(is_semantic_heavy)


def _build_synthetic_pool(
    pool_n: int = 20,
) -> dict[str, list[AggregatedSearchResult]]:
    """20-doc dual-source pool with identical rank order (RRF-scale gaps)."""
    sources: dict[str, list[AggregatedSearchResult]] = {"bm25": [], "faiss": []}
    for rank in range(pool_n):
        doc_id = f"syn-doc-{rank:02d}"
        for node in ("bm25", "faiss"):
            sources[node].append(
                AggregatedSearchResult(
                    doc_id=doc_id,
                    content=f"synthetic passage {rank:02d} for F2 range probe",
                    score=float(pool_n - rank),
                    node_id=node,
                    metadata={},
                )
            )
    return sources


def _aggregate_and_slice(
    search_k: int, weights: dict[str, float]
) -> tuple[list[AggregatedResult], int, int]:
    """REAL WEIGHTED_RRF aggregate + verbatim dynamic_top_k slice (:336-368)."""
    source_results = _build_synthetic_pool(pool_n=20)
    aggregator = SearchResultAggregator()
    aggregated, _ = aggregator.aggregate_results(
        source_results,
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=search_k,
        weights=weights,
    )
    # Verbatim selection from _retrieval.py:345-356 (F2-3..F2-7).
    dynamic_cfg = DYNAMIC_TOP_K_CONFIG
    gap_threshold = float(dynamic_cfg.get("gap_threshold", 0.003))
    min_candidates = int(dynamic_cfg.get("min_candidates", 12))
    max_candidates = int(dynamic_cfg.get("max_candidates", 18))
    if len(aggregated) >= 10:
        score_gap = aggregated[0].aggregated_score - aggregated[9].aggregated_score
        dynamic_top_k = min_candidates if score_gap > gap_threshold else max_candidates
    else:
        dynamic_top_k = min_candidates
    final_docs = aggregated[:dynamic_top_k]
    return final_docs, dynamic_top_k, len(aggregated)


async def test_stage1_cap_makes_max_candidates_unreachable() -> None:
    """(a) RED: search_k=15 caps the pool so max_candidates=18 is dead."""
    # F2-1/F2-10: Stage-1 k comes from config (15, not the 25 fallback).
    search_k = int(RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 25))
    assert (
        search_k == 18
    )  # fixed-world encoding: k raised so max_candidates reachable (RED pinned 15 to prove the dead range, history preserved in git)
    # F2-12: dynamic range is {12, 18}.
    assert int(DYNAMIC_TOP_K_CONFIG.get("min_candidates", 12)) == 12
    assert int(DYNAMIC_TOP_K_CONFIG.get("max_candidates", 18)) == 18
    weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
    final_docs, dynamic_top_k, pooled = _aggregate_and_slice(search_k, weights)
    # RRF-scale gap (< 0.003) selects the max branch ...
    assert dynamic_top_k == 18  # F2-7 selector; passes
    assert pooled == search_k  # F2-2 aggregator top_k cut; passes
    # ... but the 20-doc pool was already cut to 15: 18 never reached.
    assert len(final_docs) == 18, (
        f"dead range: pool capped at search_k={search_k}, "
        f"dynamic_top_k={dynamic_top_k} unreachable "
        f"(final={len(final_docs)}, pooled={pooled})"
    )


async def test_dual_match_fallthrough_is_silent(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """(b) RED: dual-match falls to default [0.4, 0.6] with no log/flag."""
    # F2-11: static default is [0.4, 0.6]; F2-13: guard lists are real.
    assert list(ENSEMBLE_WEIGHTS) == [0.4, 0.6]
    caplog.set_level(logging.INFO)
    for query in DUAL_MATCH_QUERIES:
        caplog.clear()
        is_keyword, is_semantic = _real_dual_flags(query)
        assert is_keyword and is_semantic, (
            f"{query!r} must dual-match via real config lists (F2-13)"
        )
        result = await _run_real_preprocess(query)
        weights: dict[str, float] = result["search_weights"]
        # Silent default today: passes — the defect is the missing signal.
        assert [weights["bm25"], weights["faiss"]] == [0.4, 0.6], (
            f"{query!r}: expected silent default, got {weights}"
        )
        log_marks = "dual_match_fallthrough" in caplog.text
        state_marks = result.get("dual_match_fallthrough") is True
        assert log_marks or state_marks, (
            f"{query!r}: dual-match fell through to default [0.4, 0.6] "
            "with no dual_match_fallthrough log/flag (F2-8/F2-9)"
        )


async def test_raised_search_k_makes_18_reachable() -> None:
    """Control (PASS): search_k>=18 removes the cap, 18 is reached."""
    weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
    final_docs, dynamic_top_k, pooled = _aggregate_and_slice(20, weights)
    assert dynamic_top_k == 18
    assert pooled == 20
    assert len(final_docs) == 18


async def test_single_signal_queries_keep_exclusive_routing() -> None:
    """Control (PASS): single-signal queries keep exclusive routing."""
    kw_result = await _run_real_preprocess(PURE_KEYWORD_QUERY)
    kw_weights: dict[str, float] = kw_result["search_weights"]
    assert kw_weights == {"bm25": 0.8, "faiss": 0.2}
    sm_result = await _run_real_preprocess(PURE_SEMANTIC_QUERY)
    sm_weights: dict[str, float] = sm_result["search_weights"]
    assert sm_weights == {"bm25": 0.2, "faiss": 0.8}
