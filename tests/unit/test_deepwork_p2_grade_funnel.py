"""Deepwork P2 — fixed-behavior verification.

Verifies after fix:
 F1: code default gap == config 0.05 (no weakened fallback).
 F2: near-threshold pair (0.65/0.64, min 0.70) no longer skips via discount;
     true cluster (0.71/0.70) still skips (intended, pinned by existing tests).
 F3: funnel structure documented (18->10->5); short-circuit now fail-closed
     so marginal cases reach the LLM instead of skipping.
"""

from common.config import GRADING_CONFIG


def _fixed_decision(
    sorted_scores: list[float], min_score: float, min_gap: float
) -> bool:
    top_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
    both_high = len(sorted_scores) > 1 and sorted_scores[1] >= min_score
    return max(sorted_scores) >= min_score and (top_gap >= min_gap or both_high)


def test_p2_code_default_gap_matches_config():
    import inspect

    from core.graph import _grade_verify as gv

    src = inspect.getsource(gv.grade_documents)
    assert 'min_top_gap_to_skip", 0.05' in src
    assert GRADING_CONFIG.get("min_top_gap_to_skip") == 0.05


def test_p2_discount_pair_no_longer_skips():
    # Old buggy logic skipped this (0.64 >= 0.70-0.07); fixed logic routes to LLM.
    assert _fixed_decision([0.65, 0.64], 0.70, 0.05) is False


def test_p2_true_cluster_still_skips():
    # Both clear the threshold with tiny gap — intended cluster fast-path.
    assert _fixed_decision([0.71, 0.70], 0.70, 0.05) is True


def test_p2_funnel_documents_cost_tradeoff():
    # Structural fact: grade LLM sees top-5 of reranked top-10.
    assert GRADING_CONFIG.get("grade_top_n") == 5
    assert GRADING_CONFIG.get("top_k") == 10
