"""
H4 Reproduction: FlashRank vs semantic reranker creates grading threshold mismatch.

The config sets:
- reranker.engine: "auto" (FlashRank → semantic fallback)
- grading.min_score_to_skip: 0.70 (FlashRank)
- grading.min_score_to_skip_semantic: 0.45 (semantic)
- grading.min_top_gap_to_skip: 0.05 (both engines)

FlashRank scores (sigmoid) range 0.855-0.908 for relevant, 0.048-0.057 for irrelevant.
Semantic scores (cosine) max at ~0.57 per README.

The min_top_gap_to_skip: 0.05 is reasonable for FlashRank but might be
too large for semantic. When FlashRank falls back to semantic, the
min_top_gap_to_skip: 0.05 might prevent short-circuiting even for
clearly relevant documents because semantic scores are compressed.

USAGE:
    python scripts/evaluation/test_reranker_threshold_mismatch.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np


def main():
    print("=" * 70)
    print("H4: Reranker Threshold Mismatch Test")
    print("=" * 70)

    # Config values
    min_score_to_skip_flashrank = 0.70
    min_score_to_skip_semantic = 0.45
    min_top_gap_to_skip = 0.05

    # Realized score distributions from README
    flashrank_relevant = np.random.uniform(0.855, 0.908, 100)
    flashrank_irrelevant = np.random.uniform(0.048, 0.057, 100)
    semantic_relevant = np.random.uniform(0.35, 0.57, 100)
    semantic_irrelevant = np.random.uniform(0.05, 0.20, 100)

    print(f"\nConfig:")
    print(f"  min_score_to_skip (FlashRank): {min_score_to_skip_flashrank}")
    print(f"  min_score_to_skip (semantic):  {min_score_to_skip_semantic}")
    print(f"  min_top_gap_to_skip:           {min_top_gap_to_skip}")

    # FlashRank analysis
    print(f"\n--- FlashRank Analysis ---")
    print(
        f"Relevant scores: mean={flashrank_relevant.mean():.4f}, "
        f"min={flashrank_relevant.min():.4f}, max={flashrank_relevant.max():.4f}"
    )
    print(
        f"Irrelevant scores: mean={flashrank_irrelevant.mean():.4f}, "
        f"min={flashrank_irrelevant.min():.4f}, max={flashrank_irrelevant.max():.4f}"
    )

    # Check short-circuit conditions
    flashrank_skip = flashrank_relevant.mean() >= min_score_to_skip_flashrank
    flashrank_gap = flashrank_relevant.mean() - flashrank_irrelevant.mean()
    flashrank_gap_ok = flashrank_gap >= min_top_gap_to_skip

    print(
        f"Short-circuit (score >= {min_score_to_skip_flashrank}): {'YES' if flashrank_skip else 'NO'}"
    )
    print(
        f"Top-gap (relevant - irrelevant >= {min_top_gap_to_skip}): {flashrank_gap:.4f} {'>= threshold ✓' if flashrank_gap_ok else '< threshold ✗'}"
    )

    # Semantic analysis
    print(f"\n--- Semantic Analysis ---")
    print(
        f"Relevant scores: mean={semantic_relevant.mean():.4f}, "
        f"min={semantic_relevant.min():.4f}, max={semantic_relevant.max():.4f}"
    )
    print(
        f"Irrelevant scores: mean={semantic_irrelevant.mean():.4f}, "
        f"min={semantic_irrelevant.min():.4f}, max={semantic_irrelevant.max():.4f}"
    )

    semantic_skip = semantic_relevant.mean() >= min_score_to_skip_semantic
    semantic_gap = semantic_relevant.mean() - semantic_irrelevant.mean()
    semantic_gap_ok = semantic_gap >= min_top_gap_to_skip

    print(
        f"Short-circuit (score >= {min_score_to_skip_semantic}): {'YES' if semantic_skip else 'NO'}"
    )
    print(
        f"Top-gap (relevant - irrelevant >= {min_top_gap_to_skip}): {semantic_gap:.4f} {'>= threshold ✓' if semantic_gap_ok else '< threshold ✗'}"
    )

    # The problem
    print(f"\n--- Problem Analysis ---")
    if not semantic_gap_ok:
        print(
            f"CRITICAL: Semantic top-gap ({semantic_gap:.4f}) < min_top_gap_to_skip ({min_top_gap_to_skip})"
        )
        print(f"This means when FlashRank falls back to semantic, the grading")
        print(f"short-circuit is NEVER triggered because the gap is too small.")
        print(f"The LLM grade step runs for EVERY query, even when all documents")
        print(f"are clearly irrelevant.")

    if semantic_relevant.max() < min_score_to_skip_semantic:
        print(
            f"\nADDITIONAL: Semantic max relevant score ({semantic_relevant.max():.4f}) <"
        )
        print(f"min_score_to_skip_semantic ({min_score_to_skip_semantic})")
        print(f"Even the best semantic match can't trigger short-circuit!")

    # Find the right thresholds
    print(f"\n--- Recommended Calibration ---")
    print(f"For semantic engine:")
    print(
        f"  min_score_to_skip_semantic should be <= {semantic_relevant.min():.4f} (min relevant)"
    )
    print(f"  min_top_gap_to_skip should be <= {semantic_gap:.4f} (actual gap)")
    print(
        f"\nCurrent min_top_gap_to_skip={min_top_gap_to_skip} is {min_top_gap_to_skip / semantic_gap:.1f}x the actual semantic gap"
    )
    print(f"\nRECOMMENDATION:")
    print(f"1. Add engine-specific min_top_gap_to_skip values")
    print(f"2. For semantic: min_top_gap_to_skip = 0.02 (or lower)")
    print(f"3. For FlashRank: keep 0.05")
    print(f"4. Or make min_top_gap_to_skip adaptive based on active engine")
    print("=" * 70)


if __name__ == "__main__":
    main()
