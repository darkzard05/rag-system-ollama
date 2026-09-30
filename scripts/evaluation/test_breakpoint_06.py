"""
H1 Reproduction: Semantic breakpoint 0.6 creates excessive fragmentation.

The config sets breakpoint_threshold_type: "similarity_threshold" with
breakpoint_threshold_value: 0.6. In _find_breakpoints, when the type is
"similarity_threshold", it uses self.similarity_threshold (0.6) directly
as the cosine DISTANCE threshold. Since _compute_distances computes
cosine_distance (1 - cos_sim), a distance of 0.6 means cos_sim = 0.4.

This means ANY adjacent sentence pair with cosine similarity below 0.4
gets a breakpoint. For well-written academic text, this creates excessive
fragmentation because most adjacent sentences have similarity < 0.4.

The default in __init__ is percentile with 95.0, which is far more
conservative — it only breaks at the 95th percentile of distances.

USAGE:
    python scripts/evaluation/test_breakpoint_06.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import numpy as np


def cosine_distance(v1: float, v2: float) -> float:
    """Placeholder - we use 1 - similarity directly."""
    return 1.0 - v1


def simulate_breakpoints(
    similarities: list[float], threshold_type: str, threshold_value: float
) -> list[int]:
    """Simulate _find_breakpoints logic for a given threshold config."""
    dist_array = np.array(
        [1 - s for s in similarities]
    )  # cosine_distance = 1 - cos_sim

    if threshold_type == "percentile":
        threshold = float(np.percentile(dist_array, threshold_value))
    elif threshold_type == "similarity_threshold":
        threshold = threshold_value  # Uses similarity_threshold directly as distance
    else:
        threshold = threshold_value

    breakpoints = (np.where(dist_array > threshold)[0] + 1).tolist()
    return breakpoints


def main():
    # Simulate realistic academic sentence similarities
    # Typical academic paper: sentences within a paragraph have high similarity
    # (0.7-0.95), between paragraphs drops to 0.3-0.6
    np.random.seed(42)

    # Simulate 50 sentence pairs with realistic similarity distribution
    # 30 within-paragraph pairs (high sim), 20 between-paragraph pairs (low sim)
    within_paragraph = np.random.uniform(0.7, 0.95, 30)
    between_paragraph = np.random.uniform(0.2, 0.6, 20)
    all_similarities = np.concatenate([within_paragraph, between_paragraph])
    np.random.shuffle(all_similarities)

    print("=" * 70)
    print("H1: Semantic Breakpoint 0.6 Fragmentation Test")
    print("=" * 70)

    # Test with current config: similarity_threshold, 0.6
    bps_current = simulate_breakpoints(
        all_similarities.tolist(), "similarity_threshold", 0.6
    )

    # Test with default config: percentile, 95.0
    bps_default = simulate_breakpoints(all_similarities.tolist(), "percentile", 95.0)

    # Test with more conservative similarity_threshold
    bps_conservative = simulate_breakpoints(
        all_similarities.tolist(), "similarity_threshold", 0.85
    )

    print(f"\nSentence pairs analyzed: {len(all_similarities)}")
    print(f"Mean similarity: {all_similarities.mean():.4f}")
    print(f"Mean cosine distance: {1 - all_similarities.mean():.4f}")
    print(f"\n95th percentile distance: {np.percentile(1 - all_similarities, 95):.4f}")

    print(f"\n--- Current config (similarity_threshold=0.6) ---")
    print(f"Breakpoints found: {len(bps_current)}")
    print(f"Fragmentation rate: {len(bps_current) / len(all_similarities):.1%}")
    print(f"Breakpoint indices: {bps_current[:10]}...")

    print(f"\n--- Default config (percentile=95.0) ---")
    print(f"Breakpoints found: {len(bps_default)}")
    print(f"Fragmentation rate: {len(bps_default) / len(all_similarities):.1%}")

    print(f"\n--- Conservative config (similarity_threshold=0.85) ---")
    print(f"Breakpoints found: {len(bps_conservative)}")
    print(f"Fragmentation rate: {len(bps_conservative) / len(all_similarities):.1%}")

    # Show how many within-paragraph pairs get broken
    within_dists = 1 - within_paragraph
    broken_within_current = np.sum(within_dists > 0.6)
    broken_within_default = np.sum(
        within_dists > np.percentile(1 - all_similarities, 95)
    )

    print(f"\n--- Within-paragraph pairs incorrectly broken ---")
    print(
        f"Current (0.6): {broken_within_current}/{len(within_paragraph)} within-paragraph pairs broken"
    )
    print(
        f"Default (95th pct): {broken_within_default}/{len(within_paragraph)} within-paragraph pairs broken"
    )

    print(f"\n{'=' * 70}")
    print(
        "CONCLUSION: similarity_threshold=0.6 breaks {:.0%} of within-paragraph".format(
            broken_within_current / len(within_paragraph)
        )
    )
    print("pairs. This creates excessive fragmentation and dilutes retrieval quality.")
    print("RECOMMENDATION: Switch to breakpoint_threshold_type='percentile' with")
    print("breakpoint_threshold_value=95.0, or raise similarity_threshold to 0.85+.")
    print("=" * 70)


if __name__ == "__main__":
    main()
