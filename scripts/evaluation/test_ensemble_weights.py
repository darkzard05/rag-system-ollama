"""
H3 Reproduction: Ensemble weights [0.4, 0.6] starve BM25 for keyword queries.

The config sets ensemble_weights: [0.4, 0.6] (BM25, FAISS). Dynamic
weighting is enabled with keyword_weight: 0.8 and semantic_weight: 0.8.
But dynamic weighting only triggers when the query is EXCLUSIVELY
keyword-heavy OR EXCLUSIVELY semantic-heavy. For mixed queries, the
base [0.4, 0.6] applies, giving FAISS 60% weight even when BM25 should
dominate.

This means keyword-based queries (e.g., "CM3 tokenization parameters")
get FAISS-weighted results that may miss exact keyword matches.

USAGE:
    python scripts/evaluation/test_ensemble_weights.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import re
import numpy as np

# From config.yml
DYNAMIC_WEIGHTING_CONFIG = {
    "enabled": True,
    "keyword_weight": 0.8,
    "semantic_weight": 0.8,
    "keyword_patterns": [r"v?\d+\.\d+(\.\d+)?", r"[a-zA-Z]+\d+", r"0x[0-9a-fA-F]+"],
    "semantic_keywords": [
        "차이",
        "작동",
        "이유",
        "철학",
        "설계",
        "구조",
        "개념",
        "설명해",
        "방법",
        "무엇",
        "어떻게",
        "장단점",
        "특징",
        "비교",
        "요약",
        "정의",
        "뭔가요",
        "뭐예요",
        "뭘까",
        "뭐야",
        "무엇인가요",
        "뭐죠",
    ],
}

ENSEMBLE_WEIGHTS = [0.4, 0.6]  # [BM25, FAISS]


def classify_query(query: str) -> dict:
    """Replicate the intent classification logic from _retrieval.py preprocess."""
    keyword_patterns = DYNAMIC_WEIGHTING_CONFIG["keyword_patterns"]
    semantic_keywords = DYNAMIC_WEIGHTING_CONFIG["semantic_keywords"]

    is_keyword_heavy = any(re.search(p, query) for p in keyword_patterns)
    is_semantic_heavy = any(k in query for k in semantic_keywords)

    if is_keyword_heavy and not is_semantic_heavy:
        kw_w = DYNAMIC_WEIGHTING_CONFIG["keyword_weight"]
        weights = {"bm25": kw_w, "faiss": round(1.0 - kw_w, 1)}
        intent = "keyword"
    elif is_semantic_heavy and not is_keyword_heavy:
        sm_w = DYNAMIC_WEIGHTING_CONFIG["semantic_weight"]
        weights = {"bm25": round(1.0 - sm_w, 1), "faiss": sm_w}
        intent = "semantic"
    else:
        weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
        intent = "mixed"

    return {
        "intent": intent,
        "weights": weights,
        "is_keyword_heavy": is_keyword_heavy,
        "is_semantic_heavy": is_semantic_heavy,
    }


def main():
    print("=" * 70)
    print("H3: Ensemble Weights Starvation Test")
    print("=" * 70)

    test_queries = [
        # Keyword-heavy queries (should trigger BM25 dominance)
        "CM3 tokenization parameters v1.2",
        "model 2201.07520 architecture",
        "0x4A3B encoding scheme",
        "BERT-base-uncased fine-tuning",
        # Semantic-heavy queries (should trigger FAISS dominance)
        "차이점은 무엇인가요",
        "모델의 작동 원리를 설명해 주세요",
        "이유를 설명해 주세요",
        # Mixed queries (should use base ensemble)
        "What is the primary objective of the CM3 model?",
        "How does the tokenization process work?",
        "Explain the multi-modal alignment mechanism",
        # Short queries
        "CM3",
        "tokenization",
    ]

    print(f"\nBase ensemble_weights: {ENSEMBLE_WEIGHTS} (BM25, FAISS)")
    print(
        f"Dynamic weighting: keyword_weight={DYNAMIC_WEIGHTING_CONFIG['keyword_weight']}, "
    )
    print(
        f"                   semantic_weight={DYNAMIC_WEIGHTING_CONFIG['semantic_weight']}"
    )
    print(f"   (Only triggers when query is EXCLUSIVELY one type)")

    print(
        f"\n{'Query':<50} {'Intent':<12} {'BM25 Wt':<10} {'FAISS Wt':<10} {'Problem'}"
    )
    print("-" * 95)

    problems = []
    for query in test_queries:
        result = classify_query(query)
        weights = result["weights"]
        problem = ""

        if result["intent"] == "mixed" and result["is_keyword_heavy"]:
            problem = "BM25 starved! Has keyword patterns but gets base 0.4"
            problems.append(query)
        elif result["intent"] == "mixed" and result["is_semantic_heavy"]:
            problem = "FAISS starved! Has semantic keywords but gets base 0.6"

        print(
            f"{query:<50} {result['intent']:<12} {weights['bm25']:<10.2f} {weights['faiss']:<10.2f} {problem}"
        )

    print(f"\n--- Analysis ---")
    print(f"Queries with keyword patterns but mixed intent: {len(problems)}")
    print(f"These queries have numeric/alphanumeric patterns that should")
    print(f"trigger keyword-heavy weighting, but the mixed intent check")
    print(f"prevents dynamic weighting from activating.")
    print(f"\nThe problem: is_keyword_heavy AND is_semantic_heavy queries")
    print(f"fall through to base ensemble [0.4, 0.6], starving BM25.")
    print(f"\nRECOMMENDATION:")
    print(f"1. Change base ensemble_weights to [0.5, 0.5] for neutrality")
    print(f"2. OR change the logic to trigger dynamic weighting when EITHER")
    print(f"   is_keyword_heavy OR is_semantic_heavy (not XOR)")
    print(f"3. OR add a third 'mixed' intent with balanced weights")
    print("=" * 70)


if __name__ == "__main__":
    main()
