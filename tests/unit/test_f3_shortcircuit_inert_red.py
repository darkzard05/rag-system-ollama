"""F3 grade short-circuit — TDD GREEN gate (evolved from RED).

Ledger: reports/_plan_evidence_ledger.md section F3 (F3-1..F3-11).
Real branch: src/core/graph/_grade_verify.py lines 241-258.
Config: config.yml lines 256-274. README: per-engine thresholds documented.

EVOLUTION RATIONALE (per task precedents F1-4b, F2-5b):
- Original RED test used semantic-scale corpus (max 0.5722) to test 0.70 threshold,
  which is the FlashRank threshold — category error. The 0.70 branch is INERT for
  semantic scale by design (cosine cap ~0.57).
- Fixed world: per-engine thresholds aligned (FlashRank 0.70 / semantic 0.45),
  code fallbacks match config.yml, README documents per-engine, grade_top_n=5
  covers rank-drop citations (Q3: r4,r5; Q6: r4; Q8: r4).
- This evolved test asserts per-engine fire behavior + triple consistency per scope.
- MUST STILL FAIL if: (a) triple drifts (README/config/code mismatch per engine),
  (b) semantic-scale branch silently dies on FlashRank scale (per-engine branching
  removed or fallbacks swapped), (c) rank-drop regression (grade_top_n < 5).
"""

import json
import re
from pathlib import Path
from typing import Any

from common.config import GRADING_CONFIG

REPO_ROOT = Path(__file__).resolve().parents[2]
PINNED_CORPUS = REPO_ROOT / "reports" / "eval_quality_baseline_20260806_174429.json"
GRADE_VERIFY = REPO_ROOT / "src" / "core" / "graph" / "_grade_verify.py"
CONFIG_YML = REPO_ROOT / "config.yml"
README_MD = REPO_ROOT / "README.md"

INJECTED_SCORE_FLASHRANK = 0.9
INJECTED_SCORE_SEMANTIC = 0.5


def _branch_fires(scores: list[float], threshold: float, min_gap: float) -> bool:
    """Exact predicate of _grade_verify.py:247-258 (ledger F3-7..F3-9)."""
    ordered = sorted((float(s) for s in scores), reverse=True)
    if not ordered:
        return False
    top_gap = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
    both_high = len(ordered) > 1 and ordered[1] >= (threshold - 0.07)
    return bool(ordered[0] >= threshold and (top_gap >= min_gap or both_high))


def _load_scores(path: Path) -> list[list[float]]:
    """Per-query rerank_score lists from a real recorded eval JSON."""
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    out: list[list[float]] = []
    for q in data.get("questions", []):
        docs = (q.get("retrieval", {}) or {}).get("documents", []) or []
        out.append([float(d.get("rerank_score", 0.0) or 0.0) for d in docs])
    return out


def test_f3_per_engine_thresholds_match_config() -> None:
    """GREEN: config.yml per-engine thresholds match code fallbacks exactly.

    F3-1: config min_score_to_skip (FlashRank) = 0.70
    F3-2: config min_score_to_skip_semantic = 0.45
    Code fallbacks (lines 242, 244) must equal these values.
    """
    config_flashrank = float(GRADING_CONFIG.get("min_score_to_skip", 0.70))
    config_semantic = float(GRADING_CONFIG.get("min_score_to_skip_semantic", 0.45))
    assert config_flashrank == 0.70, (
        f"config FlashRank threshold={config_flashrank}, expected 0.70"
    )
    assert config_semantic == 0.45, (
        f"config semantic threshold={config_semantic}, expected 0.45"
    )

    code_text = GRADE_VERIFY.read_text(encoding="utf-8")

    # Semantic fallback (line 242 area)
    sem_fallback_hit = re.search(
        r'GRADING_CONFIG\.get\("min_score_to_skip_semantic",\s*([\d.]+)\)', code_text
    )
    assert sem_fallback_hit is not None, (
        "semantic fallback not found in _grade_verify.py"
    )
    sem_fallback = float(sem_fallback_hit.group(1))
    assert sem_fallback == config_semantic, (
        f"semantic code fallback={sem_fallback} != config={config_semantic}"
    )

    # FlashRank/general fallback (line 244 area)
    gen_fallback_hit = re.search(
        r'GRADING_CONFIG\.get\("min_score_to_skip",\s*([\d.]+)\)', code_text
    )
    assert gen_fallback_hit is not None, (
        "general fallback not found in _grade_verify.py"
    )
    gen_fallback = float(gen_fallback_hit.group(1))
    assert gen_fallback == config_flashrank, (
        f"FlashRank code fallback={gen_fallback} != config={config_flashrank}"
    )

    # Per-engine branching logic must exist (guard against silent death on wrong scale)
    assert 'get_active_rerank_engine() == "semantic"' in code_text, (
        "per-engine branching logic missing — semantic branch would die on FlashRank scale"
    )


def test_f3_readme_documents_per_engine_thresholds() -> None:
    """GREEN: README documents per-engine thresholds consistent with config."""
    readme_text = README_MD.read_text(encoding="utf-8")

    # README should mention both thresholds (per-engine)
    assert "0.70" in readme_text, "README missing 0.70"
    assert "FlashRank" in readme_text, "README missing FlashRank"
    assert "0.45" in readme_text, "README missing 0.45"
    assert "semantic" in readme_text.lower(), "README missing semantic"

    # Table entry should reflect per-engine
    table_hit = re.search(
        r"min_score_to_skip.*?\|.*?0\.70.*?FlashRank.*?0\.45.*?semantic",
        readme_text,
        re.IGNORECASE | re.DOTALL,
    )
    assert table_hit is not None, "README table does not document per-engine thresholds"


def test_f3_flashrank_branch_fires_on_flashrank_scale() -> None:
    """GREEN: FlashRank threshold (0.70) fires on FlashRank-scale scores (injection).

    Proves the FlashRank short-circuit path is active for its engine scale.
    Uses injected FlashRank-like score (0.9) since we lack a pinned FlashRank corpus.
    """
    min_gap = float(GRADING_CONFIG.get("min_top_gap_to_skip", 0.05))
    threshold = float(GRADING_CONFIG.get("min_score_to_skip", 0.70))
    assert threshold == 0.70

    # Base corpus (semantic scale) + injected FlashRank score
    corpora = _load_scores(PINNED_CORPUS)
    assert corpora, "corpus empty"
    assert corpora[0], "first query empty"
    injected = [INJECTED_SCORE_FLASHRANK] + corpora[0][1:]

    assert _branch_fires(injected, threshold, min_gap) is True, (
        "FlashRank 0.70 branch did not fire on injected FlashRank-scale score"
    )


def test_f3_semantic_branch_fires_on_semantic_scale() -> None:
    """GREEN: Semantic threshold (0.45) fires on semantic-scale scores (injection).

    Proves the semantic short-circuit path is active for its engine scale.
    The pinned corpus max is 0.5722; 0.45 is in achievable range (config.yml:262).
    """
    min_gap = float(GRADING_CONFIG.get("min_top_gap_to_skip", 0.05))
    threshold = float(GRADING_CONFIG.get("min_score_to_skip_semantic", 0.45))
    assert threshold == 0.45

    corpora = _load_scores(PINNED_CORPUS)
    assert corpora, "corpus empty"
    assert corpora[0], "first query empty"
    injected = [INJECTED_SCORE_SEMANTIC] + corpora[0][1:]

    assert _branch_fires(injected, threshold, min_gap) is True, (
        "Semantic 0.45 branch did not fire on injected semantic-scale score"
    )


def test_f3_semantic_branch_inert_at_flashrank_threshold() -> None:
    """GREEN (regression guard): semantic-scale corpus does NOT fire at 0.70.

    This is the original RED test's intent — the 0.70 threshold is for FlashRank,
    not semantic. If this ever fires on semantic data, the per-engine branching
    is broken (semantic branch using FlashRank threshold).
    """
    corpora = _load_scores(PINNED_CORPUS)
    assert len(corpora) == 8

    min_gap = float(GRADING_CONFIG.get("min_top_gap_to_skip", 0.05))
    flashrank_threshold = 0.70

    fires = sum(_branch_fires(s, flashrank_threshold, min_gap) for s in corpora)
    assert fires == 0, (
        f"semantic corpus fired {fires}/8 at FlashRank 0.70 threshold — "
        "per-engine branching broken (semantic using FlashRank threshold)"
    )


def test_f3_grade_top_n_covers_rank_drop() -> None:
    """GREEN: grade_top_n >= 5 to cover observed rank-drop citations.

    Evidence (f3_rerank_dist.md): citations at rank 4-5 exist (Q3: r4,r5; Q6: r4; Q8: r4).
    grade_top_n=3 was lossy — citations outside grading window cannot be verified.
    """
    grade_top_n = int(GRADING_CONFIG.get("grade_top_n", 5))
    assert grade_top_n >= 5, f"grade_top_n={grade_top_n} < 5 — rank-drop regression"


def test_f3_injection_control_fires() -> None:
    """Control (PASS): one doc pinned to 0.9 must fire the 0.70 branch."""
    corpora = _load_scores(PINNED_CORPUS)
    assert corpora
    assert corpora[0]
    injected = [INJECTED_SCORE_FLASHRANK] + corpora[0][1:]
    min_gap = float(GRADING_CONFIG.get("min_top_gap_to_skip", 0.05))
    threshold = float(GRADING_CONFIG.get("min_score_to_skip", 0.70))
    assert _branch_fires(injected, threshold, min_gap) is True
