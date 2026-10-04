"""F3 numbers-first measurement: grade short-circuit branch inertness gate.

Ledger: reports/_plan_evidence_ledger.md section F3 (F3-1..F3-11).
Real branch: src/core/graph/_grade_verify.py lines 241-258 (engine-aware
threshold resolution + ``max_rerank >= threshold AND (gap OR both-high)``).
Config: config.yml lines 256-274 (grading triple + top_k/grade_top_n).
README ground truth: ``rag.prompts.grading.min_score_to_skip`` documented
as 0.85 while config sets 0.70 and code falls back to 0.45.

Every number in the output traces to a real eval JSON (never invented).
Usage:
    python scripts/measure_grade_shortcircuit.py \
        --eval-json <latest reports/eval_quality_*.json> \
        --out reports/f3_rerank_dist.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

THRESHOLDS_DEFAULT: list[float] = [0.85, 0.70, 0.45]
GRADE_TOP_N_DEFAULT = 3


def _load_grading_config() -> dict[str, Any]:
    """Real grading thresholds: common.config.GRADING_CONFIG if importable."""
    try:
        from common.config import GRADING_CONFIG  # noqa: PLC0415

        return dict(GRADING_CONFIG)
    except Exception:
        pass
    try:
        import yaml  # noqa: PLC0415

        raw = yaml.safe_load((REPO_ROOT / "config.yml").read_text(encoding="utf-8"))
        grading = raw["rag"]["prompts"]["grading"]
        return dict(grading)
    except Exception:
        return {}


def _active_engine() -> str:
    """Real engine probe: core.async_reranker.get_active_rerank_engine."""
    try:
        from core.async_reranker import get_active_rerank_engine  # noqa: PLC0415

        return str(get_active_rerank_engine())
    except Exception:
        return "unknown"


def shortcircuit_fires(scores: list[float], threshold: float, min_gap: float) -> bool:
    """Exact branch logic of _grade_verify.py:247-258 (ledger F3-7..F3-9).

    sorted_scores desc; top_gap = s[0]-s[1] (1.0 if single doc);
    both_high_confidence = s[1] >= threshold - 0.07;
    fires iff max >= threshold AND (top_gap >= min_gap OR both_high).
    """
    ordered = sorted((float(s) for s in scores), reverse=True)
    if not ordered:
        return False
    top_gap = ordered[0] - ordered[1] if len(ordered) > 1 else 1.0
    both_high = len(ordered) > 1 and ordered[1] >= (threshold - 0.07)
    return bool(ordered[0] >= threshold and (top_gap >= min_gap or both_high))


def _scale_label(max_score: float) -> str:
    """Heuristic engine-scale label (eval JSON records no engine field).

    FlashRank sigmoid observed 0.06~0.91, bi-encoder cosine 0.32~0.57
    (_grade_verify.py:235-236, R3b-02). Anything above 0.60 cannot be a
    cosine score, so it is FlashRank-scale; the rest is semantic-like.
    """
    return "flashrank-like" if max_score > 0.60 else "semantic-like"


def _load_eval(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def _query_rows(eval_data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for q in eval_data.get("questions", []):
        retr = q.get("retrieval", {}) or {}
        docs = retr.get("documents", []) or []
        scores = [float(d.get("rerank_score", 0.0) or 0.0) for d in docs]
        rows.append(
            {
                "query": str(q.get("query", "")),
                "scores": scores,
                "max_rerank": float(retr.get("max_rerank_score", 0.0) or 0.0),
                "ttft": q.get("ttft"),
                "judge": q.get("judge"),
                "judge_excluded": bool(q.get("judge_excluded", False)),
                "answer": q.get("answer"),
                "docs": docs,
            }
        )
    return rows


def _judge_valid(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Judge-bias note: exclude runs where judge was added mid-batch.

    Mid-batch judge backfills surface as judge=None with a judge_error, or
    judge_excluded=True. Only rows with a real numeric judge count.
    """
    valid = [
        r
        for r in rows
        if isinstance(r["judge"], (int, float)) and not r["judge_excluded"]
    ]
    return valid


def _cited_ranks(row: dict[str, Any]) -> list[tuple[float, int | None]]:
    """Rank (1-based, by rerank_score desc) of each finally-cited doc.

    Citation ``score`` is rounded to 3 decimals in the answer JSON, so match
    with tolerance 0.0015 and prefer the same page on ties. Returns
    (citation_score, rank or None if unmatchable).
    """
    docs = row["docs"]
    ordered = sorted(
        range(len(docs)),
        key=lambda i: float(docs[i].get("rerank_score", 0.0) or 0.0),
        reverse=True,
    )
    out: list[tuple[float, int | None]] = []
    try:
        answer = json.loads(row["answer"]) if isinstance(row["answer"], str) else {}
    except (json.JSONDecodeError, TypeError, ValueError):
        return out
    citations = answer.get("citations", []) if isinstance(answer, dict) else []
    for cit in citations:
        try:
            cscore = float(cit.get("score", -1.0))
            cpage = cit.get("page")
        except (TypeError, ValueError):
            continue
        best: int | None = None
        for rank, idx in enumerate(ordered, start=1):
            dscore = float(docs[idx].get("rerank_score", 0.0) or 0.0)
            if abs(dscore - cscore) <= 0.0015:
                if best is None or docs[idx].get("page") == cpage:
                    best = rank
                if docs[idx].get("page") == cpage:
                    break
        out.append((cscore, best))
    return out


def _render(
    eval_path: str,
    compare_path: str | None,
    thresholds: list[float],
    min_gap: float,
    grade_top_n: int,
    engine: str,
    grading_cfg: dict[str, Any],
) -> str:
    data = _load_eval(eval_path)
    rows = _query_rows(data)
    meta = data.get("meta", {})
    valid = _judge_valid(rows)
    lines: list[str] = []
    lines.append("# F3 grade short-circuit distribution (numbers-first gate)")
    lines.append("")
    lines.append(f"- corpus: `{os.path.basename(eval_path)}`")
    lines.append(
        f"- meta: tag={meta.get('tag')} ts={meta.get('timestamp')} "
        f"no_llm={meta.get('no_llm')}"
    )
    lines.append(f"- active rerank engine (this process): `{engine}`")
    lines.append(
        f"- grading config: min_score_to_skip="
        f"{grading_cfg.get('min_score_to_skip')} "
        f"semantic={grading_cfg.get('min_score_to_skip_semantic')} "
        f"gap={min_gap} top_k={grading_cfg.get('top_k')} "
        f"grade_top_n={grade_top_n}"
    )
    lines.append(
        f"- judge-valid rows: {len(valid)}/{len(rows)} "
        "(mid-batch None->score backfills excluded; see note)"
    )
    lines.append("")

    lines.append("## 1. max_rerank distribution (engine-scale split)")
    lines.append("")
    lines.append(
        "Scale label is heuristic (>0.60 = flashrank-like; eval JSON "
        "records no engine field; ranges per _grade_verify.py:235)."
    )
    lines.append("")
    lines.append("| query # | max_rerank | scale | n_docs |")
    lines.append("| --- | --- | --- | --- |")
    for i, r in enumerate(rows, start=1):
        lines.append(
            f"| {i} | {r['max_rerank']:.4f} | "
            f"{_scale_label(r['max_rerank'])} | {len(r['scores'])} |"
        )
    lines.append("")
    bins = [0.10, 0.45, 0.70, 0.85]
    counts = [0] * (len(bins) + 1)
    for r in rows:
        m = r["max_rerank"]
        placed = False
        for bi, edge in enumerate(bins):
            if m < edge:
                counts[bi] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = ["<0.10", "0.10-0.45", "0.45-0.70", "0.70-0.85", ">=0.85"]
    lines.append("| max_rerank bin | count |")
    lines.append("| --- | --- |")
    for lab, c in zip(labels, counts, strict=True):
        lines.append(f"| {lab} | {c} |")
    lines.append("")

    lines.append("## 2. branch-fire counts (exact _grade_verify.py:256 logic)")
    lines.append("")
    lines.append("| threshold | fires / n | fraction |")
    lines.append("| --- | --- | --- |")
    fire_by_t: dict[float, list[bool]] = {}
    for t in thresholds:
        fired = [shortcircuit_fires(r["scores"], t, min_gap) for r in rows]
        fire_by_t[t] = fired
        frac = sum(fired) / len(fired) if fired else 0.0
        lines.append(f"| {t:.2f} | {sum(fired)}/{len(fired)} | {frac:.3f} |")
    lines.append("")

    lines.append("## 3. rerank log sample (recorded ranks) + rank-drop table")
    lines.append("")
    lines.append(
        "Eval logs record only the top-5 retrieved docs per query "
        "(prod pool is top_k=10 -> grade_top_n=3); ranks below are "
        "within the recorded slice."
    )
    lines.append("")
    for i, r in enumerate(rows[:3], start=1):
        srt = sorted(r["scores"], reverse=True)
        lines.append(f"- Q{i} ranks: " + ", ".join(f"{s:.4f}" for s in srt))
    lines.append("")
    lines.append(f"| query # | cited scores -> rank | lossy (rank>{grade_top_n}) |")
    lines.append("| --- | --- | --- |")
    for i, r in enumerate(rows, start=1):
        cited = _cited_ranks(r)
        if not cited:
            lines.append(f"| {i} | (no citations) | - |")
            continue
        desc = ", ".join(
            f"{s:.3f}->r{rk}" if rk is not None else f"{s:.3f}->?" for s, rk in cited
        )
        lossy = any(rk is not None and rk > grade_top_n for _, rk in cited)
        lines.append(f"| {i} | {desc} | {'YES' if lossy else 'no'} |")
    lines.append("")

    lines.append("## 4. TTFT: totals by branch outcome + stage-split note")
    lines.append("")
    fired070 = fire_by_t.get(0.70, [False] * len(rows))
    fast = [
        r["ttft"]
        for r, f in zip(rows, fired070, strict=True)
        if f and isinstance(r["ttft"], (int, float))
    ]
    slow = [
        r["ttft"]
        for r, f in zip(rows, fired070, strict=True)
        if not f and isinstance(r["ttft"], (int, float))
    ]
    lines.append("| outcome @0.70 | n | mean TTFT(s) |")
    lines.append("| --- | --- | --- |")
    for name, vals in (
        ("fires (short-circuit path)", fast),
        ("no-fire (full LLM grade path)", slow),
    ):
        mean = sum(vals) / len(vals) if vals else 0.0
        lines.append(f"| {name} | {len(vals)} | {mean:.2f} |")
    lines.append("")
    lines.append(
        "NOTE (stage split): scripts/evaluation/eval_quality.py "
        "records TTFT totals only — no retrieve(k)/rerank/grade "
        "stage breakdown exists in the schema. A per-stage "
        "retrieve-vs-rerank-vs-grade(LLM-vs-shortcircuit) split "
        "needs an instrumented live run and is out of scope for "
        "this distribution gate; the by-outcome proxy above stands "
        "in for it."
    )
    lines.append("")

    if compare_path and os.path.exists(compare_path):
        cdata = _load_eval(compare_path)
        crows = _query_rows(cdata)
        lines.append("## 5. engine-scale comparison (recorded corpus)")
        lines.append("")
        lines.append(
            f"- compare: `{os.path.basename(compare_path)}` (semantic/bi-encoder scale)"
        )
        lines.append("| threshold | latest fires | compare fires |")
        lines.append("| --- | --- | --- |")
        for t in thresholds:
            a = sum(shortcircuit_fires(r["scores"], t, min_gap) for r in rows)
            b = sum(shortcircuit_fires(r["scores"], t, min_gap) for r in crows)
            lines.append(f"| {t:.2f} | {a}/{len(rows)} | {b}/{len(crows)} |")
        lines.append("")

    lines.append("## 6. verdict (numbers only)")
    lines.append("")
    lines.append(
        "- README documents min_score_to_skip=0.85; config sets "
        "0.70 (semantic 0.45); code fallbacks are 0.45 general / "
        "0.40 semantic / gap 0.03 — triple inconsistent."
    )
    lines.append(
        "- Fire counts above are computed with the exact "
        "max>=threshold AND (gap OR both-high) predicate; "
        "0.70 is unreachable for semantic-scale scores "
        "(cosine cap ~0.57 per config.yml:262)."
    )
    lines.append(
        "- Task 6 fix is gated on these numbers; no thresholds "
        "were touched by this script."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entry: measure recorded corpus, write markdown report."""
    parser = argparse.ArgumentParser(description="F3 short-circuit measurement")
    parser.add_argument(
        "--eval-json", default="", help="eval JSON path (default: latest reports file)"
    )
    parser.add_argument(
        "--compare-json",
        default="",
        help="optional second corpus for engine-scale compare",
    )
    parser.add_argument("--out", required=True, help="markdown output path")
    parser.add_argument(
        "--thresholds", default=",".join(str(t) for t in THRESHOLDS_DEFAULT)
    )
    args = parser.parse_args()

    eval_path = args.eval_json
    if not eval_path:
        candidates = sorted(
            glob.glob(str(REPO_ROOT / "reports" / "eval_quality_*.json")),
            key=os.path.getmtime,
        )
        if not candidates:
            print("no reports/eval_quality_*.json found", file=sys.stderr)
            return 2
        eval_path = candidates[-1]
    thresholds = [float(t) for t in str(args.thresholds).split(",") if t]
    grading_cfg = _load_grading_config()
    min_gap = float(grading_cfg.get("min_top_gap_to_skip", 0.05) or 0.05)
    grade_top_n = int(grading_cfg.get("grade_top_n", GRADE_TOP_N_DEFAULT) or 3)
    compare = args.compare_json or None
    if compare is None:
        legacy = REPO_ROOT / "reports" / "eval_quality_baseline_20260806_174429.json"
        if legacy.exists() and str(legacy) != eval_path:
            compare = str(legacy)
    md = _render(
        eval_path,
        compare,
        thresholds,
        min_gap,
        grade_top_n,
        _active_engine(),
        grading_cfg,
    )
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
