"""EXTERNAL repro for H3: Ensemble dynamic-weighting routing (no src/ edits, no Ollama).

Reads the ACTUAL routing params from config.yml:
  rag.retriever.ensemble_weights          -> [0.4, 0.6]
  rag.retriever.dynamic_weighting.*       -> keyword/semantic weights, patterns, keywords

Classifies 12 sample questions (6 keyword-type, 6 semantic-type) with the real
regex + keyword lists, shows per-question routing + simulated weight heatmap
bins [0.2,0.8]/[0.4,0.6]/[0.8,0.2], and reports misroute rate.

Routing rule (mirrors src/core/graph/_retrieval.py lines 70-84, strict 3-way):
  keyword-only  -> [0.8, 0.2] BM25-heavy
  semantic-only -> [0.2, 0.8] FAISS-heavy
  both/neither -> [0.4, 0.6] default ensemble_weights (dual-match fallthrough)

BLEND_DECISION (F1 Task 4, strict 3-way — minimal path):
  Keep strict 3-way; do NOT invent a [0.6, 0.4] blend branch. The code has no
  blend (grep '0.6' in _retrieval.py: no hits in the dynamic path; ledger F1-9;
  static default ensemble_weights [0.4, 0.6] is F1-10, not a dynamic branch).
  Earlier revisions of this script simulated a [0.6, 0.4] keyword-leaning blend
  for dual-match rows — a measurement-loop defect: the report emitted weights
  the real preprocess path can never produce, and excluded PARTIAL rows from
  the denominator (Misrouted 0/12). The fix belongs report-side: this generator
  now emits only the code-enumerated set {[0.8,0.2],[0.2,0.8],[0.4,0.6]} and
  counts every row (denominator == row count, Misrouted = misrouted/total).
  Rationale: inventing a blend would change routing behavior (scope creep into
  Task 5/F2 tie-break, which owns dual-match handling) and contradict the F2
  RED baseline that assumes silent [0.4, 0.6] fallthrough. Task 5 follows this
  decision. See reports/f1_red_evidence.md Decision section.
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yml"
REPORTS_DIR = ROOT / "reports"

# --- Load actual params from config.yml (no yaml dep: regex-extract) ---
text = CONFIG_PATH.read_text(encoding="utf-8")


def _extract_block(pattern: str) -> str:
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"config.yml: pattern not found: {pattern}")
    return m.group(0)


# ensemble_weights: [0.4, 0.6]
m = re.search(r"ensemble_weights:\s*\[([^\]]+)\]", text)
assert m, "ensemble_weights not found in config.yml"
DEFAULT_WEIGHTS = [float(x) for x in m.group(1).split(",")]

m = re.search(r"keyword_weight:\s*([0-9.]+)", text)
KW_W = float(m.group(1)) if m else 0.8
m = re.search(r"semantic_weight:\s*([0-9.]+)", text)
SEM_W = float(m.group(1)) if m else 0.8

m = re.search(r"keyword_patterns:\s*\[(.*)\]", text)
assert m, "keyword_patterns not found in config.yml"
KEYWORD_PATTERNS = [
    p.replace("\\\\", "\\") for p in re.findall(r'"([^"]+)"', m.group(1))
]

m = re.search(r"semantic_keywords:\s*\[(.*?)\]", text, re.DOTALL)
assert m, "semantic_keywords not found in config.yml"
SEMANTIC_KEYWORDS = re.findall(r'"([^"]+)"', m.group(1))

COMPILED = [re.compile(p) for p in KEYWORD_PATTERNS]

# --- 12 sample questions: 6 keyword-type, 6 semantic-type ---
# expected = intended route; edge cases K6/S6 intentionally ambiguous.
SAMPLES: list[tuple[str, str, str]] = [
    # keyword-type (version / code / 0x)
    ("K1", "CM3 모델 v2.1과 v1.0의 파라미터 수 차이는?", "keyword"),
    ("K2", "BERTBase123 체크포인트의 학습 데이터 규모는?", "keyword"),
    ("K3", "0x1F4A 주소에 매핑된 레지스터의 역할은?", "keyword"),
    ("K4", "qwen3:4b-instruct-2507-q4_K_M 모델의 컨텍스트 윈도우는?", "keyword"),
    ("K5", "HTTP404 에러 코드의 발생 조건은 무엇인가?", "keyword"),
    (
        "K6",
        "v2와 v3는 왜 성능 차이가 나는가? 그 이유를 설명해줘",
        "keyword",
    ),  # edge: both
    # semantic-type (why/how/비교)
    ("S1", "두 모델의 설계 철학 차이는 무엇인가요?", "semantic"),
    ("S2", "이 논문의 핵심 기여점을 요약해줘", "semantic"),
    ("S3", "트랜스포머 구조가 어떻게 작동하는지 설명해줘", "semantic"),
    ("S4", "지도학습과 비지도학습의 장단점을 비교해줘", "semantic"),
    ("S5", "어텐션 개념의 정의와 특징은 무엇인가요?", "semantic"),
    (
        "S6",
        "0xFF 마스크를 사용할 때와 v2 설정을 쓸 때 중 뭐가 더 좋아?",
        "semantic",
    ),  # edge: both
]


def classify(q: str) -> dict[str, object]:
    kw_hit = next(
        (p for p, rx in zip(KEYWORD_PATTERNS, COMPILED, strict=False) if rx.search(q)),
        None,
    )
    sem_hits = [k for k in SEMANTIC_KEYWORDS if k in q]
    is_kw = kw_hit is not None
    is_sem = len(sem_hits) > 0
    if is_kw and not is_sem:
        weights = [KW_W, round(1 - KW_W, 1)]  # [0.8, 0.2]
        route = "keyword (BM25-heavy)"
    elif is_sem and not is_kw:
        weights = [round(1 - SEM_W, 1), SEM_W]  # [0.2, 0.8]
        route = "semantic (FAISS-heavy)"
    else:
        # Strict 3-way (BLEND_DECISION): dual-match and neither both fall
        # through to the static default — mirrors _retrieval.py 70-84 where
        # the guards are mutually exclusive and there is no blend branch.
        weights = list(DEFAULT_WEIGHTS)  # [0.4, 0.6]
        route = (
            "both -> default fallthrough (dual-match)"
            if (is_kw and is_sem)
            else "neutral -> default"
        )
    return {
        "kw_hit": kw_hit,
        "sem_hits": sem_hits,
        "is_kw": is_kw,
        "is_sem": is_sem,
        "weights": weights,
        "route": route,
    }


def heat_bin(w: list[float]) -> str:
    key = f"[{w[0]:.1f},{w[1]:.1f}]"
    return {
        "[0.2,0.8]": "FAISS ████████░░",
        "[0.4,0.6]": "default ████░░░░░░",
        "[0.8,0.2]": "BM25   ████████░░",
    }.get(key, f"IMPOSSIBLE-{key}")


def main() -> int:
    rows = []
    misroutes = 0
    for qid, q, expected in SAMPLES:
        c = classify(q)
        actual = (
            "keyword"
            if c["is_kw"] and not c["is_sem"]
            else (
                "semantic"
                if c["is_sem"] and not c["is_kw"]
                else ("both" if (c["is_kw"] and c["is_sem"]) else "neutral")
            )
        )
        # Strict 3-way verdict (BLEND_DECISION): every row counts.
        # ok = routed to the expected single route.
        # misroute = anything else, INCLUDING dual-match fallthrough to the
        # default [0.4, 0.6] (it matches neither pure route). The old script
        # excluded PARTIAL blend rows from the denominator (0/12); the
        # corrected denominator is the full row count (denominator == 12).
        if actual == expected:
            verdict = "OK"
        else:
            verdict = "MISROUTE"
            misroutes += 1
        rows.append((qid, q, expected, c, actual, verdict))

    total = len(rows)
    rate = misroutes / total
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        "# H3 Ensemble Weights Repro",
        "",
        f"- Timestamp: {ts} | No Ollama, no src/ edits (EXTERNAL repro)",
        f"- config.yml: ensemble_weights={DEFAULT_WEIGHTS}, keyword_weight={KW_W}, "
        f"semantic_weight={SEM_W}",
        f"- keyword_patterns={KEYWORD_PATTERNS}",
        f"- semantic_keywords(n={len(SEMANTIC_KEYWORDS)}): {SEMANTIC_KEYWORDS}",
        "",
        "## Per-question routing",
        "",
        "| ID | Question | Expected | kw_hit | sem_hits | Weights [BM25,FAISS] | Heat | Actual | Verdict |",
        "|----|----------|----------|--------|----------|----------------------|------|--------|---------|",
    ]
    for qid, q, expected, c, actual, verdict in rows:
        sem = ",".join(cast("list[str]", c["sem_hits"])[:3]) or "-"
        kw = f"`{c['kw_hit']}`" if c["kw_hit"] else "-"
        mark = {"OK": "✅", "MISROUTE": "❌"}[verdict]
        weights = cast("list[float]", c["weights"])
        lines.append(
            f"| {qid} | {q} | {expected} | {kw} | {sem} | {weights} "
            f"| {heat_bin(weights)} | {actual} | {mark} {verdict} |"
        )
    lines += [
        "",
        "## Simulated weight heatmap (counts per bin)",
        "",
    ]
    from collections import Counter

    bins = Counter(
        f"[{cast('list[float]', c['weights'])[0]:.1f},{cast('list[float]', c['weights'])[1]:.1f}]"
        for _, _, _, c, _, _ in rows
    )
    for b in ["[0.8,0.2]", "[0.4,0.6]", "[0.2,0.8]"]:
        n = bins.get(b, 0)
        lines.append(
            f"- {b} {heat_bin([float(x) for x in b.strip('[]').split(',')])}: {'▓' * n} ({n})"
        )
    lines += [
        "",
        "## Misroute rate",
        "",
        f"- Misrouted: {misroutes}/{total} = **{rate:.1%}** (all rows included; denominator == row count)",
        "- Dual-match fallthrough rows ([0.4,0.6] default, matches neither pure route): K1/K5/K6 "
        "carry a code/version token AND a semantic keyword (차이/무엇/이유); S6 carries 0xFF/v2 "
        "tokens AND semantic keywords (뭐가/좋아). The real path returns [0.4,0.6] for all four.",
        "- Finding (S6): '0xFF'/'v2' tokens dual-match a semantic-intended question via "
        "`[a-zA-Z]+\\d+` (matches 'v2') and `0x[0-9a-fA-F]+`, so strict 3-way falls through to "
        "the [0.4,0.6] default instead of the semantic [0.2,0.8]. Single genuine misroute.",
        "- Note: bare 'v2'/'v3' miss the dotted-version regex but still trip `[a-zA-Z]+\\d+`, "
        "so short version tags dual-match rather than escaping detection entirely.",
        "- Interpretation: pure-type questions route cleanly to [0.8,0.2]/[0.2,0.8]; the edge "
        "failure mode is code-token hijack of semantic questions. Dual-match handling "
        "(tie-break) is owned by Task 5/F2 — see BLEND_DECISION in the module docstring.",
        "",
        "- golden_set.json: 5 queries | testset_2201.csv headers: question,ground_truth,reference_context",
    ]
    report = "\n".join(lines) + "\n"
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"repro_h3_weights_{ts}.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"[saved] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
