"""CI self-check for F1 routing-weight reports (BLEND_DECISION: strict 3-way).

Validates a repro_h3_weights report file:
  1. Every emitted Weights [BM25,FAISS] cell is in the code-enumerated set
     {[0.8,0.2],[0.4,0.6],[0.2,0.8]} — any [0.6,0.4] blend (or IMPOSSIBLE
     marker) fails, since src/core/graph/_retrieval.py lines 70-84 has no
     blend branch (ledger F1-9).
  2. The Misrouted denominator equals the data-row count (PARTIAL rows must
     be included, never excluded).

Usage: python scripts/evaluation/check_f1_report.py <report.md>
Exit 0 when all checks pass, 1 otherwise. Stdlib only.
"""

from __future__ import annotations

import re
import sys

CODE_ENUMERATED = {"[0.8,0.2]", "[0.4,0.6]", "[0.2,0.8]"}

ROW_RE = re.compile(r"^\| (K\d+|S\d+) \|")
WEIGHT_RE = re.compile(r"\[(\d\.\d), ?(\d\.\d)\]")
MISROUTE_RE = re.compile(r"Misrouted:\s*(\d+)\s*/\s*(\d+)")


def check_report(path: str) -> list[str]:
    errors: list[str] = []
    with open(path, encoding="utf-8") as f:
        text = f.read()
    rows = [ln for ln in text.splitlines() if ROW_RE.match(ln)]
    if not rows:
        return ["no data rows found"]
    for ln in rows:
        m = WEIGHT_RE.search(ln)
        if not m:
            errors.append(f"row without weights cell: {ln[:60]}")
            continue
        key = f"[{m.group(1)},{m.group(2)}]"
        if key not in CODE_ENUMERATED:
            errors.append(
                f"impossible weight {key} not in code-enumerated set: {ln[:80]}"
            )
    if "IMPOSSIBLE-" in text:
        errors.append("report contains IMPOSSIBLE-marked weights")
    m = MISROUTE_RE.search(text)
    if not m:
        errors.append("Misrouted: m/n line not found")
    elif int(m.group(2)) != len(rows):
        errors.append(
            f"denominator {m.group(2)} != row count {len(rows)} (PARTIAL rows excluded?)"
        )
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <report.md>", file=sys.stderr)
        return 2
    errors = check_report(sys.argv[1])
    for e in errors:
        print(f"FAIL: {e}", file=sys.stderr)
    if errors:
        return 1
    print(
        f"OK: {sys.argv[1]} — weights in code-enumerated set, denominator == row count"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
