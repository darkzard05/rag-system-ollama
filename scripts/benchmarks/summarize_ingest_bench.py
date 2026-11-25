"""Summarize paired ingest benchmark artifacts.

Usage:
    python scripts/benchmarks/summarize_ingest_bench.py --before before_p --after after_p

Pairs artifacts by their numeric suffix, so A/B interleaving survives a rerun.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _load(label: str) -> dict[int, dict[str, Any]]:
    pattern = re.compile(rf"^bench_ingest_{re.escape(label)}(\d+)_(\d+)\.json$")
    found: dict[int, dict[str, Any]] = {}
    for path in (ROOT / "reports").glob(f"bench_ingest_{label}*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        index, stamp = int(match.group(1)), int(match.group(2))
        current = found.get(index)
        if current is None or stamp > current["_stamp"]:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["_stamp"] = stamp
            payload["_path"] = str(path.relative_to(ROOT))
            found[index] = payload
    return found


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "median": round(statistics.median(values), 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _metric(payload: dict[str, Any], dotted_key: str) -> float:
    value: Any = payload
    for part in dotted_key.split("."):
        value = value[part]
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--metric", default="wall_s")
    args = parser.parse_args()

    before, after = _load(args.before), _load(args.after)
    indices = sorted(set(before) & set(after))
    if not indices:
        print("[ERROR] no paired artifacts found", file=sys.stderr)
        return 1

    before_values = [_metric(before[i], args.metric) for i in indices]
    after_values = [_metric(after[i], args.metric) for i in indices]
    deltas = [b - a for b, a in zip(before_values, after_values, strict=True)]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric": args.metric,
        "pairs": len(indices),
        "before": _stats(before_values),
        "after": _stats(after_values),
        "paired_improvement": {
            "median_saved": round(statistics.median(deltas), 2),
            "min_saved": round(min(deltas), 2),
            "max_saved": round(max(deltas), 2),
            "pct_of_before_median": round(
                100.0 * statistics.median(deltas) / statistics.median(before_values), 1
            ),
        },
        "artifacts": {
            "before": [before[i]["_path"] for i in indices],
            "after": [after[i]["_path"] for i in indices],
        },
    }
    slug = args.metric.replace(".", "_")
    out = ROOT / "reports" / f"ingest_compare_{slug}_{int(time.time())}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"artifact: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
