"""Replay a recorded build trace through completion-indicator variants.

Usage:
    python scripts/verification/compare_build_completion.py [--trace PATH]

Reads a raw BuildEvent trace from reports/, replays it through BuildState and
to_viewmodel, then prints observable metrics for each completion design. The
JSON output is the scratch input for the Streamlit/Playwright DOM check.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from core.build_state import (  # noqa: E402
    BuildEvent,
    BuildPhase,
    apply_event,
    complete_build,
    initial_build_state,
)
from ui.components.doc_analysis.viewmodel import BuildVM, to_viewmodel  # noqa: E402
from ui.strings import set_lang, t  # noqa: E402

FIXED_ELAPSED = 7


def latest_trace() -> str:
    matches = sorted(
        glob.glob(os.path.join(ROOT, "reports", "build_event_trace_*.json"))
    )
    if not matches:
        raise SystemExit("no trace found; run record_build_event_trace.py first")
    return matches[-1]


def replay(trace_path: str) -> tuple[list[BuildVM | None], BuildVM]:
    with open(trace_path, encoding="utf-8") as fh:
        events = json.load(fh)

    build = initial_build_state("trace.pdf", "hash_trace")
    snapshots: list[BuildVM | None] = [to_viewmodel(build)]
    for item in events:
        event = BuildEvent(
            phase=BuildPhase(item["phase"]),
            pct=int(item["pct"]),
            detail_key=str(item.get("detail_key", "")),
            status=str(item.get("status", "")),
            ts=float(item["ts"]),
        )
        apply_event(build, event)
        snapshots.append(to_viewmodel(build))

    complete_build(build, cache_used=False)
    build.started_at = build.finished_at - FIXED_ELAPSED
    final = to_viewmodel(build)
    assert final is not None
    return snapshots, final


def _cache_tag(label: str) -> str:
    for key in ("cache_tag_reused", "cache_tag_new"):
        tag = t(key).strip()
        if tag in label:
            return tag
    return ""


def variants(final: BuildVM) -> dict[str, dict[str, Any]]:
    name = "trace.pdf"
    tag = _cache_tag(final.label)
    elapsed = FIXED_ELAPSED

    a_label = f"{name} · {t('build_complete_status')}{tag}"
    a_caption = (
        t("build_complete_caption").format(elapsed=elapsed)
        if elapsed
        else t("build_complete_default")
    )
    b_caption = None
    b_label = f"{name} · {t('build_complete_status')}{tag} · {elapsed}s"
    plus_label = (
        f"{name} · {t('build_complete_status')}{tag} · "
        f"{t('build_complete_elapsed').format(elapsed=elapsed)}"
    )
    c_caption = " · ".join(
        f"✓ {t(key)}"
        for key in (
            "build_step_extract",
            "build_step_chunk",
            "build_step_index",
            "build_step_finalize",
        )
    )
    c_caption = f"{c_caption} · {elapsed}s"
    d_label = f"{name} · {t('build_complete_default')}{tag}"
    d_caption = f"{t('chat_placeholder_ready')} · {elapsed}s"

    return {
        "A_baseline": {"label": a_label, "caption": a_caption},
        "B_one_line": {"label": b_label, "caption": b_caption},
        "B_plus_shipped": {"label": plus_label, "caption": a_caption},
        "C_step_summary": {"label": a_label, "caption": c_caption},
        "D_ready_prompt": {"label": d_label, "caption": d_caption},
    }


def metrics(
    snapshots: list[BuildVM | None], variants_by_name: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    complete_flags = [
        vm is not None and vm.status_state == "complete" for vm in snapshots
    ]
    terminal_transitions = sum(
        1
        for now, before in zip(
            complete_flags, [False, *complete_flags[:-1]], strict=True
        )
        if now and not before
    )
    result: dict[str, Any] = {
        "events": len(snapshots) - 1,
        "complete_transitions_in_running_path": terminal_transitions,
        "variants": {},
    }
    for name, spec in variants_by_name.items():
        caption = spec["caption"]
        result["variants"][name] = {
            "label": spec["label"],
            "caption": caption,
            "visible_lines": 1 if caption is None else 2,
            "visible_chars": len(spec["label"]) + len(caption or ""),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", default="")
    parser.add_argument("--lang", choices=("en", "ko"), default="en")
    args = parser.parse_args()

    set_lang(args.lang)
    trace = args.trace or latest_trace()
    snapshots, final = replay(trace)
    variants_by_name = variants(final)
    result = metrics(snapshots, variants_by_name)
    result["trace"] = os.path.relpath(trace, ROOT)
    result["lang"] = args.lang

    out_dir = os.path.join(ROOT, "reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(
        out_dir,
        f"completion_variants_{args.lang}_{time.strftime('%Y%m%d_%H%M%S')}.json",
    )
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    print(f"trace={result['trace']} lang={args.lang} events={result['events']}")
    transitions = result["complete_transitions_in_running_path"]
    print(f"complete transitions before finalize={transitions}")
    for name, spec in result["variants"].items():
        print(
            f"{name:<16} lines={spec['visible_lines']} "
            f"chars={spec['visible_chars']:<4} label={spec['label']!r}"
        )
        if spec["caption"]:
            print(f"{'':<16} caption={spec['caption']!r}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
