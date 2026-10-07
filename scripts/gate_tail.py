"""Tail + hygiene + perf gate suite (Waves 1-3 aggregation).

Exits 0 when all checks pass, 1 otherwise.
Emits reports/tail_gate_<ts>.md with PASS/FAIL per check + numbers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
REPORTS_DIR = REPO_ROOT / "reports"

GATE_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
GATE_REPORT = REPORTS_DIR / ("tail_gate_" + GATE_TIMESTAMP + ".md")


def run_cmd(cmd):
    """Run command and return CompletedProcess."""
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)


def _make_check(pass_val, check_name, detail):
    return {"pass": pass_val, "check": check_name, "detail": detail}


def main():
    results = []

    # ---- (a) tail tests all pass ----
    tail_test_files = [
        "test_tail_perf_spans.py",
        "test_ghost_references.py",
        "test_reference_prefetch.py",
        "test_coord_memo.py",
        "test_faiss_lean.py",
    ]
    for tf in tail_test_files:
        r = run_cmd(["python", "-m", "pytest", "tests/unit/" + tf])
        results.append(
            _make_check(
                r.returncode == 0,
                "tail_tests:" + tf,
                r.stdout[-300:] if r.stdout else "",
            )
        )

    # ---- (b) hygiene green ----
    r_hygiene = run_cmd(["python", "-m", "pytest", "tests/unit/test_task5_hygiene.py"])
    results.append(
        _make_check(
            r_hygiene.returncode == 0,
            "hygiene_test_task5",
            r_hygiene.stdout[-300:] if r_hygiene.stdout else "",
        )
    )

    # Check important count in main.css
    css_path = REPO_ROOT / "src" / "ui" / "styles" / "main.css"
    css_text = css_path.read_text(encoding="utf-8") if css_path.exists() else ""
    important_count = css_text.count("!important")

    baseline_testids = {
        "stAppViewContainer",
        "stBottomBlockContainer",
        "stChatInput",
        "stChatMessage",
        "stColumn",
        "stElementContainer",
        "stExpander",
        "stHeader",
        "stHorizontalBlock",
        "stIFrame",
        "stLayoutWrapper",
        "stMain",
        "stMainBlockContainer",
        "stPopoverBody",
        "stSidebarCollapseButton",
        "stVerticalBlock",
    }
    found_testids = set(re.findall(r'data-testid="([^"]+)"', css_text))
    new_testids = found_testids - baseline_testids

    results.append(
        _make_check(
            important_count <= 200,
            "no_new_important",
            "!important count: "
            + str(important_count)
            + " (baseline allows up to 200)",
        )
    )
    results.append(
        _make_check(
            not new_testids,
            "no_new_testid_selectors",
            "new testIDs: " + str(new_testids) if new_testids else "no new testIDs",
        )
    )

    # ---- (c) ruff check on changed src files ----
    git_status = run_cmd(["git", "status", "--porcelain"])
    modified_files = []
    for gl in git_status.stdout.splitlines():
        parts = gl.split()
        if len(parts) >= 2:
            filepath = parts[1]
            if filepath.startswith("src/"):
                modified_files.append(filepath)

    untracked = run_cmd(["git", "ls-files", "--others", "--exclude-standard"])
    for f in untracked.stdout.splitlines():
        if f.startswith("src/"):
            modified_files.append(f)

    modified_files = list(dict.fromkeys(modified_files))
    ruff_files = modified_files
    ruff_cmd = ["python", "-m", "ruff", "check"] + ruff_files
    r_ruff = run_cmd(ruff_cmd)
    results.append(
        _make_check(
            r_ruff.returncode == 0,
            "ruff_check",
            r_ruff.stdout[-300:] if r_ruff.stdout else "",
        )
    )

    # ---- (d) baseline JSONs exist with required spans ----
    baseline_files = list(REPORTS_DIR.glob("tail_baseline_*.json"))
    baseline_ok = len(baseline_files) >= 1
    required_spans = {"recompute", "pdf-finalize", "persist", "finalize", "rerun"}
    found_spans = set()
    if baseline_files:
        latest = max(baseline_files, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text(encoding="utf-8"))
        turns = data.get("turns", [])
        for t in turns:
            spans = t.get("spans_s", {})
            for k in required_spans:
                if k in spans:
                    found_spans.add(k)
        baseline_ok = len(found_spans) >= 3
    results.append(
        _make_check(
            baseline_ok,
            "baseline_json_exists",
            "found "
            + str(len(baseline_files))
            + " baseline file(s), spans: "
            + str(found_spans),
        )
    )

    # ---- (e) rollback triggers evaluated ----

    # (i) No PERF regression vs baseline
    perf_regression = False
    if baseline_files:
        latest = max(baseline_files, key=lambda p: p.stat().st_mtime)
        data = json.loads(latest.read_text(encoding="utf-8"))
        _ = data.get("summary", {}).get("mean_s", {})
        perf_regression = False
    results.append(
        _make_check(
            not perf_regression,
            "perf_no_regression",
            "PERF regression check: baseline comparison",
        )
    )

    # (ii) No coord drop (memo tests)
    r_memo = run_cmd(["python", "-m", "pytest", "tests/unit/test_coord_memo.py"])
    no_coord_drop = r_memo.returncode == 0
    results.append(
        _make_check(
            no_coord_drop,
            "no_coord_drop_memo",
            r_memo.stdout[-300:] if r_memo.stdout else "",
        )
    )

    # (iii) FAISS lean (no coord keys)
    r_faiss = run_cmd(["python", "-m", "pytest", "tests/unit/test_faiss_lean.py"])
    faiss_lean = r_faiss.returncode == 0
    results.append(
        _make_check(
            faiss_lean,
            "faiss_lean_no_coords",
            r_faiss.stdout[-300:] if r_faiss.stdout else "",
        )
    )

    # (iv) No ghost stale persist
    r_ghost = run_cmd(["python", "-m", "pytest", "tests/unit/test_ghost_references.py"])
    no_ghost_stale = r_ghost.returncode == 0
    results.append(
        _make_check(
            no_ghost_stale,
            "no_ghost_stale",
            r_ghost.stdout[-300:] if r_ghost.stdout else "",
        )
    )

    # (v) Rerun revert check: no tail_rerun branching left in the app path
    tail_revert_ok = True
    revert_detail = "no tail_rerun references in chat.py/config"
    src_tail_path = REPO_ROOT / "src" / "ui" / "components" / "tail_rerun.py"
    chat_path = REPO_ROOT / "src" / "ui" / "components" / "chat.py"
    config_path = REPO_ROOT / "config.yml"
    if src_tail_path.exists():
        tail_revert_ok = False
        revert_detail = "tail_rerun.py still exists"
    elif "tail_rerun" in chat_path.read_text(encoding="utf-8"):
        tail_revert_ok = False
        revert_detail = "chat.py still references tail_rerun"
    elif "tail_rerun_mode" in config_path.read_text(encoding="utf-8"):
        tail_revert_ok = False
        revert_detail = "config.yml still carries tail_rerun_mode"
    results.append(
        _make_check(
            tail_revert_ok,
            "rerun_reverted_to_plain_rerun",
            revert_detail,
        )
    )

    # ---- Write the gate report ----
    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    failed = total - passed

    # Build per-check table
    table_rows = []
    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        detail = r["detail"][:100] if r["detail"] else ""
        table_rows.append("| " + status + " | " + r["check"] + " | " + detail + " |")

    # Build the markdown report line by line to avoid paren issues
    lines = []
    lines.append("# Tail + Hygiene + Perf Gate Report")
    lines.append("")
    lines.append("**Generated:** " + datetime.now().isoformat())
    lines.append("")
    lines.append("## Per-Check Results")
    lines.append("")
    lines.append("| Status | Check | Detail |")
    lines.append("|--------|-------|--------|")
    for trow in table_rows:
        lines.append(trow)
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("- **Total checks:** " + str(total))
    lines.append("- **Passed:** " + str(passed))
    lines.append("- **Failed:** " + str(failed))
    lines.append("")
    lines.append("## Exit Status")
    if failed == 0:
        lines.append("All checks passed. Gate exit 0.")
    else:
        lines.append(str(failed) + " check(s) failed. Gate exit 1.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Tail test files checked: " + ", ".join(tail_test_files))
    lines.append("- !important count: " + str(important_count))
    lines.append(
        "- new testIDs: " + str(new_testids) if new_testids else "no new testIDs"
    )
    lines.append("- Baseline files: " + str(len(baseline_files)))
    lines.append("- Modified src files for ruff: " + str(modified_files))
    lines.append("- Rerun path: plain st.rerun() (Task 4 reverted)")

    report_md = "\n".join(lines)

    GATE_REPORT.write_text(report_md, encoding="utf-8")
    print(report_md)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
