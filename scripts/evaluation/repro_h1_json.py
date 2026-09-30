"""EXTERNAL repro for H1 Strict-JSON failure (structured_output v2.0).

No edits to src/ or config.yml. Stdlib + yaml only. No Ollama call required.

What it does:
  1. Loads rag.prompts.prompt_templates (version 2.0) from config.yml.
  2. Builds 7 adversarial contexts (quotes / newlines / repeated facts /
     version numbers / tabs / hash-like doc ids).
  3. Simulates a naive small-LLM JSON emission: interpolates context spans
     VERBATIM into the JSON schema (i.e. without applying rule 10 Strict JSON
     Escaping) -> strict json.loads() expected to fail on quote/newline cases.
  4. Measures: parse-failure rate, echo (5-gram repeat) rate, citation SSoT
     violation rate (inline [doc:N] tag inside final_answer).
  5. Prints a run report (stdout) and optionally writes reports/repro_h1_*.md.

Reference (read-only, not imported):
  - config.yml lines 263-333 (prompt_templates structured_output v2.0)
  - src/core/graph/_generate.py format_context + json.loads section
  - src/core/graph/_graph_core.py _repair_json
  - scripts/evaluation/eval_quality.py SCORE_PROMPT usage
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML (yaml) is required: pip install pyyaml", file=sys.stderr)
    raise SystemExit(2)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.yml"

INLINE_DOC_TAG = re.compile(r"\[doc:\s*[^\]]+\]")


# ---------------------------------------------------------------------------
# 1. Load template from config (read-only)
# ---------------------------------------------------------------------------
def load_template() -> tuple[str, str, dict]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        pt = cfg["rag"]["prompts"]["prompt_templates"]
    except KeyError as e:
        raise SystemExit(f"config path rag.prompts.prompt_templates missing: {e}")
    version = str(pt.get("version", "?"))
    template = pt.get("structured_output", "")
    return version, template, pt


# ---------------------------------------------------------------------------
# 2. Adversarial cases (7)
# ---------------------------------------------------------------------------
def adversarial_cases() -> list[dict]:
    """Each case: id, query, span (raw context text), section, echo_demo, ssoT_demo."""
    return [
        {
            "id": "Q1-quotes-causal-masking",
            "query": 'CM3는 어떤 "causal masking" 방식을 사용하나?',
            "span": 'CM3은 "causal masking"을 사용한다',
            "section": "모델 구조",
            "note": "raw double-quotes inside value (rule 10 violation)",
        },
        {
            "id": "Q2-version-numbers",
            "query": "비교 대상 모델들의 파라미터 수 차이는?",
            "span": 'DeepSeek-R1은 671B이며 논문은 "v2.0 (2026-01-15)" 개정판을 인용한다',
            "section": "비교",
            "note": "quotes + parens + version/date tokens",
        },
        {
            "id": "Q3-raw-newline",
            "query": "학습 절차를 요약하라.",
            "span": "1단계: 수집\n2단계: 정제\n3단계: 학습",  # literal newlines
            "section": "학습 절차",
            "note": "literal newline inside JSON string value",
        },
        {
            "id": "Q4-echo-repeat",
            "query": "MoE 파라미터 수를 답하라.",
            "span": "DeepSeek-R1은 671B 파라미터를 가진 MoE 모델입니다",
            "section": "모델 구조",
            "note": "repeated fact; echo demo repeats claim 3x (rule 11 violation)",
            "echo": True,
        },
        {
            "id": "Q5-nested-quotes-ko-en",
            "query": '"FlashRank"의 한국어 설명은?',
            "span": '리랭커 "FlashRank"는 "cross-encoder"이며 저자는 "매우 빠르다"고 말한다',
            "section": "리랭킹",
            "note": "multiple raw quote pairs",
        },
        {
            "id": "Q6-tab-backslash",
            "query": "아티팩트 경로는?",
            "span": "결과는\tC:\\models\\qwen3\\out.json 에 저장된다",
            "section": "실험 설정",
            "note": "raw tab + backslashes inside value",
        },
        {
            "id": "Q7-docid-ssot",
            "query": "671B 주장의 근거는?",
            "span": "671B 파라미터를 가진 MoE 모델입니다",
            "section": "모델 구조",
            "note": "final_answer embeds inline [doc:0] tag -> SSoT violation",
            "inline_tag": True,
        },
    ]


# ---------------------------------------------------------------------------
# Helpers: format_context mirror (no src import), naive emission, metrics
# ---------------------------------------------------------------------------
def format_context_mirror(case: dict, idx: int) -> str:
    # Mirrors src/core/graph/_generate.py format_context token layout:
    # [doc:<id>] [section:X] [page:Y] [score:Z]\n<content>
    return (
        f"[doc:{idx}] [section:{case['section']}] [page:3] [score:0.950]\n"
        f"{case['span']}\n\n"
    )


def naive_llm_emit(case: dict, idx: int) -> str:
    """Simulate a naive small model copying the span VERBATIM into JSON.

    Deliberately skips rule-10 escaping so strict json.loads fails on
    quote/newline/tab cases -- this is the H1 failure being reproduced.
    """
    span = case["span"]
    if case.get("echo"):
        answer = f"{span}. {span}. {span}."  # rule-11 echo violation demo
    elif case.get("inline_tag"):
        answer = f"{span} [doc:{idx}]."  # SSoT violation demo
    else:
        answer = f"{span}." if not span.endswith(".") else span
    # NOTE: raw f-string interpolation, no json.dumps escaping on values.
    return (
        '{"final_answer": "'
        + answer
        + '", "citations": [{"doc_id": '
        + str(idx)
        + ', "text_span": "'
        + span
        + '", "section": "'
        + case["section"]
        + '", "page": 3, "score": 0.95}], "confidence": 0.9}'
    )


def five_gram_echo_rate(text: str, n: int = 5) -> tuple[float, int, int]:
    toks = text.split()
    if len(toks) < n:
        return 0.0, 0, 0
    grams = [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    counts = Counter(grams)
    repeated = sum(1 for g in grams if counts[g] > 1)
    return repeated / len(grams), repeated, len(grams)


def try_strict_parse(s: str) -> tuple[bool, str]:
    try:
        obj = json.loads(s)
        assert isinstance(obj, dict) and "final_answer" in obj
        return True, ""
    except Exception as e:  # noqa: BLE001 - repro harness, report any parse error
        return False, f"{type(e).__name__}: {e}"


def check_required_rules(template: str) -> dict[str, bool]:
    return {
        "rule10_strict_escaping": "Strict JSON Escaping" in template,
        "rule11_no_repetition": "No Repetition" in template,
        "ssot_citations_only": "citations[] 배열이 유일한" in template
        or "Single Source of Truth" in template,
        "no_inline_doc_tag_rule": "직접 삽입하지 마십시오" in template,
        "json_schema_present": '"final_answer"' in template
        and '"citations"' in template,
    }


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="H1 Strict-JSON external repro")
    ap.add_argument("--report", default="", help="write markdown report to this path")
    args = ap.parse_args()

    t0 = time.perf_counter()
    version, template, _ = load_template()
    cases = adversarial_cases()

    rule_checks = check_required_rules(template)
    rows: list[dict] = []
    for idx, case in enumerate(cases):
        ctx = format_context_mirror(case, idx)
        # Template substitution check (mirrors _generate.py safe_prompt.format)
        try:
            rendered = template.replace("{context}", ctx).replace(
                "{query}", case["query"]
            )
            render_ok = "{context}" not in rendered and "{query}" not in rendered
        except Exception as e:  # noqa: BLE001
            rendered = ""
            render_ok = False
            print(f"WARN template render failed: {e}", file=sys.stderr)

        emitted = naive_llm_emit(case, idx)
        ok, err = try_strict_parse(emitted)

        # Metrics on the *emitted final_answer* (extract naively for echo/SSoT
        # even when strict parse fails: cut between markers).
        m = re.search(r'"final_answer":\s*"(.*)"\s*,\s*"citations"', emitted, re.S)
        answer_text = m.group(1) if m else emitted
        echo_rate, echo_rep, echo_tot = five_gram_echo_rate(answer_text)
        ssot_violation = bool(INLINE_DOC_TAG.search(answer_text))

        rows.append(
            {
                "id": case["id"],
                "note": case["note"],
                "render_ok": render_ok,
                "parse_ok": ok,
                "parse_err": err,
                "echo_rate": echo_rate,
                "echo_rep": echo_rep,
                "echo_tot": echo_tot,
                "ssot_violation": ssot_violation,
                "emitted": emitted,
            }
        )

    n = len(rows)
    n_fail = sum(1 for r in rows if not r["parse_ok"])
    parse_fail_rate = n_fail / n if n else 0.0
    n_echo = sum(1 for r in rows if r["echo_rate"] > 0)
    echo_rate_cases = n_echo / n if n else 0.0
    n_ssot = sum(1 for r in rows if r["ssot_violation"])
    ssot_rate = n_ssot / n if n else 0.0
    elapsed = time.perf_counter() - t0

    # ---- stdout report ----
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(
        "H1 Strict-JSON repro (EXTERNAL, no src/ edits, stdlib+yaml, no Ollama)"
    )
    lines.append("=" * 72)
    lines.append(f"config: {CONFIG_PATH}")
    lines.append(f"prompt_templates.version: {version} (expected 2.0)")
    lines.append("rules present: " + json.dumps(rule_checks, ensure_ascii=False))
    lines.append(
        f"SCORE_PROMPT ref: scripts/evaluation/eval_quality.py (faithfulness judge)"
    )
    lines.append(f"format_context ref: src/core/graph/_generate.py::format_context")
    lines.append(f"cases: {n} adversarial (quotes/newlines/repeats/versions/tabs/ssot)")
    lines.append("-" * 72)
    for r in rows:
        status = "PARSE-OK " if r["parse_ok"] else "PARSE-FAIL"
        lines.append(f"[{status}] {r['id']}")
        lines.append(f"  note : {r['note']}")
        if not r["parse_ok"]:
            lines.append(f"  error: {r['parse_err'][:160]}")
        lines.append(
            f"  echo : 5-gram repeat {r['echo_rep']}/{r['echo_tot']} "
            f"(rate={r['echo_rate']:.3f})"
        )
        lines.append(
            f"  ssot : {'VIOLATION [doc:N] inline' if r['ssot_violation'] else 'ok (citations[] only)'}"
        )
    lines.append("-" * 72)
    lines.append(f"parse-failure rate : {n_fail}/{n} = {parse_fail_rate:.1%}")
    lines.append(
        f"echo-case rate     : {n_echo}/{n} = {echo_rate_cases:.1%} (5-gram repeat > 0)"
    )
    lines.append(
        f"ssot-violation rate: {n_ssot}/{n} = {ssot_rate:.1%} (inline [doc:N] in final_answer)"
    )
    lines.append(f"elapsed: {elapsed:.2f}s (budget <60s)")
    lines.append(
        "conclusion: "
        + (
            "H1 REPRODUCED (strict json.loads fails on naive "
            "quote/newline emission under v2.0 template)"
            if n_fail
            else "H1 NOT reproduced"
        )
    )
    report = "\n".join(lines)
    print(report, flush=True)

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        md = [
            "# H1 Strict-JSON repro report",
            "",
            f"- Timestamp (UTC): {ts}",
            f"- Config: `{CONFIG_PATH}`",
            f"- prompt_templates.version: `{version}` (expected `2.0`)",
            f"- Rules present: `{json.dumps(rule_checks, ensure_ascii=False)}`",
            "- Refs (read-only):",
            "  - `config.yml` prompt_templates structured_output v2.0 (§263-333)",
            "  - `src/core/graph/_generate.py` format_context + json.loads + _repair_json fallback",
            "  - `scripts/evaluation/eval_quality.py` SCORE_PROMPT (faithfulness judge)",
            "",
            "## Metrics",
            "",
            f"- parse-failure rate: **{n_fail}/{n} = {parse_fail_rate:.1%}**",
            f"- echo-case rate (5-gram repeat > 0): **{n_echo}/{n} = {echo_rate_cases:.1%}**",
            f"- SSoT-violation rate (inline `[doc:N]` in final_answer): **{n_ssot}/{n} = {ssot_rate:.1%}**",
            f"- elapsed: {elapsed:.2f}s (budget <60s)",
            "",
            "## Per-case results",
            "",
            "| case | strict parse | 5-gram echo | SSoT | note |",
            "|---|---|---|---|---|",
        ]
        for r in rows:
            md.append(
                f"| `{r['id']}` | {'OK' if r['parse_ok'] else 'FAIL: ' + r['parse_err'][:80]} "
                f"| {r['echo_rep']}/{r['echo_tot']} ({r['echo_rate']:.2f}) "
                f"| {'VIOLATION' if r['ssot_violation'] else 'ok'} | {r['note']} |"
            )
        md += [
            "",
            "## Interpretation",
            "",
            "Naive small-LLM emission copies adversarial spans verbatim (no rule-10 "
            'escaping), so strict `json.loads` fails on raw `"` / literal newline / '
            "tab cases — the H1 Strict-JSON failure. `_repair_json` in "
            "`src/core/graph/_graph_core.py` exists as a fallback, but the repro shows "
            "the raw emission itself violates v2.0 §10 before repair. Echo (rule 11) is "
            "demonstrated by Q4 (same 5-gram repeated 3x); SSoT violation by Q7 "
            "(inline `[doc:N]` inside `final_answer` instead of citations[] only).",
            "",
            "## Repro command",
            "",
            "```powershell",
            "python scripts/evaluation/repro_h1_json.py --report reports/repro_h1_<ts>.md",
            "```",
            "",
        ]
        out.write_text("\n".join(md), encoding="utf-8")
        print(f"[repro] markdown report written: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
