"""Offline evaluation harness for real-usage Q&A traces (``.jsonl``).

Reads ``logs/qa_trace.jsonl`` (written by ``common.qa_trace``), joins each
query with ground truth from ``tests/data/testset_2201.csv`` when available,
optionally scores answers with the configured judge model, then writes a JSON
and Markdown report under ``reports/``.

Usage:
  python scripts/evaluation/eval_qa_trace.py --no-judge
  python scripts/evaluation/eval_qa_trace.py --limit 50
  python scripts/evaluation/eval_qa_trace.py --exclude-cached --intent rag
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.append(_path)

# ruff: noqa: E402 - sys.path 부트스트랩 이후 임포트 (scripts 표준 패턴)

TRACE_PATH_DEFAULT = ROOT_DIR / "logs" / "qa_trace.jsonl"
TESTSET_PATH_DEFAULT = ROOT_DIR / "tests" / "data" / "testset_2201.csv"
REPORTS_DIR_DEFAULT = ROOT_DIR / "reports"

SCORE_PROMPT = """
당신은 RAG 시스템 평가 전문가입니다. 아래 [질문], [정답], [모델 답변], [Context]를 비교하여 답변의 품질과 근거성을 1점에서 5점 사이로 평가하세요.

[질문]: {question}
[정답]: {ground_truth}
[모델 답변]: {answer}
[Context]: {context}

[평가 기준]:
- 5점: 답변이 정답과 의미적으로 완벽하게 일치하며 정확하고, 답변의 모든 주장이 제공된 Context에 근거함.
- 3점: 답변의 핵심은 맞지만 세부 정보가 부족하거나 약간의 노이즈가 있음.
- 1점: 답변이 틀렸거나 질문과 관련이 없음.
- 근거성(Faithfulness): 답변의 모든 주장이 제공된 Context에 근거하는지 함께 평가하세요. Context에 없는 주장을 포함하면 감점합니다.

결과는 반드시 숫자 하나(예: 5)만 출력하세요. 설명은 생략하세요.
"""


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def load_trace(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Load JSONL records. Returns ``(records, malformed_count)``."""
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                malformed += 1
    return records, malformed


def load_testset(path: Path) -> dict[str, dict[str, str]]:
    """Map normalized question -> ground_truth/reference_context."""
    mapping: dict[str, dict[str, str]] = {}
    if not path.exists():
        return mapping
    # utf-8-sig: tests/data/testset_2201.csv 는 BOM을 포함한다.
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            mapping[_normalize(question)] = {
                "ground_truth": (row.get("ground_truth") or "").strip(),
                "reference_context": (row.get("reference_context") or "").strip(),
            }
    return mapping


def filter_records(
    records: list[dict[str, Any]],
    *,
    limit: int = 0,
    exclude_cached: bool = False,
    intent: str = "",
) -> list[dict[str, Any]]:
    """Apply filters; ``limit > 0`` keeps the most recent N records."""
    selected = []
    for record in records:
        if exclude_cached and bool(record.get("is_cached")):
            continue
        if intent and str(record.get("intent") or "") != intent:
            continue
        selected.append(record)
    if limit > 0:
        selected = selected[-limit:]
    return selected


def attach_ground_truth(
    records: list[dict[str, Any]], testset: dict[str, dict[str, str]]
) -> int:
    """Attach ground truth in place. Returns joined count."""
    joined = 0
    for record in records:
        entry = testset.get(_normalize(str(record.get("query") or "")))
        if entry:
            record["ground_truth"] = entry["ground_truth"]
            record["reference_context"] = entry["reference_context"]
            joined += 1
        else:
            record["ground_truth"] = ""
            record["reference_context"] = ""
    return joined


def parse_score(text: str) -> int | None:
    match = re.search(r"[1-5]", text or "")
    return int(match.group()) if match else None


async def judge_records(
    records: list[dict[str, Any]],
    *,
    model_name: str,
    max_context_chars: int = 4000,
    max_answer_chars: int = 4000,
) -> None:
    """Score records in place with the judge model (sequential, local-safe)."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from core.model_loader import ModelManager

    # NOTE: temperature 는 모델 생성 시점 설정을 사용한다. bind(temperature=...)
    # 는 langchain_ollama 의 AsyncClient.chat() 에 전달되어 TypeError 를 낸다.
    llm = await ModelManager.get_llm(model_name)
    chain = ChatPromptTemplate.from_template(SCORE_PROMPT) | llm | StrOutputParser()

    for record in records:
        ground_truth = str(record.get("ground_truth") or "").strip()
        if not ground_truth:
            record["judge"] = None
            record["judge_excluded"] = True
            record["judge_error"] = "no ground_truth (join miss)"
            continue
        context = str(record.get("reference_context") or "").strip()
        if max_context_chars > 0:
            context = context[:max_context_chars]
        answer = str(record.get("answer") or "")
        if max_answer_chars > 0:
            answer = answer[:max_answer_chars]
        try:
            raw = str(
                await chain.ainvoke(
                    {
                        "question": str(record.get("query") or ""),
                        "ground_truth": ground_truth,
                        "answer": answer,
                        "context": context or "일상적인 대화입니다.",
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001 - per-record isolation
            record["judge"] = None
            record["judge_excluded"] = True
            record["judge_error"] = f"{type(exc).__name__}: {exc}"[:200]
            continue
        score = parse_score(raw)
        record["judge"] = score
        record["judge_raw"] = raw[:200]
        record["judge_excluded"] = score is None
        record["judge_error"] = "" if score is not None else f"parse: {raw[:200]}"


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _group_avg(records: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    groups: dict[str, list[float]] = {}
    for record in records:
        score = record.get("judge")
        if not isinstance(score, int):
            continue
        label = str(record.get(key) or "")
        groups.setdefault(label, []).append(float(score))
    return {label: _mean(values) for label, values in sorted(groups.items())}


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [int(r["judge"]) for r in records if isinstance(r.get("judge"), int)]
    answer_lengths = [len(str(r.get("answer") or "")) for r in records]
    cached_scores = [
        int(r["judge"])
        for r in records
        if isinstance(r.get("judge"), int) and bool(r.get("is_cached"))
    ]
    uncached_scores = [
        int(r["judge"])
        for r in records
        if isinstance(r.get("judge"), int) and not bool(r.get("is_cached"))
    ]
    return {
        "record_count": len(records),
        "judged_count": len(scores),
        "judge_avg": _mean([float(s) for s in scores]),
        "score_distribution": {str(i): scores.count(i) for i in range(1, 6)},
        "excluded_count": sum(1 for r in records if r.get("judge_excluded")),
        "parse_failed_count": sum(1 for r in records if r.get("parse_failed")),
        "echo_detected_count": sum(1 for r in records if r.get("echo_detected")),
        "over_budget_count": sum(1 for r in records if r.get("over_budget")),
        "cached_count": sum(1 for r in records if r.get("is_cached")),
        "avg_answer_chars": _mean([float(n) for n in answer_lengths]),
        "cached_judge_avg": _mean([float(s) for s in cached_scores]),
        "uncached_judge_avg": _mean([float(s) for s in uncached_scores]),
        "by_source": _group_avg(records, "source"),
        "by_intent": _group_avg(records, "intent"),
    }


def _escape_cell(value: Any, limit: int = 80) -> str:
    text = str(value if value is not None else "").replace("|", "\\|")
    text = " ".join(text.split())
    return text[:limit]


def render_markdown(
    meta: dict[str, Any],
    agg: dict[str, Any],
    records: list[dict[str, Any]],
) -> str:
    lines = [
        "# QA Trace Evaluation",
        "",
        f"- trace: `{meta['trace_path']}`",
        f"- generated_at: {meta['generated_at']}",
        f"- judge: {meta['judge']} (model=`{meta['judge_model']}`)",
        f"- records: {agg['record_count']} "
        f"(joined={meta['joined']}, malformed_skipped={meta['malformed']})",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "|--------|-------|",
        f"| judged_count | {agg['judged_count']} |",
        f"| judge_avg (1-5) | {agg['judge_avg']} |",
        f"| cached_judge_avg | {agg['cached_judge_avg']} |",
        f"| uncached_judge_avg | {agg['uncached_judge_avg']} |",
        f"| excluded_count | {agg['excluded_count']} |",
        f"| parse_failed / echo / over_budget | {agg['parse_failed_count']} / "
        f"{agg['echo_detected_count']} / {agg['over_budget_count']} |",
        f"| cached_count | {agg['cached_count']} |",
        f"| avg_answer_chars | {agg['avg_answer_chars']} |",
        f"| score_distribution | {agg['score_distribution']} |",
        "",
        "## Per record",
        "",
        "| # | source | intent | cached | chars | judge | error | query |",
        "|---|--------|--------|--------|-------|-------|-------|-------|",
    ]
    for index, record in enumerate(records, start=1):
        lines.append(
            f"| {index} | {_escape_cell(record.get('source'))} "
            f"| {_escape_cell(record.get('intent'))} "
            f"| {bool(record.get('is_cached'))} "
            f"| {len(str(record.get('answer') or ''))} "
            f"| {record.get('judge') if record.get('judge') is not None else ''} "
            f"| {_escape_cell(record.get('judge_error'), 40)} "
            f"| {_escape_cell(record.get('query'), 60)} |"
        )
    return "\n".join(lines) + "\n"


def write_report(
    records: list[dict[str, Any]],
    agg: dict[str, Any],
    meta: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"eval_qa_trace_{stamp}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")
    payload = {"meta": meta, "aggregate": agg, "records": records}
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(render_markdown(meta, agg, records), encoding="utf-8")
    return json_path, md_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default=str(TRACE_PATH_DEFAULT))
    parser.add_argument("--testset", default=str(TESTSET_PATH_DEFAULT))
    parser.add_argument("--output-dir", default=str(REPORTS_DIR_DEFAULT))
    parser.add_argument(
        "--limit", type=int, default=0, help="most recent N records (0 = all)"
    )
    parser.add_argument("--exclude-cached", action="store_true")
    parser.add_argument("--intent", default="", help="only this intent (e.g. rag)")
    parser.add_argument("--no-judge", action="store_true", help="skip LLM scoring")
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--max-context-chars", type=int, default=4000)
    parser.add_argument("--max-answer-chars", type=int, default=4000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trace_path = Path(args.trace).resolve()
    if not trace_path.exists():
        print(f"[eval_qa_trace] trace not found: {trace_path}", file=sys.stderr)
        return 2

    records, malformed = load_trace(trace_path)
    testset = load_testset(Path(args.testset).resolve())
    records = filter_records(
        records,
        limit=args.limit,
        exclude_cached=args.exclude_cached,
        intent=args.intent,
    )
    if not records:
        print("[eval_qa_trace] no records after filtering", file=sys.stderr)
        return 1

    joined = attach_ground_truth(records, testset)
    judge_model = args.judge_model
    if not judge_model:
        from common.config import EVAL_JUDGE_MODEL

        judge_model = EVAL_JUDGE_MODEL

    if args.no_judge:
        for record in records:
            record["judge"] = None
            record["judge_excluded"] = True
            record["judge_error"] = "--no-judge"
    else:
        asyncio.run(
            judge_records(
                records,
                model_name=judge_model,
                max_context_chars=args.max_context_chars,
                max_answer_chars=args.max_answer_chars,
            )
        )

    agg = aggregate(records)
    meta = {
        "trace_path": str(trace_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "judge": "off" if args.no_judge else "on",
        "judge_model": judge_model,
        "joined": joined,
        "malformed": malformed,
        "limit": args.limit,
        "exclude_cached": args.exclude_cached,
        "intent": args.intent,
    }
    json_path, md_path = write_report(
        records, agg, meta, Path(args.output_dir).resolve()
    )
    print(
        f"[eval_qa_trace] records={agg['record_count']} "
        f"judged={agg['judged_count']} avg={agg['judge_avg']} -> {json_path}"
    )
    print(f"[eval_qa_trace] markdown -> {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
