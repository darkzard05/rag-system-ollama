"""RAG query latency benchmark.

Measures query TTFT (time-to-first-token) across THREE explicit scenarios and
writes a JSON report. This is the measurement backbone for the pipeline
optimization plan (query cache, grade reduction, prewarm): it produces the
real before/after numbers the optimization tasks must beat.

Scenarios
---------
S1 cold  : fresh process, query cache OFF, FIRST query only
           (model load + full LLM path).
S2 warm  : same process, repeat the fixed query set `--repeat N` times
           (steady-state, no cache).
S3 cache : cache ON, first issuance (miss, full path) then SECOND issuance
           of each query (expected HIT -> no LLM calls).

TTFT is taken from the `[QUERY][TIMING] ttft_ms=...` log line emitted by
`src/core/graph_builder.py`. A logging handler captures those lines; we also
parse the per-stage breakdown (preprocess/retrieve/grade/generate) for
`per_stage_ms`.

NOTE on S3 / cache wiring
-------------------------
The query cache is a LATER task and is NOT yet wired. Until then S3 behaves
identically to S2 (every issuance hits the full LLM path), so:
  * `s3_repeat_llm_calls` is recorded as the *measured* value (may be > 0
    pre-wiring). The script does NOT fail when it is non-zero.
  * The strict "cache HIT => 0 LLM calls" assertion is enforced by the
    regression guard in a later task, not here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

from src.common.config import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)
from src.common.logging_config import setup_logging  # noqa: E402
from src.core import graph_builder as _graph_builder  # noqa: E402
from src.core.document_processor import compute_file_hash  # noqa: E402
from src.core.model_loader import ModelManager  # noqa: E402
from src.core.rag_core import RAGSystem  # noqa: E402
from src.core.session import SessionManager  # noqa: E402

# Fixed, diverse question set about the CM3 paper (tests/data/2201.07520v1.pdf).
# Mirrors the list in scripts/evaluation/evaluate_diverse_questions.py.
FIXED_QUESTIONS: list[str] = [
    "CM3 모델의 Medium과 Large 버전은 각각 몇 개의 파라미터를 가지고 있나요?",
    "Causally Masked Language Modeling이 양방향 맥락(Bidirectional context)을 제공하는 원리가 무엇인가요?",
    "CM3 모델이 제로샷(Zero-shot)으로 수행할 수 있는 작업들을 모두 나열해 주세요.",
    "Entity Linking 작업에서 CM3 모델은 기존 SOTA와 비교해 어떤 성과를 냈나요?",
    "CM3가 이미지 인필링(Image In-filling)을 수행할 때 사용하는 프롬프트 구조는 어떤 식인가요?",
]

PDF_PATH = ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf"
REPORTS_DIR = ROOT_DIR / "reports"

_TIMING_RE = re.compile(
    r"preprocess_ms=(?P<preprocess_ms>[\d.]+)\s+"
    r"retrieve_ms=(?P<retrieve_ms>[\d.]+)\s+"
    r"grade_ms=(?P<grade_ms>[\d.]+)\s+"
    r"generate_total_ms=(?P<generate_total_ms>[\d.]+)\s+"
    r"ttft_ms=(?P<ttft_ms>[\d.]+)"
)


class _TimingCaptureHandler(logging.Handler):
    """Captures `[QUERY][TIMING]` log records and parses per-stage ms."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[dict[str, float]] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        if "[QUERY][TIMING]" not in msg:
            return
        match = _TIMING_RE.search(msg)
        if not match:
            return
        self.records.append(
            {
                "preprocess_ms": float(match.group("preprocess_ms")),
                "retrieve_ms": float(match.group("retrieve_ms")),
                "grade_ms": float(match.group("grade_ms")),
                "generate_total_ms": float(match.group("generate_total_ms")),
                "ttft_ms": float(match.group("ttft_ms")),
            }
        )


class _LLMCallCounter:
    """Counts LLM generation invocations (astream / ainvoke) on a model."""

    def __init__(self) -> None:
        self.calls = 0


def _patch_llm_class(cls: type[Any], counter: _LLMCallCounter) -> None:
    """Patch a LLM CLASS's astream/ainvoke to increment `counter`.

    ``DeepThinkingChatOllama`` is a pydantic v2 model, so assigning a wrapped
    method directly on an *instance* raises ``ValueError`` ("has no field");
    patching the class is required. We patch the ROOT ``BaseChatModel`` in the
    MRO (not the concrete subclass) because an intervening parent
    (``ChatOllama``/``Runnable``) redefines ``astream`` and would shadow a
    subclass-level patch — verified: patching the subclass yields 0 calls while
    patching the root yields the real count.

    Patching the class attribute affects every instance globally — acceptable
    here because this runs only inside the benchmark process.
    """
    # Walk the MRO to the nearest ancestor that actually defines `astream`.
    target = cls
    for base in cls.__mro__:
        if "astream" in getattr(base, "__dict__", {}):
            target = base
            break
    astream_orig = target.astream  # type: ignore[arg-type]
    ainvoke_orig = getattr(target, "ainvoke", None)

    async def _wrapped_astream(self: object, *args: object, **kwargs: object) -> object:
        counter.calls += 1
        async for chunk in astream_orig(self, *args, **kwargs):  # type: ignore[operator]
            yield chunk

    target.astream = _wrapped_astream

    if ainvoke_orig is not None:

        async def _wrapped_ainvoke(
            self: object, *args: object, **kwargs: object
        ) -> object:
            counter.calls += 1
            return await ainvoke_orig(self, *args, **kwargs)  # type: ignore[operator]

        target.ainvoke = _wrapped_ainvoke


def _install_llm_counter(llm: object | None) -> _LLMCallCounter:
    """Install an LLM-generation call counter (benchmark process only).

    The LLM ``generate`` streams from is the ``DeepThinkingChatOllama`` instance
    cached in the resource pool and reused across queries; the instance stored in
    SessionManager may be None at install time, so we patch the concrete class
    directly (via the root ``astream`` in its MRO) AND monkeypatch
    ``ModelManager.get_llm`` to wrap any later fetch (model switch / pool
    eviction) as a belt-and-braces safety net.

    Returns ONE shared counter for the whole process.

    ONLY called inside the benchmark process; never in product code.
    """
    counter = _LLMCallCounter()
    # (1) Patch the concrete LLM class directly (the type generate always uses).
    if llm is not None:
        _patch_llm_class(type(llm), counter)
    try:
        from src.core.custom_ollama import DeepThinkingChatOllama

        _patch_llm_class(DeepThinkingChatOllama, counter)
    except Exception:  # noqa: BLE001 - best-effort; get_llm patch covers the rest
        pass
    # (2) Safety net: patch get_llm so any future fetch is also wrapped.
    get_llm_orig = ModelManager.get_llm  # bound classmethod

    @classmethod  # type: ignore[misc]
    async def _wrapped_get_llm(cls: type, model_name: str, **kwargs: object) -> object:
        # get_llm_orig is already bound to ModelManager; do NOT pass cls.
        result = await get_llm_orig(model_name, **kwargs)  # type: ignore[operator]
        if result is not None:
            _patch_llm_class(type(result), counter)
        return result

    ModelManager.get_llm = _wrapped_get_llm  # type: ignore[assignment]
    return counter


def _p95(values: list[float]) -> float:
    """95th percentile (nearest-rank). Single value returns itself."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = math.ceil(0.95 * len(ordered)) - 1
    return ordered[rank]


def _avg_stage(records: list[dict[str, float]], stage: str) -> float:
    if not records:
        return 0.0
    return sum(r.get(stage, 0.0) for r in records) / len(records)


async def _run_query(
    rag: RAGSystem,
    prompt: str,
    timing_handler: _TimingCaptureHandler,
) -> float:
    """Run one query, return its TTFT in ms captured from the timing log."""
    before = len(timing_handler.records)
    await rag.aquery(prompt)
    new = timing_handler.records[before:]
    if new:
        return new[-1]["ttft_ms"]
    # Fallback: wall-clock around aquery if no timing line was emitted.
    return 0.0


async def _scenario_cold(
    rag: RAGSystem,
    timing_handler: _TimingCaptureHandler,
) -> tuple[float, list[float]]:
    """S1: fresh process, cache off, FIRST query only."""
    t = await _run_query(rag, FIXED_QUESTIONS[0], timing_handler)
    return t, [t]


async def _scenario_warm(
    rag: RAGSystem,
    timing_handler: _TimingCaptureHandler,
    repeat: int,
) -> tuple[float, list[float]]:
    """S2: same process, repeat the fixed query set `repeat` times."""
    ttfts: list[float] = []
    for _ in range(repeat):
        for q in FIXED_QUESTIONS:
            ttfts.append(await _run_query(rag, q, timing_handler))
    return _p95(ttfts), ttfts


async def _scenario_cache(
    rag: RAGSystem,
    timing_handler: _TimingCaptureHandler,
    llm_counter: _LLMCallCounter,
) -> tuple[float, float, int, list[float], list[float]]:
    """S3: cache ON — first issuance (miss) then second issuance (HIT)."""
    first_ttfts: list[float] = []
    for q in FIXED_QUESTIONS:
        first_ttfts.append(await _run_query(rag, q, timing_handler))

    llm_counter.calls = 0
    repeat_ttfts: list[float] = []
    for q in FIXED_QUESTIONS:
        repeat_ttfts.append(await _run_query(rag, q, timing_handler))
    repeat_llm_calls = llm_counter.calls

    return (
        _p95(first_ttfts),
        _p95(repeat_ttfts),
        repeat_llm_calls,
        first_ttfts,
        repeat_ttfts,
    )


class _RegressionError(Exception):
    """Raised when a compare-mode assertion is violated (CI regression guard)."""


def _fmt_diff_line(
    key: str, current: float, baseline: float, ok: bool, note: str = ""
) -> str:
    marker = "OK  " if ok else "FAIL"
    delta = current - baseline
    pct = (delta / baseline * 100.0) if baseline > 0 else 0.0
    tail = f" {note}".rstrip()
    return (
        f"  [{marker}] {key}: current={current:.1f} baseline={baseline:.1f} "
        f"({delta:+.1f}, {pct:+.1f}%){tail}"
    )


def _compare_reports(
    report: dict[str, object],
    baseline: dict[str, object],
    max_regression_pct: float,
) -> None:
    """Enforce the CI regression contract between a report and baseline.

    Assertions (per scenario):
      * S1 cold TTFT must not regress beyond ``max_regression_pct``.
      * S2 warm TTFT must not regress beyond ``max_regression_pct``.
      * EXPECTED GAINS (cache ON): S3 repeat LLM calls == 0 AND
        S3 repeat TTFT <= S2 warm TTFT (cache win).

    Prints a diff table; raises ``_RegressionError`` on the FIRST violation.
    """
    violations: list[str] = []
    guard = 1.0 + max_regression_pct / 100.0

    def num(key: str) -> float:
        val = report.get(key, baseline.get(key, 0.0))
        try:
            return float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0

    base_s1 = float(baseline.get("s1_cold_ttft_p95", 0.0))  # type: ignore[arg-type]
    base_s2 = float(baseline.get("s2_warm_ttft_p95", 0.0))  # type: ignore[arg-type]
    cur_s1 = num("s1_cold_ttft_p95")
    cur_s2 = num("s2_warm_ttft_p95")
    cur_s3_repeat_ttft = num("s3_repeat_ttft_p95")
    try:
        cur_s3_llm = int(
            report.get("s3_repeat_llm_calls", baseline.get("s3_repeat_llm_calls", 0))
        )  # type: ignore[arg-type]
    except (TypeError, ValueError):
        cur_s3_llm = 0

    print("\n[COMPARE] post-change report vs baseline")
    print("-" * 70)

    # S1 cold TTFT regression guard.
    ok_s1 = cur_s1 <= base_s1 * guard
    print(
        _fmt_diff_line(
            "s1_cold_ttft_p95", cur_s1, base_s1, ok_s1, f"limit<={base_s1 * guard:.1f}"
        )
    )
    if not ok_s1:
        violations.append(
            f"S1 cold TTFT regressed: {cur_s1:.1f} > limit {base_s1 * guard:.1f} "
            f"(baseline {base_s1:.1f}, +{max_regression_pct:.0f}% guard)"
        )

    # S2 warm TTFT regression guard.
    ok_s2 = cur_s2 <= base_s2 * guard
    print(
        _fmt_diff_line(
            "s2_warm_ttft_p95", cur_s2, base_s2, ok_s2, f"limit<={base_s2 * guard:.1f}"
        )
    )
    if not ok_s2:
        violations.append(
            f"S2 warm TTFT regressed: {cur_s2:.1f} > limit {base_s2 * guard:.1f} "
            f"(baseline {base_s2:.1f}, +{max_regression_pct:.0f}% guard)"
        )

    # EXPECTED GAINS — S3 cache win.
    ok_llm = cur_s3_llm == 0
    llm_marker = "OK  " if ok_llm else "FAIL"
    print(
        f"  [{llm_marker}] s3_repeat_llm_calls: "
        f"current={cur_s3_llm} expected=0 (cache HIT => no LLM calls)"
    )
    if not ok_llm:
        violations.append(
            f"EXPECTED GAIN not realized: s3_repeat_llm_calls={cur_s3_llm} != 0 "
            f"(query cache did not eliminate repeat LLM calls)"
        )

    ok_cache_win = cur_s3_repeat_ttft <= cur_s2
    print(
        _fmt_diff_line(
            "s3_repeat_ttft_p95",
            cur_s3_repeat_ttft,
            cur_s2,
            ok_cache_win,
            "expect<= S2 warm",
        )
    )
    if not ok_cache_win:
        violations.append(
            f"EXPECTED GAIN not realized: s3_repeat_ttft_p95={cur_s3_repeat_ttft:.1f} "
            f"> s2_warm_ttft_p95={cur_s2:.1f} (cache did not reduce repeat TTFT)"
        )

    print("-" * 70)
    if violations:
        msg = "CI regression guard FAILED:\n  - " + "\n  - ".join(violations)
        raise _RegressionError(msg)
    print(
        f"[OK] all assertions passed (max regression guard {max_regression_pct:.0f}%, "
        f"expected gains realized)"
    )


def _validate_baseline(
    report: dict[str, object], baseline_path: Path, max_regression_pct: float
) -> None:
    """Compare a report against a baseline JSON (regression guard)."""
    if not baseline_path.exists():
        print(f"[WARN] baseline not found: {baseline_path}", file=sys.stderr)
        return
    with open(baseline_path, encoding="utf-8") as f:
        baseline = json.load(f)
    _compare_reports(report, baseline, max_regression_pct)


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    setup_logging(log_level="INFO")

    timing_handler = _TimingCaptureHandler()
    # Attach to the ROOT logger so we receive the [QUERY][TIMING] record
    # regardless of which child logger emits it (propagation guarantees
    # delivery). The handler itself filters by message content.
    logging.getLogger().addHandler(timing_handler)
    logging.getLogger().setLevel(logging.INFO)

    session_id = f"bench-{int(datetime.now().timestamp())}"
    SessionManager.init_session(session_id=session_id)
    rag = RAGSystem(session_id=session_id)

    try:
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:  # noqa: BLE001 - fail fast with clear message
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "ollama" in msg:
            print(f"ERROR: {MSG_ERROR_OLLAMA_NOT_RUNNING}", file=sys.stderr)
            raise SystemExit(1) from exc
        raise

    file_name = PDF_PATH.name

    # Activate the query cache in-process (the S3 cache scenario). graph_builder
    # captures QUERY_CACHE_ENABLED as a value import, so patch BOTH the config
    # module and the bound name in graph_builder's namespace to flip the live
    # cache path on. No committed config is changed.
    if args.cache_on:
        import src.common.config as _config

        _config.QUERY_CACHE_ENABLED = True
        _graph_builder.QUERY_CACHE_ENABLED = True

        # The query cache get/set is gated on SessionManager "file_hash" (see
        # graph_builder.py:437/1427). Real entrypoints (main.py:539, api_server.py:435)
        # set it after indexing; replicate that here so the S3 cache scenario can HIT.
        _hash = compute_file_hash(str(PDF_PATH))
        SessionManager.set("file_hash", _hash, session_id=session_id)

    try:
        await rag.build_pipeline(str(PDF_PATH), file_name, embedder)
    except Exception as exc:  # noqa: BLE001 - fail fast with clear message
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "ollama" in msg:
            print(f"ERROR: {MSG_ERROR_OLLAMA_NOT_RUNNING}", file=sys.stderr)
            raise SystemExit(1) from exc
        raise

    # Install LLM-generation counter. The LLM instance `generate` actually
    # streams from is the one stored in SessionManager by build_pipeline; we wrap
    # it directly AND monkeypatch ModelManager.get_llm for any later fetch.
    llm = SessionManager.get("llm", session_id=session_id)
    llm_counter = _install_llm_counter(llm)

    print("\n[S1] cold (first query, full LLM path)...")
    s1_p95, s1_arr = await _scenario_cold(rag, timing_handler)

    print(f"[S2] warm (repeat={args.repeat}, steady-state)...")
    s2_p95, s2_arr = await _scenario_warm(rag, timing_handler, args.repeat)

    print("[S3] cache ON (first issuance then second issuance)...")
    (
        s3_first_p95,
        s3_repeat_p95,
        s3_repeat_llm,
        s3_first_arr,
        s3_repeat_arr,
    ) = await _scenario_cache(rag, timing_handler, llm_counter)

    captured = timing_handler.records
    report: dict[str, object] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "llm": DEFAULT_OLLAMA_MODEL,
            "embedder": DEFAULT_EMBEDDING_MODEL,
            "pdf": str(PDF_PATH),
            "repeat": args.repeat,
            "n_questions": len(FIXED_QUESTIONS),
            "query_cache_enabled": bool(args.cache_on),
        },
        "s1_cold_ttft_p95": round(s1_p95, 2),
        "s2_warm_ttft_p95": round(s2_p95, 2),
        "s3_first_ttft_p95": round(s3_first_p95, 2),
        "s3_repeat_ttft_p95": round(s3_repeat_p95, 2),
        "s3_repeat_llm_calls": s3_repeat_llm,
        "per_stage_ms": {
            "preprocess_ms": round(_avg_stage(captured, "preprocess_ms"), 2),
            "retrieve_ms": round(_avg_stage(captured, "retrieve_ms"), 2),
            "grade_ms": round(_avg_stage(captured, "grade_ms"), 2),
            "generate_total_ms": round(_avg_stage(captured, "generate_total_ms"), 2),
            "ttft_ms": round(_avg_stage(captured, "ttft_ms"), 2),
        },
        "scenarios": {
            "s1_cold_ttft_ms": [round(v, 2) for v in s1_arr],
            "s2_warm_ttft_ms": [round(v, 2) for v in s2_arr],
            "s3_first_ttft_ms": [round(v, 2) for v in s3_first_arr],
            "s3_repeat_ttft_ms": [round(v, 2) for v in s3_repeat_arr],
        },
    }

    if args.compare:
        _validate_baseline(report, Path(args.compare), args.max_regression_pct)

    # Defensive flag: with cache disabled, S3-repeat MUST invoke the LLM. A 0
    # here means instrumentation is missing the stream path — never report it as
    # a cache HIT. The strict ==0 enforcement belongs to the later regression
    # guard after the cache is wired (T5).
    query_cache_enabled = bool(report["config"].get("query_cache_enabled", False))  # type: ignore[union-attr]
    if s3_repeat_llm == 0 and not query_cache_enabled:
        warn = (
            "[WARN] LLM call counter read 0 with cache disabled — "
            "instrumentation may be missing the stream path"
        )
        print(warn, file=sys.stderr)
        logging.getLogger(__name__).warning(warn)

    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG query latency benchmark (3 scenarios)."
    )
    parser.add_argument(
        "--repeat", type=int, default=10, help="Repeat count for S2/S3 (default 10)"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: reports/query_bench_<ts>.json)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Baseline JSON to validate against (regression guard).",
    )
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=15.0,
        help="Max allowed TTFT regression vs baseline (default 15).",
    )
    parser.add_argument(
        "--bench-json",
        type=str,
        default=None,
        help=(
            "Existing report JSON to compare/validate instead of running a "
            "fresh capture (CI regression guard). Requires --compare (baseline)."
        ),
    )
    parser.add_argument(
        "--cache-on",
        action="store_true",
        default=False,
        help=(
            "Enable the query cache (QUERY_CACHE_ENABLED) for the S3 cache "
            "scenario so repeat queries are served from cache (0 LLM calls)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # CI regression guard: validate an already-captured report JSON against a
    # baseline WITHOUT running the live capture (keeps the capture path below
    # the single source of truth for measurements).
    if args.bench_json and args.compare:
        bench_path = Path(args.bench_json)
        baseline_path = Path(args.compare)
        if not bench_path.exists():
            print(
                f"[ERROR] --bench-json not found: {bench_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if not baseline_path.exists():
            print(
                f"[ERROR] --compare baseline not found: {baseline_path}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        with open(bench_path, encoding="utf-8") as f:
            report = json.load(f)
        with open(baseline_path, encoding="utf-8") as f:
            baseline = json.load(f)
        _compare_reports(report, baseline, args.max_regression_pct)
        print(f"\nCompared: {bench_path} vs {baseline_path}")
        return

    t0 = time.perf_counter()
    report = asyncio.run(main_async(args))
    wall_s = time.perf_counter() - t0

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.out)
        if args.out
        else REPORTS_DIR / f"query_bench_{int(datetime.now().timestamp())}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nReport written: {out_path}")
    print(f"Total wall time: {wall_s:.1f}s")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
