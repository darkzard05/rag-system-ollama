"""Live end-to-end verification of the real RAG pipeline.

Runs the production code path against a real PDF (``tests/data/2201.07520v1.pdf``),
the configured Ollama LLM and embedding model, and the golden question set.

What it verifies
----------------
- preflight: PDF asset, Ollama reachability, required models installed
- build: ``RAGSystem.build_pipeline`` return contract, stage events, session
  commit of ``file_hash``/``pdf_file_path``, optional vector-cache hit
- query: ``aquery`` result shape, non-empty answer, retrieved documents
- stream (``--mode full``): ``astream`` content chunks, TTFT, incremental and
  final citation events (``citations_add`` / ``citations``)
- signal (warn-only): ``required_chunks`` present in context, out-of-doc refusal

Exit codes
----------
- ``0``: every hard check passed (warnings allowed unless ``--strict``)
- ``1``: at least one hard check failed
- ``2``: environment precondition missing (PDF, Ollama, or models)

Usage
-----
    python scripts/verification/verify_full_pipeline.py
    python scripts/verification/verify_full_pipeline.py --mode smoke
    python scripts/verification/verify_full_pipeline.py --check-cache --json reports/e2e.json
    python scripts/verification/verify_full_pipeline.py --ollama-timeout 300  # cold start

Requires a live Ollama with the models from ``config.yml``. A cold model load can
exceed the configured 120s read timeout; ``--ollama-timeout`` raises it for the run.
Heavy imports are kept inside ``main`` so importing this module (e.g. path-guard
tests) stays cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))
PDF_PATH = ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf"
GOLDEN_PATH = ROOT_DIR / "tests" / "data" / "golden_set.json"

REFUSAL_MARKERS = (
    "확인할 수 없",
    "관련 정보가 없",
    "찾을 수 없",
    "문서에 없",
    "cannot find",
    "not mentioned",
    "no information",
)


class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str = ""


@dataclass
class Report:
    strict: bool
    checks: list[CheckResult] = field(default_factory=list)

    def add(
        self,
        name: str,
        ok: bool,
        detail: str = "",
        *,
        warn_only: bool = False,
    ) -> None:
        if ok:
            status = Status.PASS
        elif warn_only and not self.strict:
            status = Status.WARN
        else:
            status = Status.FAIL
        self.checks.append(CheckResult(name, status, detail))

    def skip(self, name: str, detail: str = "") -> None:
        self.checks.append(CheckResult(name, Status.SKIP, detail))

    @property
    def failed(self) -> bool:
        return any(check.status is Status.FAIL for check in self.checks)

    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in Status}
        for check in self.checks:
            counts[check.status.value] += 1
        return counts

    def print_all(self) -> None:
        width = max((len(check.name) for check in self.checks), default=0)
        for check in self.checks:
            line = f"[{check.status.value:<4}] {check.name:<{width}}  {check.detail}"
            print(line.rstrip())
        counts = self.counts()
        print(
            "\nSummary: "
            + " ".join(f"{key}={value}" for key, value in counts.items())
            + f" | overall={'FAIL' if self.failed else 'PASS'}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strict": self.strict,
            "overall": "FAIL" if self.failed else "PASS",
            "counts": self.counts(),
            "checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }


@dataclass
class QueryMetrics:
    query: str
    response_chars: int = 0
    documents: int = 0
    elapsed_s: float = 0.0
    ttft_s: float | None = None
    stream_chars: int = 0
    stream_chunks: int = 0
    citation_items: int = 0
    final_citations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "response_chars": self.response_chars,
            "documents": self.documents,
            "elapsed_s": round(self.elapsed_s, 3),
            "ttft_s": round(self.ttft_s, 3) if self.ttft_s is not None else None,
            "stream_chars": self.stream_chars,
            "stream_chunks": self.stream_chunks,
            "citation_items": self.citation_items,
            "final_citations": self.final_citations,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live full-pipeline verification against a real PDF."
    )
    parser.add_argument(
        "--mode",
        choices=("full", "smoke"),
        default="full",
        help="full: golden set + streaming/citations (default); smoke: build + 2 queries",
    )
    parser.add_argument("--model", default=None, help="LLM override")
    parser.add_argument("--embedding-model", default=None, help="embedding override")
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="cap the number of queries after mode selection",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=360.0,
        help="per query wall-clock timeout (must exceed --ollama-timeout)",
    )
    parser.add_argument("--build-timeout", type=float, default=900.0)
    parser.add_argument(
        "--ollama-timeout",
        type=int,
        default=None,
        help="ChatOllama read timeout in seconds (sets OLLAMA_TIMEOUT; config default 120)",
    )
    parser.add_argument(
        "--check-cache",
        action="store_true",
        help="force ENABLE_VECTOR_CACHE=true and assert a second build hits cache",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="promote warn-only checks (quality/citations/latency) to failures",
    )
    parser.add_argument("--json", default=None, help="write a JSON report to this path")
    parser.add_argument("--log-level", default="WARNING")
    return parser.parse_args()


def normalise_model(name: str) -> str:
    return name[: -len(":latest")] if name.endswith(":latest") else name


def preflight(base_url: str, required_models: list[str]) -> list[str]:
    problems: list[str] = []
    if not PDF_PATH.is_file():
        problems.append(f"missing PDF asset: {PDF_PATH}")
    if not GOLDEN_PATH.is_file():
        problems.append(f"missing golden set: {GOLDEN_PATH}")

    try:
        with urllib.request.urlopen(  # noqa: S310 - local Ollama
            f"{base_url}/api/tags", timeout=10
        ) as response:
            payload = json.load(response)
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        problems.append(
            f"Ollama unreachable at {base_url}: {type(exc).__name__}: {exc}"
        )
        return problems

    installed = {
        normalise_model(str(model.get("name") or model.get("model") or ""))
        for model in payload.get("models", [])
    }
    for required in required_models:
        if normalise_model(required) not in installed:
            problems.append(
                f"model not installed: {required} (available: {sorted(installed)})"
            )
    return problems


def load_queries(mode: str, limit: int | None) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if mode == "smoke":
        in_doc = [item for item in queries if item.get("required_chunks")]
        queries = in_doc[:2]
    if limit is not None:
        queries = queries[: max(0, limit)]
    return queries


def is_in_doc(item: dict[str, Any]) -> bool:
    return bool(item.get("required_chunks"))


def check_required_chunks(
    report: Report, label: str, item: dict[str, Any], context: str
) -> None:
    required = item.get("required_chunks") or []
    if not required:
        report.skip(f"{label}: required_chunks", "not applicable")
        return
    haystack = context.lower()
    missing = [chunk for chunk in required if chunk.lower() not in haystack]
    report.add(
        f"{label}: required_chunks in context",
        not missing,
        f"missing={missing}" if missing else f"matched={len(required)}",
        warn_only=True,
    )


def check_out_of_doc(report: Report, label: str, response: str) -> None:
    lowered = response.lower()
    matched = next(
        (marker for marker in REFUSAL_MARKERS if marker.lower() in lowered), None
    )
    report.add(
        f"{label}: out-of-doc refusal",
        matched is not None,
        f"marker={matched!r}" if matched else f"head={response[:80]!r}",
        warn_only=True,
    )


async def consume_stream(
    rag: Any, query: str, model: str | None, timeout: float
) -> tuple[float | None, int, int, int, int, float]:
    """Consume one astream; return (ttft, chars, chunks, items, final_cites, elapsed)."""

    async def _consume() -> tuple[float | None, int, int, int, int]:
        stream = await rag.astream(query, model_name=model)
        started = time.perf_counter()
        ttft: float | None = None
        chars = 0
        chunks = 0
        items = 0
        final_cites = 0
        try:
            async for kind, data in stream:
                if kind != "custom" or not isinstance(data, dict):
                    continue
                content = data.get("content")
                if isinstance(content, str) and content:
                    if ttft is None:
                        ttft = time.perf_counter() - started
                    chars += len(content)
                    chunks += 1
                if "items" in data and "base_index" in data:
                    items += len(data.get("items") or [])
                citations = data.get("citations")
                if isinstance(citations, list):
                    final_cites = max(final_cites, len(citations))
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                await aclose()
        return ttft, chars, chunks, items, final_cites

    started = time.perf_counter()
    ttft, chars, chunks, items, final_cites = await asyncio.wait_for(
        _consume(), timeout=timeout
    )
    return ttft, chars, chunks, items, final_cites, time.perf_counter() - started


async def run(args: argparse.Namespace) -> int:
    from common.config import (  # noqa: PLC0415  # type: ignore[import-not-found]
        DEFAULT_EMBEDDING_MODEL,
        DEFAULT_OLLAMA_MODEL,
        OLLAMA_BASE_URL,
    )
    from common.logging_config import setup_logging  # noqa: PLC0415  # type: ignore[import-not-found]
    from core.build_state import BuildEvent, BuildPhase  # noqa: PLC0415  # type: ignore[import-not-found]
    from core.model_loader import ModelManager  # noqa: PLC0415  # type: ignore[import-not-found]
    from core.rag_core import RAGSystem  # noqa: PLC0415  # type: ignore[import-not-found]
    from core.session import SessionManager  # noqa: PLC0415  # type: ignore[import-not-found]

    setup_logging(log_level=args.log_level, log_file=None, console_output=True)

    model = args.model or DEFAULT_OLLAMA_MODEL
    embedding_model = args.embedding_model or DEFAULT_EMBEDDING_MODEL
    report = Report(strict=args.strict)
    metrics: list[QueryMetrics] = []

    print(f"[preflight] pdf={PDF_PATH}")
    print(
        f"[preflight] model={model} embedding={embedding_model} url={OLLAMA_BASE_URL}"
    )
    problems = preflight(OLLAMA_BASE_URL, [model, embedding_model])
    if problems:
        for problem in problems:
            print(f"[preflight] FAIL: {problem}")
        return 2

    session_id = f"verify-full-{int(time.time())}"
    rag = RAGSystem(session_id=session_id)
    events: list[BuildEvent] = []
    build_cache_used = False
    build_elapsed = 0.0

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfile(PDF_PATH, tmp.name)
        temp_pdf_path = tmp.name

    try:
        embedder = await ModelManager.get_embedder(embedding_model)

        started = time.perf_counter()
        message, build_cache_used = await asyncio.wait_for(
            rag.build_pipeline(
                temp_pdf_path,
                PDF_PATH.name,
                embedder,
                on_event=events.append,
            ),
            timeout=args.build_timeout,
        )
        build_elapsed = time.perf_counter() - started
        print(
            f"[build] {build_elapsed:.2f}s cache_used={build_cache_used} msg={message!r}"
        )

        report.add(
            "build: return contract",
            isinstance(message, str)
            and bool(message.strip())
            and isinstance(build_cache_used, bool),
            f"msg={message!r} cache_used={build_cache_used!r}",
        )
        report.add(
            "build: events emitted",
            bool(events),
            f"n={len(events)} phases={[event.phase.value for event in events]}",
        )
        if events:
            last = events[-1]
            report.add(
                "build: final stage complete",
                last.phase is BuildPhase.REGISTERING and last.pct == 100,
                f"last={last.phase.value}@{last.pct}",
            )
            phases = {event.phase for event in events}
            if build_cache_used:
                report.add(
                    "build: fresh stage set",
                    {BuildPhase.INDEXING, BuildPhase.REGISTERING} <= phases,
                    f"cache hit phases={[p.value for p in phases]}",
                )
            else:
                expected = {
                    BuildPhase.EXTRACTING,
                    BuildPhase.CHUNKING,
                    BuildPhase.INDEXING,
                    BuildPhase.REGISTERING,
                }
                missing = expected - phases
                report.add(
                    "build: fresh stage set",
                    not missing,
                    f"missing={[p.value for p in missing]}",
                )
        else:
            report.add("build: final stage complete", False, "no events")

        file_hash = SessionManager.get("file_hash", session_id=session_id)
        pdf_path = SessionManager.get("pdf_file_path", session_id=session_id)
        report.add(
            "build: session commit",
            bool(file_hash) and pdf_path == temp_pdf_path,
            f"file_hash={str(file_hash)[:12]}... pdf_path={pdf_path!r}",
        )

        if args.check_cache:
            _, second_cache_used = await asyncio.wait_for(
                rag.build_pipeline(
                    str(PDF_PATH),
                    PDF_PATH.name,
                    embedder,
                    on_event=events.append,
                ),
                timeout=args.build_timeout,
            )
            report.add(
                "build: second build cache hit",
                bool(second_cache_used),
                f"cache_used={second_cache_used!r}",
                warn_only=True,
            )

        queries = load_queries(args.mode, args.query_limit)
        if not queries:
            report.add("queries: selected", False, "golden set selection is empty")
        stream_index = next(
            (index for index, item in enumerate(queries) if is_in_doc(item)), None
        )
        stream_query = queries[stream_index] if stream_index is not None else None

        for index, item in enumerate(queries, start=1):
            label = f"query[{index}]"
            query = str(item.get("query") or "")
            started = time.perf_counter()
            result = await asyncio.wait_for(
                rag.aquery(query, model_name=model), timeout=args.timeout
            )
            elapsed = time.perf_counter() - started

            required_keys = {
                "response",
                "thought",
                "context",
                "documents",
                "performance",
            }
            report.add(
                f"{label}: result keys",
                isinstance(result, dict) and required_keys <= set(result),
                f"keys={sorted(result) if isinstance(result, dict) else type(result).__name__}",
            )
            response = result.get("response") or ""
            context = result.get("context") or ""
            documents = result.get("documents") or []
            if is_in_doc(item):
                report.add(
                    f"{label}: response non-empty",
                    isinstance(response, str) and len(response.strip()) >= 80,
                    f"chars={len(response)}",
                )
                report.add(
                    f"{label}: documents retrieved",
                    len(documents) >= 1,
                    f"n={len(documents)}",
                )
            else:
                report.add(
                    f"{label}: response non-empty",
                    isinstance(response, str) and bool(response.strip()),
                    f"chars={len(response)}",
                )
            check_required_chunks(report, label, item, str(context))
            if item.get("out_of_doc"):
                check_out_of_doc(report, label, str(response))

            metrics.append(
                QueryMetrics(
                    query=query,
                    response_chars=len(response),
                    documents=len(documents),
                    elapsed_s=elapsed,
                )
            )

        if args.mode == "full" and stream_query is not None:
            query = str(stream_query.get("query") or "")
            ttft, chars, chunks, items, final_cites, elapsed = await consume_stream(
                rag, query, model, args.timeout
            )
            print(
                f"[stream] {elapsed:.2f}s ttft={ttft} chunks={chunks} "
                f"chars={chars} citation_items={items} final_citations={final_cites}"
            )
            report.add("stream: content received", chars > 0, f"chars={chars}")
            report.add(
                "stream: citations received",
                items > 0 or final_cites > 0,
                f"items={items} final={final_cites}",
                warn_only=True,
            )
            if stream_index is not None and stream_index < len(metrics):
                metrics[stream_index].ttft_s = ttft
                metrics[stream_index].stream_chars = chars
                metrics[stream_index].stream_chunks = chunks
                metrics[stream_index].citation_items = items
                metrics[stream_index].final_citations = final_cites

        report.add(
            "build: duration recorded",
            build_elapsed > 0,
            f"{build_elapsed:.2f}s",
        )
    except Exception as exc:
        report.add(
            "pipeline: unexpected error",
            False,
            f"{type(exc).__name__}: {exc}",
        )
    finally:
        try:
            rag.clear_session()
        finally:
            SessionManager.delete_session(session_id)
            try:
                os.unlink(temp_pdf_path)
            except OSError:
                pass

    report.print_all()
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": args.mode,
            "model": model,
            "embedding_model": embedding_model,
            "build_elapsed_s": round(build_elapsed, 3),
            "build_cache_used": build_cache_used,
            "queries": [metric.to_dict() for metric in metrics],
            "report": report.to_dict(),
        }
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[report] {output}")

    return 1 if report.failed else 0


def main() -> int:
    args = parse_args()
    if args.check_cache:
        os.environ["ENABLE_VECTOR_CACHE"] = "true"
    if args.ollama_timeout is not None:
        os.environ["OLLAMA_TIMEOUT"] = str(args.ollama_timeout)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
