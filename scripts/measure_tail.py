"""Tail latency baseline harness (read-only, logging only).

Runs 3 seeded synthetic turns through the Wave1-instrumented tail spans
(persist / finalize / pdf-finalize / recompute / rerun) and writes
``reports/tail_baseline_<ts>.json``. No behavior change: Streamlit and
SessionManager are stubbed via ``mock.patch`` (auto-restored), the real
coord/FAISS caches are never touched; turn 0 is labelled cold and turns
1-2 warm for the coord-cache label only.

Usage:
    python scripts/measure_tail.py
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import re
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

SPAN_RE = re.compile(r"\[PERF\] (\w[\w ]*?) took ([0-9.]+)s")

TURN_SEEDS: list[int] = [7, 8, 9]
TURN_DOC_COUNTS: list[int] = [3, 6, 12]
TURN_LABELS: list[str] = ["cold", "warm", "warm"]


class _SpanCollector(logging.Handler):
    """Captures [PERF] span durations from log records."""

    def __init__(self) -> None:
        super().__init__()
        self.spans: dict[str, float] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            match = SPAN_RE.search(record.getMessage())
        except (ValueError, TypeError, AttributeError):
            return
        if match is not None:
            self.spans[match.group(1).strip()] = float(match.group(2))


def _seeded_docs(rng: random.Random, count: int) -> list[dict[str, Any]]:
    """Deterministic fake docs supported by doc_stable_id/get_doc_metadata."""
    docs: list[dict[str, Any]] = []
    for i in range(count):
        words = " ".join(
            rng.choice(["alpha", "beta", "gamma", "delta"]) for _ in range(24)
        )
        docs.append(
            {
                "page_content": f"tail baseline doc {i} {words}.",
                "metadata": {
                    "doc_id": f"tail-doc-{i}",
                    "page": (i % 4) + 1,
                    "current_section": f"Section {i % 3}",
                },
            }
        )
    return docs


@contextmanager
def _capture_spans() -> Any:
    """Attach a span collector to the instrumented UI loggers."""
    collector = _SpanCollector()
    loggers = [
        logging.getLogger("ui.components.chat"),
        logging.getLogger("ui.components.chat_references"),
    ]
    for logger in loggers:
        logger.addHandler(collector)
    try:
        yield collector
    finally:
        for logger in loggers:
            logger.removeHandler(collector)


def _run_tail_turn(
    docs: list[dict[str, Any]],
    *,
    turn_label: str,
) -> dict[str, Any]:
    """Time one synthetic tail turn; returns span name -> seconds.

    ``rerun_requested`` is the honest signal here: the in-process span for
    ``st.rerun`` is always ~0 because the real cost of a full rerun is the whole
    app re-executing, which a stubbed ``st`` cannot observe. What this harness
    can prove is whether the tail *asked* for one.
    """
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from common.utils import extract_annotations_from_docs  # noqa: PLC0415
    from ui.components import chat as chat_mod  # noqa: PLC0415
    from ui.components import chat_references as refs  # noqa: PLC0415

    spans: dict[str, float] = {}

    fake_ref_st = MagicMock()
    with patch.object(refs, "st", fake_ref_st), _capture_spans() as collector:
        refs._render_references_content(
            "msg-baseline",
            docs,
            citations=None,
            generating=True,
        )
    spans["recompute"] = collector.spans.get("recompute", 0.0)

    t0 = time.perf_counter()
    _ = extract_annotations_from_docs(docs)
    spans["pdf-finalize"] = time.perf_counter() - t0

    fake_chat_st = MagicMock()
    fake_chat_st.write_stream.return_value = "baseline tail"

    @contextmanager
    def _fake_chat_message(*args: Any, **kwargs: Any) -> Any:
        yield MagicMock()

    fake_chat_st.chat_message.side_effect = _fake_chat_message
    placeholder = MagicMock()
    placeholder.empty.return_value = None
    placeholder.caption.return_value = None
    fake_chat_st.empty.return_value = placeholder

    with (
        patch.object(chat_mod, "st", fake_chat_st),
        patch.object(chat_mod, "_content_generator", return_value=iter(["hi "])),
        patch.object(chat_mod, "_clear_aux_state", return_value=None),
        patch.object(chat_mod, "_finalize_pdf_side_effects", return_value=None),
        patch.object(chat_mod, "render_generation_expander", return_value=None),
        patch.object(chat_mod.SessionManager, "get", return_value=""),
        patch.object(chat_mod.SessionManager, "add_message", return_value=None),
        patch.object(chat_mod.SessionManager, "set", return_value=None),
        _capture_spans() as tail_collector,
        contextlib.suppress(BaseException),
    ):
        chat_mod._render_streaming_with_write_stream(
            {"msg_id": f"msg-baseline-{turn_label}"},
            f"sid-baseline-{turn_label}",
            "baseline query",
        )
    spans["persist"] = tail_collector.spans.get("live persist", 0.0)
    spans["finalize"] = tail_collector.spans.get("finalize", 0.0)
    spans["rerun"] = tail_collector.spans.get("rerun", 0.0)
    return {
        **spans,
        "rerun_requested": bool(fake_chat_st.rerun.call_count),
    }


def main() -> None:
    """Run the seeded turns and write the baseline report."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))

    turns: list[dict[str, Any]] = []
    for seed, count, label in zip(
        TURN_SEEDS, TURN_DOC_COUNTS, TURN_LABELS, strict=True
    ):
        rng = random.Random(seed)
        docs = _seeded_docs(rng, count)
        spans = _run_tail_turn(docs, turn_label=label)
        turns.append(
            {
                "seed": seed,
                "coord_cache": label,
                "n_docs": count,
                "spans_s": spans,
            }
        )

    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        return sum(t["spans_s"][key] for t in rows) / len(rows)

    summary: dict[str, Any] = {
        "mean_s": {
            k: _mean(turns, k)
            for k in ("persist", "finalize", "pdf-finalize", "recompute", "rerun")
        },
        "rerun_requested": any(t["spans_s"]["rerun_requested"] for t in turns),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = REPORTS_DIR / f"tail_baseline_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {"tool": "measure_tail", "turns": turns, "summary": summary},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
