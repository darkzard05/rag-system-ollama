"""S5 결정성 프로브 — 동일 질문 N회 반복 검색의 흔들림 정량화 (src 수정 없음).

동일 PDF·동일 질문으로 retrieve-only 스트림을 N회 소비해 후보 집합 Jaccard와
rerank 점수 분산을 측정한다. S2 A/B 판독 가능 여부를 가르는 기준선.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.append(_path)

from src.api.streaming_handler import StreamingResponseHandler  # noqa: E402
from src.common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL  # noqa: E402
from src.common.logging_config import setup_logging  # noqa: E402
from src.core.model_loader import ModelManager  # noqa: E402
from src.core.rag_core import RAGSystem  # noqa: E402


def _doc_key(doc: object) -> str:
    meta = getattr(doc, "metadata", {}) or {}
    return f"{meta.get('page', '?')}|{meta.get('current_section', '')}"


async def _one_retrieval(rag: RAGSystem, query: str) -> list[tuple[str, float]]:
    handler = StreamingResponseHandler()
    event_stream = await rag.astream(query, model_name=DEFAULT_OLLAMA_MODEL)
    async for chunk in handler.stream_graph_events(event_stream):
        if chunk.metadata and "documents" in chunk.metadata:
            docs = list(chunk.metadata["documents"])
            out = []
            for d in docs:
                meta = getattr(d, "metadata", {}) or {}
                try:
                    score = float(meta.get("rerank_score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                out.append((_doc_key(d), round(score, 4)))
            return out
    return []


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


async def main() -> None:
    parser = argparse.ArgumentParser(description="S5 determinism probe")
    parser.add_argument("--pdf", default="tests/data/2201.07520v1.pdf")
    parser.add_argument(
        "--query",
        default="CM3 모델이 이미지를 학습할 때 사용하는 구체적인 원리와 토큰화 방식은 뭐야?",
    )
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--report", default="reports/repro_determinism.md")
    args = parser.parse_args()

    setup_logging(log_level="WARNING")
    rag = RAGSystem(session_id=f"repro-determinism-{int(time.time())}")
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
    await rag.build_pipeline(str(ROOT_DIR / args.pdf), Path(args.pdf).name, embedder)

    runs: list[list[tuple[str, float]]] = []
    for i in range(args.n):
        docs = await _one_retrieval(rag, args.query)
        runs.append(docs)
        top1 = docs[0] if docs else ("-", 0.0)
        print(f"[run {i + 1}] docs={len(docs)} top1={top1[0]} score={top1[1]}")

    key_sets = [set(k for k, _ in r) for r in runs]
    pairs = [_jaccard(a, b) for i, a in enumerate(key_sets) for b in key_sets[i + 1 :]]
    top1_scores = [dict(r).get(next(iter(dict(r)), ""), 0.0) for r in runs]
    first_keys = [r[0][0] if r else "-" for r in runs]

    lines = [
        "# S5 Determinism Probe",
        f"- query: {args.query}",
        f"- runs: {args.n}",
        f"- mean pairwise Jaccard (candidate sets): {sum(pairs) / len(pairs) if pairs else 1.0:.4f}",
        f"- min pairwise Jaccard: {min(pairs) if pairs else 1.0:.4f}",
        f"- top1 keys across runs: {first_keys}",
        f"- top1 same every run: {len(set(first_keys)) == 1}",
        "",
        "## Per-run top-3 (key | rerank)",
        "",
    ]
    for i, r in enumerate(runs):
        lines.append(f"### run {i + 1}")
        for k, s in r[:3]:
            lines.append(f"- {k} | {s}")
    report = "\n".join(lines) + "\n"
    out = ROOT_DIR / args.report
    out.write_text(report, encoding="utf-8")
    print(f"[saved] {out}")
    print(report)


if __name__ == "__main__":
    asyncio.run(main())
