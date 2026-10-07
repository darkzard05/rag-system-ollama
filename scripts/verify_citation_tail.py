"""Citation-tail 검증 스크립트 (문제1 마무리 지연 직접 측정용).

Tier A (결정적, LLM 불필요):
    실제 ``PartialCitationScanner`` + 실제 ``merge_citations_add`` 에
    현실적인 구조화 JSON 토큰 스트림을 재생해, 첫 증분 인용이 최종 토큰
    이전(mid-stream)에 도착하는지 + 병합 결과가 전체 집합과 일치하는지
    검증한다.

Tier B (라이브, Ollama 필요):
    실제 RAG 파이프라인 + 실제 UI 소비 경로
    (``consume_stream_into_message``) 로 질문 1건을 headless 실행해
    P0 = last_accum_growth -> first_citations 도달 시간을 측정한다.
    판정: citations 비어있지 않고 P0 <= 7.0s.

실행:
    $env:PYTHONPATH="src"; python scripts/verify_citation_tail.py        # Tier A
    $env:PYTHONPATH="src"; python scripts/verify_citation_tail.py --live # A+B
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

P0_BUDGET_SEC = 7.0


def _build_realistic_json() -> tuple[str, list[dict[str, Any]]]:
    answer = (
        "CM3는 causally masked objective로 학습되며 이미지는 VQVAE-GAN "
        "토큰으로 변환해 텍스트와 함께 시퀀스 모델링한다. " * 12
    ).strip()
    citations = [
        {
            "doc_id": f"doc-{i}",
            "page": 13 + i,
            "section": f"Section {i}",
            "text_span": f"evidence span number {i} about cm3 training",
        }
        for i in range(6)
    ]
    import json as _json

    payload = {"final_answer": answer, "citations": citations, "confidence": 0.9}
    return _json.dumps(payload, ensure_ascii=False), citations


def tier_a(chunk_size: int = 13) -> bool:
    from core.graph._partial_citations import PartialCitationScanner
    from ui.components.streaming_core import merge_citations_add

    full, expected = _build_realistic_json()
    chunks = [full[i : i + chunk_size] for i in range(0, len(full), chunk_size)]
    scanner = PartialCitationScanner()
    accumulated = ""
    emits: list[tuple[int, list[tuple[int, dict]]]] = []
    for idx, piece in enumerate(chunks):
        accumulated += piece
        new_cites = scanner.feed(accumulated)
        if new_cites:
            emits.append((idx, new_cites))
    total = len(chunks)
    first_idx = emits[0][0] if emits else -1
    merged: list[dict[str, Any]] = []
    for _idx, batch in emits:
        items = [obj for _, obj in batch]
        merged = merge_citations_add(
            merged, items=items, base_index=batch[0][0], arr_len=scanner.emitted
        )
    mid_stream = 0 < first_idx < total - 1
    complete = merged == expected
    print(f"[TierA] chunks={total} emits={len(emits)} first_add_at_chunk={first_idx}")
    print(f"[TierA] mid_stream={mid_stream} merged_complete={complete}")
    ok = mid_stream and complete and bool(emits)
    print(f"[TierA] {'PASS' if ok else 'FAIL'}")
    return ok


def tier_b(query: str = "cm3") -> bool:
    from common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
    from common.logging_config import setup_logging
    from core.model_loader import ModelManager
    from core.rag_core import RAGSystem
    from core.session import SessionManager
    from ui.components.streaming import consume_stream_into_message

    setup_logging(log_level="WARNING")
    sid = f"verify-tail-{int(time.time())}"
    SessionManager.init_session(session_id=sid)
    rag = RAGSystem(session_id=sid)

    async def _build() -> None:
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
        test_pdf = str(ROOT_DIR / "tests" / "data" / "2201.07520v1.pdf")
        msg, cache_used = await rag.build_pipeline(
            test_pdf, "2201.07520v1.pdf", embedder
        )
        print(f"[TierB] pipeline: {msg} (cache={cache_used})")

    asyncio.run(_build())

    snaps: list[tuple[float, int, int]] = []  # (t, accum_len, n_citations)

    def _on_chunk(snap: dict[str, Any]) -> None:
        snaps.append(
            (time.perf_counter(), len(snap["accumulated"]), len(snap["citations"]))
        )

    t0 = time.perf_counter()
    result = consume_stream_into_message(
        sid, query, DEFAULT_OLLAMA_MODEL, on_chunk=_on_chunk
    )
    wall = time.perf_counter() - t0
    if not result or not snaps:
        print("[TierB] FAIL: no result/snapshots")
        return False
    t_first_cit = next((t for t, _, n in snaps if n > 0), None)
    grown = [t for t, ln, _ in snaps if ln > 0]
    t_last_accum = max(grown) if grown else None
    n_final = len(result.get("citations", []))
    p0 = (t_first_cit - t_last_accum) if (t_first_cit and t_last_accum) else None
    print(f"[TierB] wall={wall:.1f}s final_citations={n_final}")
    print(f"[TierB] t_last_accum={t_last_accum:.3f} t_first_cit={t_first_cit}")
    print(f"[TierB] P0(last_token->first_citations)={p0}")
    ok = n_final > 0 and p0 is not None and p0 <= P0_BUDGET_SEC
    print(f"[TierB] {'PASS' if ok else 'FAIL'} (budget={P0_BUDGET_SEC}s)")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--query", default="cm3")
    args = ap.parse_args()
    ok_a = tier_a()
    ok_b = tier_b(args.query) if args.live else True
    print(
        f"RESULT: TierA={'PASS' if ok_a else 'FAIL'}"
        + (f" TierB={'PASS' if ok_b else 'FAIL'}" if args.live else "")
    )
    return 0 if (ok_a and ok_b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
