"""BM25 backend A/B harness: rank-bm25 (A) vs bm25s (B, if installed).

Measures index_ms, per-query query_ms (p50/p95/QPS), and top-k overlap.
bm25s missing -> B branch SKIP (exit 0, A-only report).

Usage:
    python scripts/bench_bm25_backends.py [--docs 300] [--repeat 5] [--k 10]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = REPO_ROOT / "reports"

TOPICS = [
    "인공지능 검색 증강 생성 파이프라인 문서 분석",
    "attention mechanism transformer 논문 구조 설명",
    "데이터베이스 인덱싱과 벡터 검색 하이브리드 가중치",
    "요리 레시피 된장찌개 재료와 조리 방법",
    "금융 보고서 분기 매출과 영업 이익 표 분석",
    "쿠버네티스 배포 롤백과 헬스체크 설정",
    "수식 미분 적분 선형대수 행렬 고유값",
    "법률 계약 조항 책임 범위와 손해배상",
    "여행 일정 서울 부산 교통 숙박 추천",
    "기계학습 임베딩 코사인 유사도 리랭킹",
]

QUERIES = [
    "attention mechanism이란 무엇인가요",
    "하이브리드 검색 가중치는 어떻게 적용되나요",
    "된장찌개 레시피를 알려주세요",
    "분기 매출 표에서 영업이익은 얼마인가요",
    "임베딩 유사도 리랭킹 방법 설명",
    "쿠버네티스 롤백 설정 방법",
    "행렬 고유값이란 무엇인가요",
    "계약서 손해배상 조항 설명",
    "서울 여행 교통편 추천",
    "문서 분석 파이프라인 구조",
]


def _build_corpus(n_docs: int) -> list:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from langchain_core.documents import Document

    docs = []
    for i in range(n_docs):
        topic = TOPICS[i % len(TOPICS)]
        filler = f"문서 번호 {i} 섹션 내용 보강 텍스트 반복 " * 4
        docs.append(
            Document(
                page_content=f"{topic} {filler} 고유키워드{i % 50}",
                metadata={"doc_id": f"bench-{i}"},
            )
        )
    return docs


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


def _measure(retriever: object, k: int, repeat: int) -> dict:
    from core.retriever_factory import _bm25_results_with_scores

    per_query_ms: list[float] = []
    tops: dict[str, list[str]] = {}
    for q in QUERIES:
        qs: list[float] = []
        top: list[str] = []
        for _ in range(repeat):
            t0 = time.perf_counter()
            out = _bm25_results_with_scores(retriever, q, k)  # type: ignore[arg-type]
            qs.append((time.perf_counter() - t0) * 1000.0)
            top = [d.page_content[:60] for d in out]
        per_query_ms.extend(qs)
        tops[q] = top
    return {
        "query_ms_p50": _pct(per_query_ms, 0.5),
        "query_ms_p95": _pct(per_query_ms, 0.95),
        "query_ms_mean": statistics.fmean(per_query_ms),
        "qps": 1000.0 / statistics.fmean(per_query_ms) if per_query_ms else 0.0,
        "tops": tops,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from core.bm25_backends import BACKEND_BM25S, is_bm25s_available
    from core.retriever_factory import create_bm25_retriever

    try:
        import bm25s as _bm25s_mod  # type: ignore[import-not-found]

        bm25s_version: str | None = getattr(_bm25s_mod, "__version__", "unknown")
    except ImportError:
        bm25s_version = None
    try:
        import rank_bm25  # type: ignore[import-not-found]

        rank_version: str | None = getattr(rank_bm25, "__version__", "unknown")
    except ImportError:
        rank_version = None

    docs = _build_corpus(args.docs)

    t0 = time.perf_counter()
    retr_a = create_bm25_retriever(docs, backend="rank-bm25")
    index_a_ms = (time.perf_counter() - t0) * 1000.0
    res_a = _measure(retr_a, args.k, args.repeat)

    report: dict = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "docs": args.docs,
        "repeat": args.repeat,
        "k": args.k,
        "rank_bm25_version": rank_version,
        "bm25s_version": bm25s_version,
        "A_rank_bm25": {
            "index_ms": index_a_ms,
            **{k_: v for k_, v in res_a.items() if k_ != "tops"},
        },
    }

    if is_bm25s_available():
        t0 = time.perf_counter()
        retr_b = create_bm25_retriever(docs, backend=BACKEND_BM25S)
        index_b_ms = (time.perf_counter() - t0) * 1000.0
        res_b = _measure(retr_b, args.k, args.repeat)
        # Top-k overlap per query (set overlap ratio).
        overlaps = []
        for q in QUERIES:
            sa, sb = set(res_a["tops"][q]), set(res_b["tops"][q])
            overlaps.append(len(sa & sb) / max(1, args.k))
        report["B_bm25s"] = {
            "index_ms": index_b_ms,
            **{k_: v for k_, v in res_b.items() if k_ != "tops"},
            "topk_overlap_mean": statistics.fmean(overlaps),
            "topk_overlap_min": min(overlaps),
        }
        print(
            f"[bm25] A rank-bm25: index={index_a_ms:.1f}ms "
            f"q_p50={res_a['query_ms_p50']:.2f}ms p95={res_a['query_ms_p95']:.2f}ms "
            f"QPS={res_a['qps']:.1f}"
        )
        print(
            f"[bm25] B bm25s:     index={index_b_ms:.1f}ms "
            f"q_p50={res_b['query_ms_p50']:.2f}ms p95={res_b['query_ms_p95']:.2f}ms "
            f"QPS={res_b['qps']:.1f} overlap={report['B_bm25s']['topk_overlap_mean']:.2f}"
        )
    else:
        report["B_bm25s"] = {"status": "SKIP (bm25s not installed)"}
        print(
            f"[bm25] A rank-bm25: index={index_a_ms:.1f}ms "
            f"q_p50={res_a['query_ms_p50']:.2f}ms p95={res_a['query_ms_p95']:.2f}ms "
            f"QPS={res_a['qps']:.1f}"
        )
        print("[bm25] B bm25s: SKIP (not installed)")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"bm25_bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bm25] report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
