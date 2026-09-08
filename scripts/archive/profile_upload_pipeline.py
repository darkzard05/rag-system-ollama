"""업로드 → 인덱싱 파이프라인 단계별 프로파일러.

실제 파이프라인과 동일한 호출 경로(get_embedder → load_pdf_docs →
split_documents → create_vector_store/bm25)를 사용하여 각 단계의 실제
소요시간과 전체 대비 비중을 측정한다.

사용법:
    python scripts/verification/profile_upload_pipeline.py <pdf_path> [embedder_model]

예:
    python scripts/verification/profile_upload_pipeline.py sample.pdf
    python scripts/verification/profile_upload_pipeline.py sample.pdf nomic-embed-text-v2-moe
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from core.model_loader import ModelManager
from core.session import SessionManager
from core.document_processor import load_pdf_docs
from core.chunking import split_documents
from core.retriever_factory import create_vector_store, create_bm25_retriever


def _fmt(sec: float) -> str:
    if sec >= 60:
        return f"{sec / 60:.1f}m"
    if sec >= 1:
        return f"{sec:.2f}s"
    return f"{sec * 1000:.0f}ms"


async def main() -> None:
    if len(sys.argv) < 2:
        print("사용법: python profile_upload_pipeline.py <pdf_path> [embedder_model]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    embedder_name = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(pdf_path):
        print(f"[ERROR] 파일 없음: {pdf_path}")
        sys.exit(1)

    file_name = os.path.basename(pdf_path)
    sid = "profiler_session"
    SessionManager.set_session_id(sid)

    stages: list[tuple[str, float, dict[str, str]]] = []
    total_start = time.time()

    # ── 1. 임베더 로드 (콜드 로드 여부 확인) ──
    t = time.time()
    embedder = await ModelManager.get_embedder(embedder_name)
    embed_model = getattr(embedder, "model", getattr(embedder, "model_name", "?"))
    stages.append(
        ("1. 임베더 로드 (get_embedder)", time.time() - t, {"model": str(embed_model)})
    )

    # ── 2. PDF 로드 + 단어 좌표 추출 (hydration) ──
    t = time.time()
    docs = await load_pdf_docs(pdf_path, file_name, session_id=sid)
    load_sec = time.time() - t
    page_count = docs[0].metadata.get("total_pages") if docs else 0
    stages.append(
        (
            "2. PDF 로드 + 단어좌표 추출",
            load_sec,
            {"pages": str(page_count), "blocks": str(len(docs))},
        )
    )

    if not docs:
        print("[ERROR] 문서 추출 실패 (빈 결과)")
        sys.exit(1)

    # ── 3. 청킹 + 임베딩 (의미론적 청킹 시 문장 단위 임베딩) ──
    t = time.time()
    doc_splits, vectors = await split_documents(docs, embedder=embedder, session_id=sid)
    chunk_sec = time.time() - t
    stages.append(
        (
            "3. 청킹 + 임베딩 (split_documents)",
            chunk_sec,
            {
                "chunks": str(len(doc_splits)),
                "vectors": "yes" if vectors is not None else "no",
            },
        )
    )

    if not doc_splits:
        print("[ERROR] 청크 생성 실패")
        sys.exit(1)

    # ── 4. 벡터 스토어 + BM25 병렬 생성 ──
    t = time.time()
    vs_future = asyncio.to_thread(
        create_vector_store, doc_splits, embedder, vectors=vectors, session_id=sid
    )
    bm25_future = asyncio.to_thread(create_bm25_retriever, doc_splits)
    vector_store, bm25_retriever = await asyncio.gather(vs_future, bm25_future)
    store_sec = time.time() - t
    stages.append(
        (
            "4. VectorStore + BM25 생성",
            store_sec,
            {"vector_dim": str(len(vectors[0])) if vectors else "-"},
        )
    )

    # ── 5. 캐시 저장 (VectorStoreCache는 file_path/emb_model_name/해시 필요) ──
    # 프로파일 목적상 실제 캐시 저장은 생략(병목이 아니며 의존 인자 과다).
    # 대신 디스크 쓰기 점유 비중을 0으로 표기하고 경고만 출력한다.
    stages.append(
        (
            "5. 디스크 캐시 저장 (미측정)",
            0.0,
            {"note": "생략 — pipeline_builder 전용 인자 필요"},
        )
    )

    total_sec = time.time() - total_start

    # ── 리포트 ──
    print("\n" + "=" * 70)
    print(f"업로드→인덱싱 파이프라인 프로파일: {file_name}")
    print(f"  임베더: {embed_model} | 페이지: {page_count} | 청크: {len(doc_splits)}")
    print("=" * 70)
    print(f"{'단계':<38}{'소요':>10}{'비중':>10}  상세")
    print("-" * 70)
    for name, sec, meta in stages:
        pct = (sec / total_sec * 100) if total_sec else 0
        meta_str = "  ".join(f"{k}={v}" for k, v in meta.items())
        print(f"{name:<38}{_fmt(sec):>10}{pct:>9.1f}%  {meta_str}")
    print("-" * 70)
    print(f"{'총계':<38}{_fmt(total_sec):>10}{'100.0%':>10}")

    # 가장 느린 단계 강조
    slowest = max(stages, key=lambda s: s[1])
    print(
        f"\n▶ 최대 병목: {slowest[0]} ({_fmt(slowest[1])}, "
        f"{slowest[1] / total_sec * 100:.1f}%)"
    )

    # 청킹/임베딩이 전체의 절반 이상이면 의미론적 청킹 의심
    chunk_pct = chunk_sec / total_sec * 100
    if chunk_pct >= 50:
        print("⚠️  청킹+임베딩 단계가 절반 이상을 차지합니다.")
        print("    semantic_chunker.enabled 가 true 면 문장 단위 임베딩으로")
        print("    텍스트 호출량이 청크 단위 대비 5~10배 커집니다 (config.yml:134).")


if __name__ == "__main__":
    asyncio.run(main())
