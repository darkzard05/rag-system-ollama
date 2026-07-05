"""
문서 분할(Chunking)을 담당하는 모듈.
"""

import logging
import os

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import (
    EMBEDDING_BATCH_SIZE,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
)
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.session import SessionManager
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


def _get_optimal_batch_size(embedder: Embeddings) -> int:
    """하드웨어 사양에 따른 최적 배치 사이즈 결정"""
    if isinstance(EMBEDDING_BATCH_SIZE, int):
        return EMBEDDING_BATCH_SIZE

    import torch

    if getattr(embedder, "model_kwargs", {}).get("device") == "cuda":
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return 128 if total_mem > 10 else (64 if total_mem > 4 else 32)
        except Exception:
            return 32
    return min(max(4, os.cpu_count() or 4), 16)


def _init_semantic_chunker(embedder: Embeddings) -> EmbeddingBasedSemanticChunker:
    """설정값을 기반으로 의미론적 분할기 초기화"""
    cfg = SEMANTIC_CHUNKER_CONFIG
    return EmbeddingBasedSemanticChunker(
        embedder=embedder,
        breakpoint_threshold_type=cfg.get("breakpoint_threshold_type", "percentile"),
        breakpoint_threshold_value=float(cfg.get("breakpoint_threshold_value", 95.0)),
        sentence_split_regex=cfg.get("sentence_split_regex", r"[.!?]\s+"),
        min_chunk_size=int(cfg.get("min_chunk_size", 100)),
        max_chunk_size=int(cfg.get("max_chunk_size", 800)),
        similarity_threshold=float(cfg.get("similarity_threshold", 0.5)),
        batch_size=_get_optimal_batch_size(embedder),
    )


def _postprocess_metadata(split_docs: list[Document]) -> None:
    """청크별 메타데이터 정리 및 내용 유형(콘텐츠/참고문헌 등) 식별"""
    noise_keywords = ["index", "references", "bibliography", "doi:", "isbn"]

    # 참고문헌 시작 섹션 탐지 (전체 문서 스캔)
    ref_start_idx = None
    for i, doc in enumerate(split_docs):
        content_lower = doc.page_content.lower()
        if any(kw in content_lower[:50] for kw in ["## references", "references\n---"]):
            ref_start_idx = i
            break

    for i, doc in enumerate(split_docs):
        doc.metadata = doc.metadata.copy()
        doc.metadata["chunk_index"] = i
        content_lower = doc.page_content.lower()

        # 노이즈 판별
        is_noise = any(kw in content_lower[:100] for kw in noise_keywords)
        if not is_noise and (
            content_lower.count("doi:") > 2 or content_lower.count(",") > 25
        ):
            is_noise = True

        # 참고문헌 여부: ref_start_idx 이후만 True (한 번 True면 계속 True인 버그 제거)
        is_reference = ref_start_idx is not None and i >= ref_start_idx

        doc.metadata.update(
            {
                "is_content": not (is_noise or is_reference),
                "is_reference": is_reference,
                "is_anchor": doc.metadata.get("is_anchor", False)
                if i == 0
                else False,  # 첫 페이지만 앵커 유지
                "is_header": bool(doc.metadata.get("page") == 1 and i < 3),
            }
        )


async def split_documents(
    docs: list[Document],
    embedder: Embeddings | None = None,
    session_id: str | None = None,
) -> tuple[list[Document], list[np.ndarray] | None]:
    """설정에 따라 문서를 분할하고 벡터를 생성합니다."""
    if not docs:
        return [], None

    is_already_chunked = docs[0].metadata.get("is_already_chunked", False)
    split_docs: list[Document] = []
    vectors: list[np.ndarray] | None = None

    # [최적화] 페이지 단위로 이미 분할되었더라도 너무 긴 경우(오버플로우) 재분할 수행
    max_chunk_size = TEXT_SPLITTER_CONFIG.get("chunk_size", 500)
    needs_sub_chunking = is_already_chunked and any(
        len(d.page_content) > max_chunk_size * 1.5 for d in docs
    )

    if is_already_chunked and not needs_sub_chunking:
        SessionManager.add_status_log(
            f"기존 분할 구조 활용 ({len(docs)}개 섹션)", session_id=session_id
        )
        split_docs = docs
        if embedder:
            SessionManager.add_status_log("지식 벡터화 중...", session_id=session_id)
            import asyncio

            # [최적화] 동기 임베딩 생성을 비동기 스레드로 분리
            raw_vectors = await asyncio.to_thread(
                embedder.embed_documents, [d.page_content for d in split_docs]
            )
            vectors = [np.array(v) for v in raw_vectors]
    else:
        if needs_sub_chunking:
            SessionManager.add_status_log(
                "대형 섹션 감지: 정밀 검색을 위한 하위 분할 시작",
                session_id=session_id,
            )
        else:
            SessionManager.add_status_log(
                "문서 분할 및 문맥 추출 중...", session_id=session_id
            )

        use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)

        if use_semantic and embedder:
            with monitor.track_operation(
                OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
            ):
                semantic_chunker = _init_semantic_chunker(embedder)
                split_docs, vectors = await semantic_chunker.split_documents(docs)
                msg = f"의미론적 분할 완료 ({len(split_docs)}개 조각)"
        else:
            import asyncio

            recursive_chunker = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            )
            # [최적화] CPU 집약적 청킹 분리
            split_docs = await asyncio.to_thread(
                recursive_chunker.split_documents, docs
            )
            if embedder:
                # [최적화] 동기 임베딩 생성을 비동기 스레드로 분리
                raw_vectors = await asyncio.to_thread(
                    embedder.embed_documents, [d.page_content for d in split_docs]
                )
                vectors = [np.array(v) for v in raw_vectors]
            msg = f"표준 분할 완료 ({len(split_docs)}개 조각)"

        SessionManager.add_status_log(msg, session_id=session_id)
        logger.info(
            f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
        )

    _postprocess_metadata(split_docs)
    return split_docs, vectors
