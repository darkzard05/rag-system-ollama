"""
문서 분할(Chunking)을 담당하는 모듈.
"""

import asyncio
import logging
import os

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from common.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
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


def _is_ollama_embedder(embedder: Embeddings) -> bool:
    """Ollama 기반 임베더인지 판별합니다.

    model_loader가 이미 langchain_ollama를 로드한 뒤 호출되므로 추가 임포트
    비용 없이 정확한 isinstance 판별이 가능합니다. langchain_ollama가
    설치되지 않은 환경(HuggingFace 전용)에서는 False를 반환합니다.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        return False
    return isinstance(embedder, OllamaEmbeddings)


def _get_optimal_batch_size(embedder: Embeddings) -> int:
    """하드웨어 사양에 따른 최적 배치 사이즈 결정"""
    if isinstance(EMBEDDING_BATCH_SIZE, int):
        return EMBEDDING_BATCH_SIZE

    import torch

    # HuggingFace 임베더: model_kwargs.device로 명시적 판별 (기존 동작 유지)
    device = getattr(embedder, "model_kwargs", {}).get("device")

    # [수정] Ollama 임베더: 실행 디바이스가 model_kwargs에 노출되지 않아
    # (extra="forbid" Pydantic 모델) 항상 CPU 분기로 떨어지던 문제 해결.
    # 로컬 CUDA 가용 여부 또는 명시적 embedding_device 설정(cuda) 기준으로
    # GPU 배치 크기를 적용한다. OllamaEmbeddings.embed_documents는 입력 전체를
    # 단일 HTTP 요청으로 전송하므로, 배치 증가 = HTTP 왕복 횟수 감소.
    is_ollama = _is_ollama_embedder(embedder)
    uses_gpu = device == "cuda" or (
        is_ollama and (torch.cuda.is_available() or EMBEDDING_DEVICE == "cuda")
    )

    if uses_gpu:
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
    ref_start_idx = None
    noise_keywords = ["index", "references", "bibliography", "doi:", "isbn"]
    ref_keywords = ["## references", "references\n---", "## 참고문헌", "참고문헌\n---"]

    for i, doc in enumerate(split_docs):
        content_lower = doc.page_content.lower()

        if ref_start_idx is None and any(
            kw in content_lower[:50] for kw in ref_keywords
        ):
            ref_start_idx = i

        doc.metadata = doc.metadata.copy()
        doc.metadata["chunk_index"] = i

        is_noise = any(kw in content_lower[:100] for kw in noise_keywords)
        if not is_noise and (
            content_lower.count("doi:") > 2 or content_lower.count(",") > 25
        ):
            is_noise = True

        is_reference = ref_start_idx is not None and i >= ref_start_idx

        # [R2-08] 문서 단위 is_anchor/is_header 휴리스틱은 제품 파이프라인에서
        # 소비처 0건(테스트 포함) — 데드 계산 제거. is_content/is_reference는
        # 기존 테스트(metadata audit)가 소비하므로 유지한다.
        doc.metadata.update(
            {
                "is_content": not (is_noise or is_reference),
                "is_reference": is_reference,
            }
        )


async def _embed_documents_chunks(
    chunks: list[Document], embedder: Embeddings
) -> list[np.ndarray]:
    """Generate embeddings for document chunks asynchronously (CPU-offloaded)."""
    raw_vectors = await asyncio.to_thread(
        embedder.embed_documents, [d.page_content for d in chunks]
    )
    return [np.array(v) for v in raw_vectors]


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
        if embedder and vectors is None:
            SessionManager.add_status_log("지식 벡터화 중...", session_id=session_id)
            vectors = await _embed_documents_chunks(split_docs, embedder)
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
            with get_performance_monitor().track_operation(
                OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
            ):
                semantic_chunker = _init_semantic_chunker(embedder)
                split_docs, vectors = await semantic_chunker.split_documents(docs)
                msg = f"의미론적 분할 완료 ({len(split_docs)}개 조각)"
        else:
            recursive_chunker = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            )
            # [최적화] CPU 집약적 청킹 분리
            split_docs = await asyncio.to_thread(
                recursive_chunker.split_documents, docs
            )
            if embedder and vectors is None:
                vectors = await _embed_documents_chunks(split_docs, embedder)
            msg = f"표준 분할 완료 ({len(split_docs)}개 조각)"

        SessionManager.add_status_log(msg, session_id=session_id)
        logger.info(
            f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
        )

    _postprocess_metadata(split_docs)
    return split_docs, vectors
