"""
문서 분할(Chunking)을 담당하는 모듈.
"""

import asyncio
import contextlib
import io
import logging
import os

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# [후속-transformers-print] langchain_text_splitters 패키지 __init__이
# .sentence_transformers 서브모듈을 끌어오고, 이어 sentence_transformers →
# transformers 리프 이미지 프로세서(deepseek_vl_hybrid / kimi_k25 / paddleocr_vl)
# import 시 @auto_docstring 데코레이터가 "[ERROR] ... but not documented"를
# stdout print로 뿌린다 (logging/warnings 아님 — 공식 문서는 기여자용 CI
# 진단으로만 의도). 패키지 __init__ 실행은 회피 불가하므로 import 자체를
# stdout 억제 하에 수행한다. 프로젝트 코드는 stdout print를 사용하지 않고
# (전부 stderr 로깅) import는 임포트 락 하에 1회만 실행되므로 안전하다.
with contextlib.redirect_stdout(io.StringIO()):
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
    from core.resource_manager import get_resource_manager

    coordinator = get_resource_manager()
    async with coordinator.use_embedder(embedder=embedder):
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
        chunk_backend = str(SEMANTIC_CHUNKER_CONFIG.get("backend", "builtin")).lower()
        # [Phase C1] 오버플로우 페이지만 하위분할: 20p 중 1페이지가 750자를
        # 초과해도 전체를 의미분할(문장 임베딩 수백 회)로 보내던 과발동을
        # 수정한다. 정상 페이지는 그대로 재사용하고, 초과분만 의미분할한다.
        # (신규 의존성 없이 기존 builtin 경로만 사용. 벡터 캐시 복구 없음.)
        if needs_sub_chunking and is_already_chunked:
            threshold = max_chunk_size * 1.5
            overflow_docs = [d for d in docs if len(d.page_content) > threshold]
            logger.info(
                "[RAG] [CHUNKING] 오버플로우 하위분할 | 전체: %d | 초과: %d",
                len(docs),
                len(overflow_docs),
            )
            if use_semantic and embedder and overflow_docs:
                semantic_chunker = _init_semantic_chunker(embedder)
                # [최적화] 단일 임베딩 왕복: 오버플로우 문장 텍스트와 정상 청크
                # 텍스트를 합쳐 embed_documents 1회로 처리한다. Ollama는 입력
                # 전체를 단일 HTTP 요청으로 전송하므로 분할 호출은 왕복만 늘린다.
                # 문장 벡터는 l2 정규화 후 경계 탐색에, 정상 벡터는 그대로 쓴다.
                full_text, doc_ranges, sentence_texts = (
                    semantic_chunker.prepare_overflow_texts(overflow_docs)
                )
                # 정상 페이지는 임베딩 1회로 벡터화 후 순서 보존 병합.
                normal_docs = [d for d in docs if len(d.page_content) <= threshold]
                normal_texts = [d.page_content for d in normal_docs]
                combined_texts = [*sentence_texts, *normal_texts]
                split_docs = []
                vectors = []
                if combined_texts:
                    all_vectors = await semantic_chunker.embed_texts(combined_texts)
                    n_sent = len(sentence_texts)
                    normal_vectors = [np.array(v) for v in all_vectors[n_sent:]]
                    if n_sent:
                        sent_vectors = semantic_chunker.l2_normalize(
                            all_vectors[:n_sent]
                        )
                        (
                            split_docs,
                            vectors,
                        ) = await semantic_chunker.assemble_overflow_chunks(
                            full_text, doc_ranges, sent_vectors
                        )
                    # 순서 보존: 원본 docs 순서대로 재조립
                    normal_iter = iter(zip(normal_docs, normal_vectors, strict=False))
                    # overflow 의미분할은 원본 1페이지가 N청크가 되므로
                    # 페이지 단위 순서로 병합 (정상-초과 교차 순서는
                    # 원본 페이지 순서 기준 근사).
                    merged_docs: list[Document] = []
                    merged_vectors: list[np.ndarray] = []
                    overflow_set = {id(d) for d in overflow_docs}
                    pending_overflow = list(zip(split_docs, vectors, strict=False))
                    for d in docs:
                        if id(d) in overflow_set:
                            continue
                        nd, nv = next(normal_iter)
                        merged_docs.append(nd)
                        merged_vectors.append(nv)
                    for od, ov in pending_overflow:
                        merged_docs.append(od)
                        merged_vectors.append(ov)
                    split_docs, vectors = merged_docs, merged_vectors
                msg = f"오버플로우 하위분할 완료 ({len(split_docs)}개 청크 생성)"
                SessionManager.add_status_log(msg, session_id=session_id)
                logger.info(
                    f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
                )
                _postprocess_metadata(split_docs)
                return split_docs, vectors
            # 의미분할 비활성/임베더 없음: 초과분만 Recursive로 분할
            recursive_chunker = RecursiveCharacterTextSplitter(
                chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
            )
            split_docs = []
            for d in docs:
                if len(d.page_content) > threshold:
                    split_docs.extend(
                        await asyncio.to_thread(recursive_chunker.split_documents, [d])
                    )
                else:
                    split_docs.append(d)
            if embedder and vectors is None:
                vectors = await _embed_documents_chunks(split_docs, embedder)
            msg = f"오버플로우 표준분할 완료 ({len(split_docs)}개 청크 생성)"
            SessionManager.add_status_log(msg, session_id=session_id)
            logger.info(
                f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
            )
            _postprocess_metadata(split_docs)
            return split_docs, vectors

        # P0-2 opt-in: chonkie 헤더-어웨어 어댑터. 미설치/실패 시 builtin 폴백.
        if chunk_backend == "chonkie":
            try:
                from core.chonkie_adapter import (
                    chunk_with_chonkie,
                    is_chonkie_available,
                )

                if is_chonkie_available():
                    with get_performance_monitor().track_operation(
                        OperationType.SEMANTIC_CHUNKING,
                        {"doc_count": len(docs), "backend": "chonkie"},
                    ):
                        split_docs = await asyncio.to_thread(chunk_with_chonkie, docs)
                        if embedder:
                            vectors = await _embed_documents_chunks(
                                split_docs, embedder
                            )
                    msg = f"chonkie 의미 분할 완료 ({len(split_docs)}개 청크 생성)"
                    SessionManager.add_status_log(msg, session_id=session_id)
                    logger.info(
                        f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
                    )
                    _postprocess_metadata(split_docs)
                    return split_docs, vectors
                logger.warning("[CHUNKING] chonkie 미설치로 builtin 폴백")
            except Exception as e:
                logger.warning(f"[CHUNKING] chonkie 실패로 builtin 폴백: {e}")

        if use_semantic and embedder:
            with get_performance_monitor().track_operation(
                OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
            ):
                semantic_chunker = _init_semantic_chunker(embedder)
                split_docs, vectors = await semantic_chunker.split_documents(docs)
                msg = f"의미론적 분할 완료 ({len(split_docs)}개 청크 생성)"
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
            msg = f"표준 분할 완료 ({len(split_docs)}개 청크 생성)"

        SessionManager.add_status_log(msg, session_id=session_id)
        logger.info(
            f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
        )

    _postprocess_metadata(split_docs)
    return split_docs, vectors
