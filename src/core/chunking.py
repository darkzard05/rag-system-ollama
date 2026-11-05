"""
문서 분할(Chunking)을 담당하는 모듈.
"""

import asyncio
import logging
import os
import re
import threading
import time
from functools import lru_cache
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import (
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
)
from common.logging_config import silence_transformers_auto_docstring
from common.text_utils import count_tokens_rough, split_by_headers
from core.model_loader import is_ollama_embedder
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.session import SessionManager
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)

# 임베딩 모델 컨텍스트(v2-moe 512)에서 안전 마진을 뺀 토큰 상한.
# 페이지 오버플로우 판정과 최종 청크 분할이 동일한 값을 사용한다.
_DEFAULT_EMBEDDING_TOKEN_LIMIT = 448
_TOKEN_OVERLAP = 64
_HF_TOKENIZER_REPO = "nomic-ai/nomic-embed-text-v2-moe"


_TOKENIZER_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def _load_hf_embedding_tokenizer() -> Any | None:
    """v2-moe 실제 토크나이저를 지연 로드한다 (오프라인/미설치 시 None).

    ``count_tokens_rough`` 는 한국어/표/수식에서 실제 HF 토큰을 1.5배 이상
    과소평가한다. 최종 토큰 상한 강제는 이 토크나이저로 실제 값을 재고,
    사용할 수 없을 때만 보수적 배수 폴백을 쓴다.
    """
    from transformers import AutoTokenizer

    # 캐시된 토크나이저를 우선 사용해 HF Hub 메타데이터 조회(수 초)를 피한다.
    for local_only in (True, False):
        try:
            return AutoTokenizer.from_pretrained(
                _HF_TOKENIZER_REPO,
                trust_remote_code=True,
                local_files_only=local_only,
            )
        except Exception:
            if local_only:
                continue
    logger.warning("[RAG] [CHUNKING] HF 토크나이저 로드 실패 — rough×1.6 폴백")
    return None


def _hf_embedding_tokenizer() -> Any | None:
    """단일 스레드 로드 보장 래퍼.

    백그라운드 워밍업과 청킹 스레드가 동시에 호출해도 HF Hub 로드가
    중복되지 않도록 직렬화한다. 로드 완료 후에는 lru_cache 히트만 남는다.
    """
    with _TOKENIZER_LOAD_LOCK:
        return _load_hf_embedding_tokenizer()


def warm_embedding_tokenizer() -> bool:
    """임베딩 토크나이저를 미리 로드한다. 성공 여부를 반환한다."""
    started = time.perf_counter()
    tokenizer = _hf_embedding_tokenizer()
    logger.info(
        "[RAG] [CHUNKING] 토크나이저 워밍업 | loaded=%s | elapsed_ms=%.1f",
        tokenizer is not None,
        (time.perf_counter() - started) * 1000,
    )
    return tokenizer is not None


async def warm_embedding_tokenizer_async() -> bool:
    """이벤트 루프를 막지 않고 토크나이저를 예열한다."""
    return await asyncio.to_thread(warm_embedding_tokenizer)


def _embedding_token_counts(texts: list[str]) -> list[int]:
    """임베딩 모델 기준 토큰 수를 배치로 계산한다. HF 실패 시 보수적 폴백."""
    tokenizer = _hf_embedding_tokenizer()
    if tokenizer is not None:
        try:
            encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
            return [len(ids) for ids in encoded]
        except Exception as e:
            logger.debug("[RAG] [CHUNKING] 배치 토큰 계산 실패 — 폴백: %s", e)
    return [int(count_tokens_rough(text) * 1.6) for text in texts]


def _embedding_token_count(text: str) -> int:
    """단일 텍스트의 임베딩 토큰 수 (스플리터 length_function용)."""
    return _embedding_token_counts([text])[0]


def _recursive_splitter_cls() -> Any:
    """``RecursiveCharacterTextSplitter`` 를 지연 임포트한다.

    ``langchain_text_splitters`` 패키지 ``__init__`` 은 ``sentence_transformers``
    서브모듈을 끌어오고, 그 경로에서 transformers ``auto_docstring`` 데코레이터가
    import 시점에 stdout print를 뿌릴 수 있다. 패키지 ``__init__`` 실행은 회피할
    수 없으므로, print 차단을 먼저 적용한 뒤 임포트한다. 또한 이 임포트는
    모듈 로드 비용이 커서 실제 분할 시점까지 미룬다.
    """
    silence_transformers_auto_docstring()
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter


# Backward-compatible alias: older tests patch/import this module-local name.
_is_ollama_embedder = is_ollama_embedder


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


def _get_token_length_fn() -> Any:
    """RecursiveCharacterTextSplitter용 초고속 토큰 길이 함수.

    [최적화] 청킹 재귀 분할 루프 내부에서 수만 번 호출되는 tok.encode()의
    무거운 C-FFI 및 메모리 할당 오버헤드를 제거하기 위해,
    실측 보정된 고속 토큰 추정 함수(count_tokens_rough)로 일원화합니다.
    """
    return count_tokens_rough


def _embedding_token_limit() -> int:
    """임베딩 입력 하드 상한(토큰). config 미설정 시 448로 폴백한다."""
    raw = TEXT_SPLITTER_CONFIG.get("embedding_token_limit")
    if raw is None:
        raw = TEXT_SPLITTER_CONFIG.get("chunk_size_tokens", 512)
        return max(64, int(raw) - _TOKEN_OVERLAP)
    return max(64, int(raw))


def _page_overflow_threshold_tokens() -> int:
    """페이지 단위 하위분할을 트리거하는 토큰 임계치."""
    raw = TEXT_SPLITTER_CONFIG.get("page_overflow_threshold_tokens")
    if raw is not None:
        return max(64, int(raw))
    return _embedding_token_limit()


def _token_percentile(values: list[int], q: float) -> int:
    """정렬된 토큰 분포에서 q 분위수를 반환한다 (검증 로그용)."""
    if not values:
        return 0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(q * len(ordered)))
    return ordered[idx]


async def _enforce_embedding_token_limit(
    docs: list[Document],
    vectors: list[np.ndarray] | None,
    embedder: Embeddings | None,
) -> tuple[list[Document], list[np.ndarray] | None]:
    """모든 임베딩 입력이 토큰 상한 이하가 되도록 초과 청크를 분할한다.

    시맨틱 청커는 문자 수(max_chunk_size)로 병합하므로 한국어에서는 512토큰을
    쉽게 넘는다. 최종 산출물을 토큰 기준으로 다시 쪼개고, 쪼갠 조각만 재임베딩해
    벡터 정합성을 유지한다. 상한 이하 청크의 기존 벡터는 그대로 재사용한다.
    """
    started_at = time.perf_counter()
    limit = _embedding_token_limit()
    rough_counts = [count_tokens_rough(d.page_content) for d in docs]
    # rough가 최악 관측 배수(1.6)를 곱해도 상한 이하이면 HF 토크나이저를
    # 로드하지 않고 즉시 반환한다 (소형 청크/테스트 경로의 지연 방지).
    if not rough_counts or max(rough_counts) * 1.6 <= limit:
        logger.info(
            "[RAG] [CHUNKING] 토큰 통계 | chunks=%d | p50=%d | p95=%d | max=%d | limit=%d | total_ms=%.1f",
            len(docs),
            _token_percentile(rough_counts, 0.5),
            _token_percentile(rough_counts, 0.95),
            max(rough_counts) if rough_counts else 0,
            limit,
            (time.perf_counter() - started_at) * 1000,
        )
        return docs, vectors

    count_started = time.perf_counter()
    token_counts = await asyncio.to_thread(
        _embedding_token_counts, [d.page_content for d in docs]
    )
    count_ms = (time.perf_counter() - count_started) * 1000
    oversized = [count for count in token_counts if count > limit]
    if not oversized:
        logger.info(
            "[RAG] [CHUNKING] 토큰 통계 | chunks=%d | p50=%d | p95=%d | max=%d | limit=%d "
            "| count_ms=%.1f | total_ms=%.1f",
            len(docs),
            _token_percentile(token_counts, 0.5),
            _token_percentile(token_counts, 0.95),
            max(token_counts) if token_counts else 0,
            limit,
            count_ms,
            (time.perf_counter() - started_at) * 1000,
        )
        return docs, vectors

    if embedder is None and vectors is not None:
        raise ValueError("embedder is required to re-embed split chunks")

    split_started = time.perf_counter()
    splitter_cls = await asyncio.to_thread(_recursive_splitter_cls)
    splitter = splitter_cls(
        chunk_size=limit,
        chunk_overlap=min(_TOKEN_OVERLAP, max(0, limit // 5)),
        length_function=_embedding_token_count,
    )

    out_docs: list[Document] = []
    vector_slots: list[np.ndarray | None] | None = [] if vectors is not None else None
    vector_iter = iter(vectors) if vectors is not None else None
    pending_parts: list[Document] = []

    for doc, token_count in zip(docs, token_counts, strict=True):
        if token_count <= limit:
            out_docs.append(doc)
            if vector_slots is not None and vector_iter is not None:
                vector_slots.append(next(vector_iter))
            continue

        parts = await asyncio.to_thread(splitter.split_documents, [doc])
        if not parts:
            parts = [doc]
        for part in parts:
            part.metadata = {**part.metadata, "split_from_oversized": True}

        out_docs.extend(parts)
        pending_parts.extend(parts)
        if vector_slots is not None:
            vector_slots.extend([None] * len(parts))
        if vector_iter is not None:
            next(vector_iter, None)

    split_ms = (time.perf_counter() - split_started) * 1000

    # 모든 재분할 조각을 한 번의 임베딩 라운드로 처리한다. 조각별로 호출하면
    # 오버플로우 페이지 수만큼 Ollama 왕복이 늘어 인덱싱이 선형으로 느려진다.
    out_vectors: list[np.ndarray] | None = None
    reembed_started = time.perf_counter()
    if vector_slots is not None:
        if pending_parts:
            if embedder is None:
                raise ValueError("embedder is required to re-embed split chunks")
            part_vectors = iter(await _embed_documents_chunks(pending_parts, embedder))
            out_vectors = [
                next(part_vectors) if vec is None else vec for vec in vector_slots
            ]
        else:
            out_vectors = [vec for vec in vector_slots if vec is not None]
    reembed_ms = (time.perf_counter() - reembed_started) * 1000

    final_count_started = time.perf_counter()
    final_counts = await asyncio.to_thread(
        _embedding_token_counts, [d.page_content for d in out_docs]
    )
    final_count_ms = (time.perf_counter() - final_count_started) * 1000
    logger.warning(
        "[RAG] [CHUNKING] 임베딩 토큰 상한 강제 | limit=%d | oversized=%d | "
        "chunks=%d→%d | max_before=%d | max_after=%d | count_ms=%.1f | "
        "split_ms=%.1f | reembed_ms=%.1f | final_count_ms=%.1f | total_ms=%.1f",
        limit,
        len(oversized),
        len(docs),
        len(out_docs),
        max(token_counts) if token_counts else 0,
        max(final_counts) if final_counts else 0,
        count_ms,
        split_ms,
        reembed_ms,
        final_count_ms,
        (time.perf_counter() - started_at) * 1000,
    )
    return out_docs, out_vectors


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
        max_chunk_tokens=_embedding_token_limit(),
        token_count_fn=_embedding_token_counts,
    )


def _postprocess_metadata(split_docs: list[Document]) -> None:
    """청크별 메타데이터 정리 및 내용 유형(콘텐츠/참고문헌 등) 식별"""
    ref_start_idx = None
    noise_keywords = ["index", "references", "bibliography", "doi:", "isbn"]
    # [Phase2] 단일 `#` 헤더 포함 ("# REFERENCES" 미검출 수정 — 베이스라인 ref 0건).
    ref_keywords = [
        "## references",
        "# references",
        "# reference",
        "# bibliography",
        "references\n---",
        "## 참고문헌",
        "참고문헌\n---",
    ]

    for i, doc in enumerate(split_docs):
        content_lower = doc.page_content.lower()
        # [Phase3] 메타데이터 기반 검출: chonkie 경로는 헤더 행이 본문에 없어
        # content 매칭이 불가하므로 섹션명도 함께 본다.
        section_lower = str(
            doc.metadata.get("section_path", "")
            or doc.metadata.get("current_section", "")
        ).lower()

        if ref_start_idx is None and (
            any(kw in content_lower[:50] for kw in ref_keywords)
            # Tier1 P0-3a: 부분문자열 매칭("refer" in section)은 "Preferred Methods",
            # "Referred" 등을 오탐한다. 단어 경계 기준으로 축소.
            or re.search(r"\brefer(ence)?s?\b", section_lower) is not None
            or "bibliography" in section_lower
        ):
            ref_start_idx = i

        doc.metadata = doc.metadata.copy()
        doc.metadata["chunk_index"] = i
        # [PoC-1] 내부 귀속 키는 출력 메타에서 제거한다.
        doc.metadata.pop("_overflow_parent", None)
        # [Phase2] section_path 1레벨 (2단계 cap 구조 준비, 값=현 섹션).
        # citations[].section 키 유지·값만 교체, 렌더는 section_path 우선.
        if "current_section" in doc.metadata:
            doc.metadata["section_path"] = doc.metadata.get("current_section", "")

        is_noise = any(kw in content_lower[:100] for kw in noise_keywords)
        # Tier1 P0-3b: 콤마/doi 개수 규칙은 색인·참고문헌 같은 짧은 목록 페이지용이다.
        # 긴 본문(사양표·수식 포함)이 콤마가 많다는 이유만으로 통째로 버려지는
        # 과삭제를 막기 위해 개수 규칙은 1000자 미만 문서에만 적용한다.
        # (짧은 데이터 표 오탐은 잔여 — 색인과의 분리는 별도 측정 필요.)
        if (
            not is_noise
            and len(content_lower) < 1000
            and (content_lower.count("doi:") > 2 or content_lower.count(",") > 25)
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

    # 로그/조기 경고용 근사치. 실제 상한 강제는 _enforce 단계에서 HF로 재검증한다.
    token_counts = [count_tokens_rough(d.page_content) for d in chunks]
    max_tokens = max(token_counts) if token_counts else 0
    limit = _embedding_token_limit()
    logger.info(
        "[RAG] [EMBED] inputs=%d max_rough_tokens=%d limit=%d",
        len(chunks),
        max_tokens,
        limit,
    )
    if max_tokens > limit:
        logger.warning(
            "[RAG] [EMBED] 토큰 상한 초과 입력 감지 | max=%d limit=%d — "
            "최종 청킹 단계에서 분할 예정",
            max_tokens,
            limit,
        )

    coordinator = get_resource_manager()
    async with coordinator.use_embedder(embedder=embedder):
        raw_vectors = await asyncio.to_thread(
            embedder.embed_documents, [d.page_content for d in chunks]
        )
    return [np.array(v) for v in raw_vectors]


def _get_chonkie_chunk_fn() -> Any:
    """config chonkie.* 파라미터를 바인딩한 청킹 partial을 반환합니다."""
    import functools

    from core.chonkie_adapter import chunk_with_chonkie

    _ccfg = SEMANTIC_CHUNKER_CONFIG.get("chonkie", {})
    return functools.partial(
        chunk_with_chonkie,
        chunk_size=int(_ccfg.get("chunk_size", 512)),
        threshold=float(_ccfg.get("threshold", 0.7)),
        similarity_window=int(_ccfg.get("similarity_window", 3)),
        min_sentences_per_chunk=int(_ccfg.get("min_sentences_per_chunk", 2)),
        skip_window=int(_ccfg.get("skip_window", 0)),
        filter_window=int(_ccfg.get("filter_window", 5)),
        filter_polyorder=int(_ccfg.get("filter_polyorder", 3)),
        filter_tolerance=float(_ccfg.get("filter_tolerance", 0.2)),
        overlap_ratio=float(_ccfg.get("overlap_ratio", 0.1)),
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

    async def _finalize() -> tuple[list[Document], list[np.ndarray] | None]:
        """토큰 상한 강제 + 메타데이터 후처리를 한 곳에서 수행한다."""
        nonlocal split_docs, vectors
        split_docs, vectors = await _enforce_embedding_token_limit(
            split_docs, vectors, embedder
        )
        _postprocess_metadata(split_docs)
        return split_docs, vectors

    # [최적화] 페이지 단위로 이미 분할되었더라도 토큰 상한을 넘으면 재분할한다.
    # 문자 수 기준은 한국어에서 실제 토큰을 과소평가하므로 사용하지 않는다.
    page_overflow_threshold = _page_overflow_threshold_tokens()
    needs_sub_chunking = is_already_chunked and any(
        count_tokens_rough(d.page_content) > page_overflow_threshold for d in docs
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
        # [Phase C1] 오버플로우 페이지만 하위분할: 일반 페이지(1,200~1,800자)가
        # 낮은 임계치에 걸려 전체가 의미분할(문장 임베딩 수백 회)로 보내지던
        # 과발동을 수정한다. 정상 페이지는 그대로 재사용하고, 초과분만
        # 의미분할한다. 임계치는 config의 embedding_token_limit를 따른다.
        # (신규 의존성 없이 기존 builtin 경로만 사용. 벡터 캐시 복구 없음.)
        if needs_sub_chunking and is_already_chunked:
            threshold = page_overflow_threshold
            overflow_docs = [
                d for d in docs if count_tokens_rough(d.page_content) > threshold
            ]
            logger.info(
                "[RAG] [CHUNKING] 오버플로우 하위분할 | 전체: %d | 초과: %d",
                len(docs),
                len(overflow_docs),
            )
            if use_semantic and embedder and overflow_docs:
                semantic_chunker = _init_semantic_chunker(embedder)
                # [Phase2] 헤더 사전분할: 오버플로우 페이지를 섹션 조각으로
                # 나눠 의미분할에 전달 — 섹션 경계가 임베딩 결합 텍스트에
                # 묻히지 않고 doc_ranges 메타로 보존된다. 조각 없는 페이지는
                # 원본 그대로 전달한다.
                overflow_sections: list[Document] = []
                for d in overflow_docs:
                    base_meta = dict(d.metadata or {})
                    # [PoC-1] 동일 page 공유 시 by_page 붕괴 방지: 원본 문서
                    # 귀속 식별자를 섹션 메타에 기록한다. 청크 메타 역매핑이
                    # 첫 겹침 메타를 복사하므로 cross-section 청크도 첫 원본에
                    # 귀속되어 유실 없이 순서가 보존된다.
                    base_meta["_overflow_parent"] = id(d)
                    sections = split_by_headers(d.page_content)
                    if len(sections) <= 1:
                        overflow_sections.append(
                            Document(
                                page_content=d.page_content,
                                metadata=dict(base_meta),
                            )
                        )
                    else:
                        for title, body in sections:
                            if not body.strip():
                                continue
                            # 헤더 행을 본문에 유지 — 결합 시 헤더 검출용.
                            # (제거하면 _join_docs_text의 공백 결합에 묻혀
                            # HEADER_SIMPLE이 매치되지 않고 전역 Front Matter로 붕괴.)
                            text = f"# {title}\n{body}" if title else body
                            meta = {
                                **base_meta,
                                "current_section": title
                                or base_meta.get("current_section", ""),
                            }
                            overflow_sections.append(
                                Document(page_content=text, metadata=meta)
                            )
                # [Phase3] chonkie 위임: 백엔드가 chonkie면 섹션 조각을
                # chonkie 의미분할에 위임한다 (v2-moe 임베딩, L3 캐시 경유).
                # 실패 시 builtin 경로로 폴백한다.
                if chunk_backend == "chonkie":
                    try:
                        from core.chonkie_adapter import is_chonkie_available

                        if is_chonkie_available():
                            with get_performance_monitor().track_operation(
                                OperationType.SEMANTIC_CHUNKING,
                                {
                                    "doc_count": len(docs),
                                    "backend": "chonkie-overflow",
                                },
                            ):
                                _chunk_fn = _get_chonkie_chunk_fn()
                                chonk_docs = await asyncio.to_thread(
                                    _chunk_fn, overflow_sections
                                )
                                normal_docs = [
                                    d
                                    for d in docs
                                    if count_tokens_rough(d.page_content) <= threshold
                                ]
                                chonk_vectors = await _embed_documents_chunks(
                                    chonk_docs, embedder
                                )
                                normal_vectors = (
                                    await _embed_documents_chunks(normal_docs, embedder)
                                    if normal_docs
                                    else []
                                )
                            # 순서 보존: 원본 페이지 위치에 끼워넣기.
                            # [PoC-1] page 키 단독 귀속은 동일 page 공유 시 첫 문서가
                            # 전부 소비한다. _overflow_parent 우선, page 폴백으로
                            # 원본별 큐를 분리한다.
                            co_merged_docs: list[Document] = []
                            co_merged_vectors: list[np.ndarray] = []
                            co_overflow_set = {id(d) for d in overflow_docs}
                            co_by_parent: dict[
                                Any, list[tuple[Document, np.ndarray]]
                            ] = {}
                            co_page_fallback: dict[
                                Any, list[tuple[Document, np.ndarray]]
                            ] = {}
                            for od, ov in zip(chonk_docs, chonk_vectors, strict=True):
                                parent = od.metadata.get("_overflow_parent")
                                if parent is not None:
                                    co_by_parent.setdefault(parent, []).append((od, ov))
                                else:
                                    co_page_fallback.setdefault(
                                        od.metadata.get("page"), []
                                    ).append((od, ov))
                            co_normal_iter = iter(
                                zip(normal_docs, normal_vectors, strict=False)
                            )
                            for d in docs:
                                if id(d) in co_overflow_set:
                                    bucket = co_by_parent.pop(id(d), [])
                                    if not bucket:
                                        bucket = co_page_fallback.pop(
                                            d.metadata.get("page"), []
                                        )
                                    for od, ov in bucket:
                                        co_merged_docs.append(od)
                                        co_merged_vectors.append(ov)
                                else:
                                    nd, nv = next(co_normal_iter)
                                    co_merged_docs.append(nd)
                                    co_merged_vectors.append(nv)
                            for _key, rest in list(co_by_parent.items()) + list(
                                co_page_fallback.items()
                            ):
                                for od, ov in rest:
                                    co_merged_docs.append(od)
                                    co_merged_vectors.append(ov)
                            split_docs, vectors = co_merged_docs, co_merged_vectors
                            msg = (
                                "chonkie 오버플로우 분할 완료 "
                                f"({len(split_docs)}개 청크 생성)"
                            )
                            SessionManager.add_status_log(msg, session_id=session_id)
                            logger.info(
                                "[RAG] [CHUNKING] 분할 완료 | "
                                f"원본: {len(docs)} | 청크: {len(split_docs)}"
                            )
                            return await _finalize()
                        logger.warning("[CHUNKING] chonkie 미설치로 builtin 폴백")
                    except Exception as e:
                        logger.warning(f"[CHUNKING] chonkie 실패로 builtin 폴백: {e}")
                # [최적화] 단일 임베딩 왕복: 오버플로우 문장 텍스트와 정상 청크
                # 텍스트를 합쳐 embed_documents 1회로 처리한다. Ollama는 입력
                # 전체를 단일 HTTP 요청으로 전송하므로 분할 호출은 왕복만 늘린다.
                # 문장 벡터는 l2 정규화 후 경계 탐색에, 정상 벡터는 그대로 쓴다.
                full_text, doc_ranges, sentence_texts = (
                    semantic_chunker.prepare_overflow_texts(overflow_sections)
                )
                # 정상 페이지는 임베딩 1회로 벡터화 후 순서 보존 병합.
                # 토큰 임계치 이하이므로 절삭(truncate) 없이 원문을 그대로 쓴다.
                normal_docs = [
                    d for d in docs if count_tokens_rough(d.page_content) <= threshold
                ]
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
                    # 순서 보존: 원본 docs 순서대로 재조립 — 오버플로우 청크를
                    # 원본 페이지 위치에 끼워넣는다 (기존: 정상 전부 + 초과
                    # append로 문서 순서 파괴). [PoC-1] _overflow_parent 우선
                    # 귀속으로 동일 page 공유 시 붕괴를 방지한다.
                    normal_iter = iter(zip(normal_docs, normal_vectors, strict=False))
                    merged_docs: list[Document] = []
                    merged_vectors: list[np.ndarray] = []
                    overflow_set = {id(d) for d in overflow_docs}
                    pending_overflow = list(zip(split_docs, vectors, strict=False))
                    by_parent: dict[Any, list[tuple[Document, np.ndarray]]] = {}
                    page_fallback: dict[Any, list[tuple[Document, np.ndarray]]] = {}
                    for od, ov in pending_overflow:
                        parent = od.metadata.get("_overflow_parent")
                        if parent is not None:
                            by_parent.setdefault(parent, []).append((od, ov))
                        else:
                            pg = od.metadata.get("page")
                            page_fallback.setdefault(pg, []).append((od, ov))
                    for d in docs:
                        if id(d) in overflow_set:
                            bucket = by_parent.pop(id(d), [])
                            if not bucket:
                                bucket = page_fallback.pop(d.metadata.get("page"), [])
                            for od, ov in bucket:
                                merged_docs.append(od)
                                merged_vectors.append(ov)
                        else:
                            nd, nv = next(normal_iter)
                            merged_docs.append(nd)
                            merged_vectors.append(nv)
                    # 귀속 실패 잔여(페이지 미일치)는 말미에 보존 (유실 방지).
                    for _key, rest in list(by_parent.items()) + list(
                        page_fallback.items()
                    ):
                        for od, ov in rest:
                            merged_docs.append(od)
                            merged_vectors.append(ov)
                    split_docs, vectors = merged_docs, merged_vectors
                msg = f"오버플로우 하위분할 완료 ({len(split_docs)}개 청크 생성)"
                SessionManager.add_status_log(msg, session_id=session_id)
                logger.info(
                    f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
                )
                return await _finalize()
            # 의미분할 비활성/임베더 없음: 초과분만 Recursive로 분할
            _token_limit = _embedding_token_limit()
            recursive_chunker = _recursive_splitter_cls()(
                chunk_size=_token_limit,
                chunk_overlap=min(_TOKEN_OVERLAP, max(0, _token_limit // 5)),
                length_function=_get_token_length_fn(),
            )
            split_docs = []
            for d in docs:
                if count_tokens_rough(d.page_content) > threshold:
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
            return await _finalize()

        # P0-2 opt-in: chonkie 헤더-어웨어 어댑터. 미설치/실패 시 builtin 폴백.
        if chunk_backend == "chonkie":
            try:
                from core.chonkie_adapter import is_chonkie_available

                if is_chonkie_available():
                    with get_performance_monitor().track_operation(
                        OperationType.SEMANTIC_CHUNKING,
                        {"doc_count": len(docs), "backend": "chonkie"},
                    ):
                        # [Phase3] config chonkie.* 파라미터 전달 (스위프 대상 포함).
                        _chunk_fn = _get_chonkie_chunk_fn()
                        split_docs = await asyncio.to_thread(_chunk_fn, docs)
                        if embedder:
                            vectors = await _embed_documents_chunks(
                                split_docs, embedder
                            )
                    msg = f"chonkie 의미 분할 완료 ({len(split_docs)}개 청크 생성)"
                    SessionManager.add_status_log(msg, session_id=session_id)
                    logger.info(
                        f"[RAG] [CHUNKING] 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
                    )
                    return await _finalize()
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
            _token_limit = _embedding_token_limit()
            recursive_chunker = _recursive_splitter_cls()(
                chunk_size=_token_limit,
                chunk_overlap=min(_TOKEN_OVERLAP, max(0, _token_limit // 5)),
                length_function=_get_token_length_fn(),
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

    return await _finalize()
