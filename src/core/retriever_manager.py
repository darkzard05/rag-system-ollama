"""
벡터 저장소(FAISS) 및 키워드 리트리버(BM25)의 생성, 캐싱, 관리를 담당하는 모듈.
"""

import functools
import hashlib
import json
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    import torch

from common.config import (
    CACHE_CHECK_PERMISSIONS,
    CACHE_HMAC_SECRET,
    CACHE_SECURITY_LEVEL,
    CACHE_TRUSTED_PATHS,
    RETRIEVER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from common.exceptions import (
    EmptyPDFError,
    InsufficientChunksError,
)
from common.text_utils import bm25_tokenizer
from common.typing_utils import (
    DocumentDictList,
    DocumentList,
    T,
)
from common.utils import log_operation
from core.document_processor import compute_file_hash, load_pdf_docs
from core.graph_builder import build_graph
from core.semantic_chunker import EmbeddingBasedSemanticChunker
from core.session import SessionManager
from security.cache_security import (
    CacheIntegrityError,
    CacheSecurityManager,
    CacheTrustError,
)
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)
from services.optimization.index_optimizer import get_index_optimizer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


async def split_documents(
    docs: list[Document],
    embedder: Embeddings | None = None,
    session_id: str | None = None,
) -> tuple[list[Document], list[np.ndarray] | None]:
    """
    설정에 따라 의미론적 분할기 또는 RecursiveCharacterTextSplitter를 사용해 문서를 분할합니다.
    이미 분할된 문서(Docling 등)는 분할 과정을 건너뜁니다.
    """
    import torch

    is_already_chunked = (
        docs[0].metadata.get("is_already_chunked", False) if docs else False
    )

    if is_already_chunked:
        SessionManager.add_status_log(
            "이미 구조적으로 분할된 문서를 사용합니다.", session_id=session_id
        )
        logger.info(f"[RAG] [CHUNKING] 분할 건너뜀 (이미 분할됨) | 청크: {len(docs)}")
        split_docs = docs
        vectors = None
        if embedder and split_docs:
            texts = [d.page_content for d in split_docs]
            vectors_list = embedder.embed_documents(texts)
            vectors = [np.array(v) for v in vectors_list]
    else:
        SessionManager.add_status_log("문장 분할 중...", session_id=session_id)
        with monitor.track_operation(
            OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
        ) as _:
            from langchain_text_splitters import RecursiveCharacterTextSplitter

            use_semantic = SEMANTIC_CHUNKER_CONFIG.get("enabled", False)
            split_docs = []
            vectors = None

            if use_semantic and embedder:
                if getattr(embedder, "model_kwargs", {}).get("device") == "cuda":
                    try:
                        total_mem = torch.cuda.get_device_properties(0).total_memory / (
                            1024**3
                        )  # GB
                        batch_size = (
                            128 if total_mem > 10 else (64 if total_mem > 4 else 32)
                        )
                    except Exception:
                        batch_size = 32
                else:
                    batch_size = min(max(4, os.cpu_count() or 4), 16)

                semantic_chunker = EmbeddingBasedSemanticChunker(
                    embedder=embedder,
                    breakpoint_threshold_type=SEMANTIC_CHUNKER_CONFIG.get(
                        "breakpoint_threshold_type", "percentile"
                    ),
                    breakpoint_threshold_value=float(
                        SEMANTIC_CHUNKER_CONFIG.get("breakpoint_threshold_value", 95.0)
                    ),
                    sentence_split_regex=SEMANTIC_CHUNKER_CONFIG.get(
                        "sentence_split_regex", r"[.!?]\s+"
                    ),
                    min_chunk_size=int(
                        SEMANTIC_CHUNKER_CONFIG.get("min_chunk_size", 100)
                    ),
                    max_chunk_size=int(
                        SEMANTIC_CHUNKER_CONFIG.get("max_chunk_size", 800)
                    ),
                    similarity_threshold=float(
                        SEMANTIC_CHUNKER_CONFIG.get("similarity_threshold", 0.5)
                    ),
                    batch_size=batch_size,
                )

                split_docs, vectors = await semantic_chunker.split_documents(docs)
                logger.info(
                    f"[RAG] [CHUNKING] 의미론적 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
                )
            else:
                chunker = RecursiveCharacterTextSplitter(
                    chunk_size=TEXT_SPLITTER_CONFIG["chunk_size"],
                    chunk_overlap=TEXT_SPLITTER_CONFIG["chunk_overlap"],
                )
                split_docs = chunker.split_documents(docs)
                logger.info(
                    f"[RAG] [CHUNKING] 기본 분할 완료 | 원본: {len(docs)} | 청크: {len(split_docs)}"
                )

                if embedder and split_docs:
                    texts = [d.page_content for d in split_docs]
                    vectors_list = embedder.embed_documents(texts)
                    vectors = [np.array(v) for v in vectors_list]

    noise_keywords = ["index", "references", "bibliography", "doi:", "isbn"]
    found_ref_start = False
    for i, doc in enumerate(split_docs):
        doc.metadata = doc.metadata.copy()
        doc.metadata["chunk_index"] = i

        content_lower = doc.page_content.lower()

        # [추가] 참고문헌 섹션 전파 로직
        if doc.metadata.get("is_reference_start") or any(
            kw in content_lower[:50] for kw in ["## references", "references\n---"]
        ):
            found_ref_start = True

        is_noise = any(kw in content_lower[:100] for kw in noise_keywords)
        if not is_noise and (
            content_lower.count("doi:") > 2 or content_lower.count(",") > 25
        ):
            is_noise = True

        doc.metadata["is_content"] = not (is_noise or found_ref_start)
        doc.metadata["is_reference"] = found_ref_start

        # [최적화] 첫 페이지 상단만 앵커로 유지 (나머지는 제거)
        if doc.metadata.get("is_anchor") and i > 0:
            doc.metadata["is_anchor"] = False

        # [최적화] 첫 페이지 헤더 명시 (FACTOID 답변율 향상)
        if doc.metadata.get("page") == 1 and i < 3:
            doc.metadata["is_header"] = True

    # track_operation op.tokens는 is_already_chunked일 때 context를 벗어날 수 있으므로 안전하게 처리
    return split_docs, vectors


def _serialize_docs(docs: DocumentList) -> DocumentDictList:
    """Pydantic의 무거운 dict() 대신 직접 필요한 필드만 추출"""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def _deserialize_docs(docs_as_dicts: DocumentDictList) -> DocumentList:
    return [Document(**d) for d in docs_as_dicts]


@functools.lru_cache(maxsize=1)
def _compute_config_hash() -> str:
    """설정 변경 감지용 해시 생성"""
    config_dict = {
        "version": "2.1",
        "semantic_chunker": SEMANTIC_CHUNKER_CONFIG,
        "text_splitter": TEXT_SPLITTER_CONFIG,
        "retriever": RETRIEVER_CONFIG,
    }
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:12]


class VectorStoreCache:
    """벡터 저장소 및 리트리버 캐시 관리자"""

    _global_write_lock = threading.Lock()

    def __init__(
        self, file_path: str, embedding_model_name: str, file_hash: str | None = None
    ):
        self._file_hash = file_hash or compute_file_hash(file_path)
        (
            self.cache_dir,
            self.doc_splits_path,
            self.faiss_index_path,
            self.bm25_retriever_path,
        ) = self._get_cache_paths(self._file_hash, embedding_model_name)

        self.security_manager = CacheSecurityManager(
            security_level=CACHE_SECURITY_LEVEL,
            hmac_secret=CACHE_HMAC_SECRET,
            trusted_paths=CACHE_TRUSTED_PATHS,
            check_permissions=CACHE_CHECK_PERMISSIONS,
        )

    def _get_cache_paths(
        self, file_hash: str, embedding_model_name: str
    ) -> tuple[str, str, str, str]:
        model_name_slug = embedding_model_name.replace("/", "_").replace(":", "_")
        config_hash = _compute_config_hash()
        cache_dir = os.path.join(
            VECTOR_STORE_CACHE_DIR, f"{file_hash}_{model_name_slug}_{config_hash}"
        )
        return (
            cache_dir,
            os.path.join(cache_dir, "doc_splits.json"),
            os.path.join(cache_dir, "faiss_index"),
            os.path.join(cache_dir, "bm25_docs.json"),
        )

    def _purge_cache(self, reason: str):
        if os.path.exists(self.cache_dir):
            try:
                import shutil

                shutil.rmtree(self.cache_dir)
                logger.critical(
                    f"[Security] 캐시 강제 삭제됨 ({reason}): {self.cache_dir}"
                )
            except Exception as e:
                logger.error(f"캐시 삭제 실패: {e}")

    def load(
        self,
        embedder: Embeddings,
    ) -> tuple[list[Document] | None, Any | None, Any | None]:
        if not all(
            os.path.exists(p)
            for p in [
                self.doc_splits_path,
                self.faiss_index_path,
                self.bm25_retriever_path,
            ]
        ):
            return None, None, None

        try:
            import orjson
            from langchain_community.retrievers import BM25Retriever
            from langchain_community.vectorstores import FAISS

            paths_to_verify = [
                (self.doc_splits_path, "문서 데이터"),
                (self.faiss_index_path, "FAISS 인덱스"),
                (self.bm25_retriever_path, "BM25 리트리버"),
            ]

            for path, desc in paths_to_verify:
                try:
                    self.security_manager.verify_cache_trust(path)
                    if os.path.isfile(path):
                        if CACHE_SECURITY_LEVEL == "high" and not CACHE_HMAC_SECRET:
                            raise CacheIntegrityError(
                                f"보안 레벨 'high'에서는 HMAC 비밀키가 필수입니다: {path}"
                            )
                        self.security_manager.verify_cache_integrity(path)
                    elif os.path.isdir(path):
                        index_file = os.path.join(path, "index.faiss")
                        if os.path.exists(index_file):
                            self.security_manager.verify_cache_integrity(index_file)
                except (CacheTrustError, CacheIntegrityError) as e:
                    self._purge_cache(
                        reason=f"Security Violation in {desc}: {type(e).__name__}"
                    )
                    return None, None, None

            with open(self.doc_splits_path, "rb") as f:
                doc_dicts = orjson.loads(f.read())
            doc_splits = _deserialize_docs(doc_dicts)

            vector_store = FAISS.load_local(
                self.faiss_index_path,
                embedder,
                allow_dangerous_deserialization=True,
            )

            with open(self.bm25_retriever_path, "rb") as f:
                bm25_doc_dicts = orjson.loads(f.read())
            bm25_docs = _deserialize_docs(bm25_doc_dicts)

            # [개선] preprocess_func가 bm25_tokenizer임을 명시적으로 보장
            bm25_retriever = BM25Retriever.from_documents(
                bm25_docs, preprocess_func=bm25_tokenizer
            )
            bm25_retriever.k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)

            logger.info(f"RAG 캐시 안전 로드 완료 (JSON 기반): '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever

        except Exception as e:
            logger.warning(f"캐시 로드 중 예외 발생: {e}. 캐시를 폐기합니다.")
            self._purge_cache(reason="Load Error / Corruption")
            return None, None, None

    def save(
        self,
        doc_splits: DocumentList,
        vector_store: Any,
        bm25_retriever: Any,
    ) -> None:
        import shutil
        import uuid

        import orjson

        if os.path.exists(self.cache_dir):
            logger.info(f"[Cache] 캐시가 이미 존재함: {self.cache_dir}")
            return

        staging_dir = f"{self.cache_dir}.tmp.{uuid.uuid4().hex[:8]}"
        stg_doc_splits_path = os.path.join(staging_dir, "doc_splits.json")
        stg_faiss_index_path = os.path.join(staging_dir, "faiss_index")
        stg_bm25_retriever_path = os.path.join(staging_dir, "bm25_docs.json")

        try:
            os.makedirs(staging_dir, exist_ok=True)
            self.security_manager.enforce_directory_permissions(staging_dir)

            serialized_splits = _serialize_docs(doc_splits)
            with open(stg_doc_splits_path, "wb") as f:
                f.write(orjson.dumps(serialized_splits))
            self.security_manager.enforce_file_permissions(stg_doc_splits_path)

            doc_meta = self.security_manager.create_metadata_for_file(
                stg_doc_splits_path, description="Document splits cache (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_doc_splits_path + ".meta", doc_meta
            )

            vector_store.save_local(stg_faiss_index_path)
            self.security_manager.enforce_directory_permissions(stg_faiss_index_path)

            for filename in ["index.faiss", "index.pkl"]:
                file_p = os.path.join(stg_faiss_index_path, filename)
                if os.path.exists(file_p):
                    self.security_manager.enforce_file_permissions(file_p)
                    meta = self.security_manager.create_metadata_for_file(
                        file_p, description=f"FAISS index part: {filename}"
                    )
                    self.security_manager.save_cache_metadata(file_p + ".meta", meta)

            bm25_docs = getattr(bm25_retriever, "docs", doc_splits)
            serialized_bm25 = _serialize_docs(bm25_docs)
            with open(stg_bm25_retriever_path, "wb") as f:
                f.write(orjson.dumps(serialized_bm25))
            self.security_manager.enforce_file_permissions(stg_bm25_retriever_path)

            bm25_meta = self.security_manager.create_metadata_for_file(
                stg_bm25_retriever_path, description="BM25 retriever data (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_bm25_retriever_path + ".meta", bm25_meta
            )

            with self._global_write_lock:
                if not os.path.exists(self.cache_dir):
                    os.rename(staging_dir, self.cache_dir)
                    # 최종 디렉토리 권한 확인
                    self.security_manager.enforce_directory_permissions(self.cache_dir)
                    logger.info(
                        f"RAG 캐시 원자적 저장 및 검증 완료: '{self.cache_dir}'"
                    )
                else:
                    shutil.rmtree(staging_dir)
        except Exception as e:
            logger.error(f"캐시 저장 중 예외 발생: {e}")
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
            raise


@log_operation("FAISS 벡터 저장소 생성")
def create_vector_store(
    docs: list[Document],
    embedder: Embeddings,
    vectors: Any = None,
) -> Any:
    import uuid

    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy

    if vectors is None:
        logger.warning("[FAISS] 전달된 벡터가 없어 임베딩을 다시 수행합니다.")
        texts = [d.page_content for d in docs]
        vectors_list = embedder.embed_documents(texts)
        vectors = np.array(vectors_list).astype("float32")
    else:
        if isinstance(vectors, list):
            vectors = np.vstack(vectors).astype("float32")
        else:
            vectors = np.ascontiguousarray(vectors, dtype="float32")

    # ✅ [최적화] GPU 자동 감지 및 설정 (Phase 5 고급 최적화)
    use_gpu = False
    gpu_device = 0
    try:
        # GPU 가용성 확인
        if torch.cuda.is_available():
            ngpus = faiss.get_num_gpus()
            if ngpus > 0:
                use_gpu = True
                gpu_device = torch.cuda.current_device()
                logger.info(
                    f"[FAISS GPU] 활성화 (Device: {gpu_device}, Count: {ngpus})"
                )
    except Exception as e:
        logger.debug(f"[FAISS GPU] 자동 감지 실패: {e}. CPU 모드로 진행합니다.")
        use_gpu = False

    # L2 정규화 (대규모 벡터는 필수)
    if len(vectors) > 1000:
        faiss.normalize_L2(vectors)
        logger.debug("[FAISS] L2 정규화 완료")

    chunk_count = len(docs)
    d = vectors.shape[1]

    # [계층형 인덱스 전략] 벤치마크 결과 기반 최적화
    # 1. 소규모 (5,000개 미만): 정확도 100% 보장 (Flat)
    if chunk_count < 5000:
        index_type = "Flat"
        ef_search = 0
    # 2. 중규모 (50,000개 미만): 고성능 검색 및 높은 정확도 (HNSW + No Quantization)
    elif chunk_count < 50000:
        index_type = "HNSW32,Flat"
        ef_search = 128
    # 3. 대규모: 메모리 효율 중시 (HNSW + Scalar Quantization)
    else:
        index_type = "HNSW32,SQ8"
        ef_search = 256

    logger.info(f"[FAISS] 인덱스 타입 결정: {index_type} (Chunks: {chunk_count})")
    index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)

    # ✅ [최적화] GPU 인덱스로 전환 (대규모 문서 시)
    if use_gpu and chunk_count > 5000:
        try:
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # 다중 GPU 샤딩
            co.usePrecomputed = False
            gpu_index = faiss.index_cpu_to_gpu_multiple(
                faiss.StandardGpuResources(), index, co
            )
            logger.info("[FAISS GPU] CPU 인덱스를 GPU로 복사 완료")
            index = gpu_index
        except Exception as e:
            logger.warning(f"[FAISS GPU] 전환 실패, CPU 사용: {e}")
            use_gpu = False

    if "SQ" in index_type or "IVF" in index_type:
        logger.info(f"[FAISS] 인덱스 훈련 중: {index_type}")
        index.train(vectors)

    index.add(vectors)

    # HNSW 파라미터 동적 설정
    if "HNSW" in index_type and not use_gpu:
        hnsw_index = faiss.downcast_index(index)
        hnsw_index.hnsw.efSearch = ef_search
        logger.debug(f"[FAISS] HNSW efSearch 설정: {ef_search}")

    doc_ids = [str(uuid.uuid4()) for _ in range(chunk_count)]
    new_docs = {
        doc_id: Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc_id, doc in zip(doc_ids, docs, strict=False)
    }
    docstore = InMemoryDocstore(new_docs)
    index_to_docstore_id = dict(enumerate(doc_ids))

    return FAISS(
        embedding_function=embedder,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )


def create_bm25_retriever(docs: list[Document]) -> Any:
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenizer)
    retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]
    return retriever


@log_operation("검색 컴포넌트 로드/생성")
async def load_and_build_retrieval_components(
    file_path: str,
    file_name: str,
    embedder: Embeddings,
    embedding_model_name: str,
    _on_progress=None,
    _file_hash: str | None = None,
    session_id: str | None = None,
) -> tuple[DocumentList, Any, Any, bool]:
    import torch

    file_bytes = None
    if _file_hash is None:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            _file_hash = compute_file_hash(file_path, data=file_bytes)
        except Exception as e:
            logger.error(f"파일 통합 로드 실패: {e}")

    cache = VectorStoreCache(file_path, embedding_model_name, file_hash=_file_hash)

    # [최적화] 전역 캐시 활성화 여부 확인
    from common.config import ENABLE_VECTOR_CACHE

    if ENABLE_VECTOR_CACHE:
        cache_data = cache.load(embedder)
        if all(x is not None for x in cache_data):
            doc_splits, vector_store, bm25_retriever = cache_data
            if (
                doc_splits is not None
                and vector_store is not None
                and bm25_retriever is not None
            ):
                logger.info(f"[Cache] 벡터 캐시 히트: {file_name}")
                return doc_splits, vector_store, bm25_retriever, True
    else:
        logger.info("[Cache] 벡터 캐시 비활성화됨 (config.yml)")

    # 캐시 미스 시 신규 생성 로직 시작
    docs = load_pdf_docs(
        file_path,
        file_name,
        on_progress=_on_progress,
        file_bytes=file_bytes,
        session_id=session_id,
    )

    if docs:
        sample_text = docs[0].page_content[:1000]
        has_korean = any("\uac00" <= char <= "\ud7a3" for char in sample_text)
        doc_lang = "Korean" if has_korean else "English"
        SessionManager.set("doc_language", doc_lang, session_id=session_id)
        logger.info(f"[RAG] [LANG] 문서 언어 감지됨: {doc_lang}")

    if not docs:
        raise EmptyPDFError(
            filename=file_name,
            details={"reason": "PDF에서 텍스트를 추출할 수 없습니다."},
        )

    if _on_progress:
        _on_progress()

    total_text_len = sum(len(d.page_content) for d in docs)
    is_small_doc = total_text_len < 2000

    doc_splits, precomputed_vectors = await split_documents(
        docs, embedder, session_id=session_id
    )
    if _on_progress:
        _on_progress()

    if not doc_splits:
        raise InsufficientChunksError(chunk_count=0, min_required=1)

    optimized_vectors: Any = precomputed_vectors
    if not is_small_doc:
        try:
            SessionManager.add_status_log("인덱스 최적화 중", session_id=session_id)
            optimizer = get_index_optimizer()
            doc_splits, optimized_vectors, q_meta, stats = optimizer.optimize_index(
                doc_splits, optimized_vectors
            )
            if optimizer and q_meta and q_meta.get("method") != "none":
                optimized_vectors = optimizer.quantizer.dequantize_vectors(
                    optimized_vectors, q_meta
                )
            SessionManager.replace_last_status_log(
                f"중복 내용 {stats.pruned_documents}개 정리", session_id=session_id
            )
        except Exception as e:
            logger.warning(f"인덱스 최적화 단계 건너뜀: {e}")

    vector_store = create_vector_store(doc_splits, embedder, vectors=optimized_vectors)
    bm25_retriever = create_bm25_retriever(doc_splits or [])

    if ENABLE_VECTOR_CACHE:
        cache.save(doc_splits, vector_store, bm25_retriever)

    if torch.cuda.is_available():
        import contextlib

        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    # [최적화] 메모리 즉시 회수 강제
    import gc

    gc.collect()

    SessionManager.add_status_log("신규 인덱싱 완료", session_id=session_id)
    return doc_splits, vector_store, bm25_retriever, False

    # [최종 방어] 어떤 경로로든 여기에 도달하면 현재 상태 반환
    return doc_splits, vector_store, bm25_retriever, False


@log_operation("RAG 파이프라인 구축")
async def build_rag_pipeline(
    uploaded_file_name: str,
    file_path: str,
    embedder: Embeddings,
    on_progress=None,
    session_id: str | None = None,
) -> tuple[str, bool]:
    file_hash = compute_file_hash(file_path)
    SessionManager.set("file_hash", file_hash, session_id=session_id)
    emb_model_name = getattr(
        embedder, "model", getattr(embedder, "model_name", "unknown_model")
    )

    doc_splits, vector_store, bm25_retriever, cache_used = (
        await load_and_build_retrieval_components(
            file_path,
            uploaded_file_name,
            embedder=embedder,
            embedding_model_name=emb_model_name,
            _on_progress=on_progress,
            _file_hash=file_hash,
            session_id=session_id,
        )
        or (None, None, None, False)  # ✅ Null 방지 폴백 추가
    )

    if not vector_store or not bm25_retriever:
        raise RuntimeError("검색 컴포넌트(FAISS/BM25) 생성에 실패했습니다.")

    if cache_used:
        SessionManager.add_status_log("캐시 데이터 로드", session_id=session_id)
        if on_progress:
            on_progress()

    faiss_retriever = vector_store.as_retriever(
        search_type=RETRIEVER_CONFIG["search_type"],
        search_kwargs=RETRIEVER_CONFIG["search_kwargs"],
    )

    SessionManager.set("faiss_retriever", faiss_retriever, session_id=session_id)
    SessionManager.set("bm25_retriever", bm25_retriever, session_id=session_id)

    rag_engine = build_graph()

    SessionManager.set("vector_store", vector_store, session_id=session_id)
    SessionManager.set("rag_engine", rag_engine, session_id=session_id)
    SessionManager.set("pdf_processed", True, session_id=session_id)
    SessionManager.add_status_log("질문 가능", session_id=session_id)

    if on_progress:
        on_progress()

    if cache_used:
        return f"'{uploaded_file_name}' 문서 캐시 데이터 로드 완료", True
    return (f"'{uploaded_file_name}' 신규 문서 인덱싱 완료"), False


def update_llm_in_pipeline(llm: T | None) -> None:
    if not SessionManager.get("pdf_processed"):
        raise ValueError("RAG 파이프라인이 구축되지 않아 LLM을 업데이트할 수 없습니다.")
    SessionManager.set("llm", llm)
    logger.info(f"세션 LLM 업데이트 완료: '{getattr(llm, 'model', 'unknown')}'")
