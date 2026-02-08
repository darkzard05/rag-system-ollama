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
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

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


def split_documents(
    docs: list[Document],
    embedder: Embeddings | None = None,
    session_id: str | None = None,
) -> tuple[list[Document], list[np.ndarray] | None]:
    """
    설정에 따라 의미론적 분할기 또는 RecursiveCharacterTextSplitter를 사용해 문서를 분할합니다.
    """
    SessionManager.add_status_log("문장 분할 중...", session_id=session_id)
    with monitor.track_operation(
        OperationType.SEMANTIC_CHUNKING, {"doc_count": len(docs)}
    ) as op:
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
                min_chunk_size=int(SEMANTIC_CHUNKER_CONFIG.get("min_chunk_size", 100)),
                max_chunk_size=int(SEMANTIC_CHUNKER_CONFIG.get("max_chunk_size", 800)),
                similarity_threshold=float(
                    SEMANTIC_CHUNKER_CONFIG.get("similarity_threshold", 0.5)
                ),
                batch_size=batch_size,
            )

            split_docs, vectors = semantic_chunker.split_documents(docs)
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
        for i, doc in enumerate(split_docs):
            doc.metadata = doc.metadata.copy()
            doc.metadata["chunk_index"] = i

            content_lower = doc.page_content.lower()
            is_noise = any(kw in content_lower[:100] for kw in noise_keywords)
            if not is_noise and (
                content_lower.count("doi:") > 2 or content_lower.count(",") > 25
            ):
                is_noise = True

            doc.metadata["is_content"] = not is_noise

        op.tokens = sum(len(doc.page_content.split()) for doc in split_docs)
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
        # [수정] 리스트 형태인 경우 vstack으로 2D 배열 변환 보장
        if isinstance(vectors, list):
            vectors = np.vstack(vectors).astype("float32")
        else:
            vectors = np.ascontiguousarray(vectors, dtype="float32")

    faiss.normalize_L2(vectors)
    chunk_count = len(docs)
    d = vectors.shape[1]

    index_type = "Flat" if chunk_count < 1000 else "HNSW32,SQ8"
    index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)

    if "SQ" in index_type or "IVF" in index_type:
        index.train(vectors)

    index.add(vectors)

    if "HNSW" in index_type:
        faiss.downcast_index(index).hnsw.efSearch = 128

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
def load_and_build_retrieval_components(
    file_path: str,
    file_name: str,
    embedder: Embeddings,
    embedding_model_name: str,
    _on_progress=None,
    _file_hash: str | None = None,
    session_id: str | None = None,
) -> tuple[DocumentList, Any, Any, bool]:
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

    doc_splits, vector_store, bm25_retriever = (None, None, None)
    if ENABLE_VECTOR_CACHE:
        doc_splits, vector_store, bm25_retriever = cache.load(embedder)
    else:
        logger.info("[Cache] 벡터 캐시 비활성화됨 (config.yml)")

    cache_used = all(x is not None for x in [doc_splits, vector_store, bm25_retriever])

    if not cache_used:
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

        doc_splits, precomputed_vectors = split_documents(
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

        vector_store = create_vector_store(
            doc_splits, embedder, vectors=optimized_vectors
        )
        bm25_retriever = create_bm25_retriever(doc_splits or [])

        if ENABLE_VECTOR_CACHE:
            cache.save(doc_splits, vector_store, bm25_retriever)
        else:
            logger.debug("[Cache] 벡터 캐시 저장 스킵 (비활성화)")

            if torch.cuda.is_available():
                import contextlib

                with contextlib.suppress(Exception):
                    torch.cuda.empty_cache()
            SessionManager.add_status_log("신규 인덱싱 완료", session_id=session_id)
            return doc_splits, vector_store, bm25_retriever, cache_used


@log_operation("RAG 파이프라인 구축")
def build_rag_pipeline(
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
        load_and_build_retrieval_components(
            file_path,
            uploaded_file_name,
            embedder=embedder,
            embedding_model_name=emb_model_name,
            _on_progress=on_progress,
            _file_hash=file_hash,
            session_id=session_id,
        )
    )

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
