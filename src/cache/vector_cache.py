"""
벡터 저장소 캐싱을 담당하는 모듈.
"""

import json
import logging
import os
import shutil
import uuid
from typing import Any

import orjson
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import (
    RETRIEVER_CONFIG,
    SEMANTIC_CHUNKER_CONFIG,
    TEXT_SPLITTER_CONFIG,
    VECTOR_STORE_CACHE_DIR,
)
from common.text_utils import bm25_tokenizer
from common.utils import fast_hash
from security.cache_security import (
    CacheIntegrityError,
    CacheTrustError,
)

logger = logging.getLogger(__name__)


def _serialize_docs(docs: list[Document]) -> list[dict]:
    """Pydantic의 무거운 dict() 대신 직접 필요한 필드만 추출"""
    return [
        {"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs
    ]


def _deserialize_docs(doc_dicts: list[dict]) -> list[Document]:
    """dict 리스트를 Document 객체 리스트로 변환"""
    return [
        Document(page_content=d["page_content"], metadata=d["metadata"])
        for d in doc_dicts
    ]


def _build_cache_payloads(
    doc_splits: list[Document], bm25_docs: list[Document]
) -> dict[str, bytes]:
    """문서 캐시 payload를 한 번만 생성해 재사용합니다."""
    serialized_splits = _serialize_docs(doc_splits)
    serialized_bm25 = _serialize_docs(bm25_docs)
    return {
        "doc_splits_payload": orjson.dumps(serialized_splits),
        "bm25_payload": orjson.dumps(serialized_bm25),
    }


def _build_config_signature(embedding_model_name: str) -> str:
    """청킹/파싱 설정 기반의 안정적인 캐시 키 서명을 생성합니다.

    임베딩 모델명, 청크 크기/중첩, 분할 구분자, 의미론적 분할 사용 여부가
    바뀌면 서로 다른 캐시 디렉터리를 사용하도록 하여 오래된 설정으로 생성된
    청크가 재사용되지 않도록 보장합니다. (설정 변경 시 일회성 캐시 재구축)
    """
    splitter = TEXT_SPLITTER_CONFIG
    semantic = SEMANTIC_CHUNKER_CONFIG
    payload = orjson.dumps(
        {
            "embedding_model": str(embedding_model_name),
            "chunk_size": splitter.get("chunk_size"),
            "chunk_overlap": splitter.get("chunk_overlap"),
            "separators": list(splitter.get("separators") or []),
            "semantic_chunker_enabled": semantic.get("enabled", False),
        },
        option=orjson.OPT_SORT_KEYS,
    )
    return fast_hash(payload.decode())


class VectorStoreCache:
    """
    벡터 저장소와 관련 컴포넌트를 디스크에 캐싱하고 로드합니다.
    Pickle-free 로딩을 통해 보안성을 강화합니다.
    """

    def __init__(
        self,
        file_path: str,
        embedding_model_name: str,
        cache_dir: str = VECTOR_STORE_CACHE_DIR,
        file_hash: str | None = None,
    ):
        from core.document_processor import compute_file_hash

        self.file_hash = file_hash or compute_file_hash(file_path)
        cache_key = _build_config_signature(embedding_model_name)
        self.cache_dir = os.path.join(cache_dir, f"{self.file_hash}_{cache_key}")
        self.doc_splits_path = os.path.join(self.cache_dir, "doc_splits.json")
        self.faiss_index_path = os.path.join(self.cache_dir, "faiss_index")
        self.bm25_retriever_path = os.path.join(self.cache_dir, "bm25_docs.json")

        # Backward compatibility: legacy cache key using [:10] truncation
        legacy_key = embedding_model_name[:10]
        self._legacy_cache_dir: str | None = None
        if legacy_key != cache_key:
            self._legacy_cache_dir = os.path.join(
                cache_dir, f"{self.file_hash}_{legacy_key}"
            )

        from security.cache_security import get_security_manager

        self.security_manager = get_security_manager()
        self._cache_invalid = False

    def _get_cache_paths(self):
        cache_dir = self.cache_dir
        return (
            cache_dir,
            os.path.join(cache_dir, "doc_splits.json"),
            os.path.join(cache_dir, "faiss_index"),
            os.path.join(cache_dir, "bm25_docs.json"),
        )

    def _try_load_legacy(self) -> bool:
        """Check if legacy cache dir (pre-fast_hash) exists and re-point paths."""
        if not self._legacy_cache_dir:
            return False
        legacy_paths = (
            os.path.join(self._legacy_cache_dir, "doc_splits.json"),
            os.path.join(self._legacy_cache_dir, "faiss_index"),
            os.path.join(self._legacy_cache_dir, "bm25_docs.json"),
        )
        if not all(os.path.exists(p) for p in legacy_paths):
            return False
        self.cache_dir = self._legacy_cache_dir
        self.doc_splits_path = legacy_paths[0]
        self.faiss_index_path = legacy_paths[1]
        self.bm25_retriever_path = legacy_paths[2]
        return True

    def load(
        self,
        embedder: Embeddings,
        resource_manager: Any | None = None,
    ) -> tuple[list[Document] | None, Any | None, Any | None]:
        if not all(
            os.path.exists(p)
            for p in [
                self.doc_splits_path,
                self.faiss_index_path,
                self.bm25_retriever_path,
            ]
        ):
            # Backward compat: try legacy cache dir (pre-fast_hash)
            if self._legacy_cache_dir and self._try_load_legacy():
                logger.info(
                    f"[Cache] Loaded from legacy cache dir: {self._legacy_cache_dir}"
                )
            else:
                return None, None, None

        try:
            import faiss
            from langchain_community.docstore.in_memory import InMemoryDocstore
            from langchain_community.retrievers import BM25Retriever
            from langchain_community.vectorstores import FAISS
            from langchain_community.vectorstores.utils import DistanceStrategy

            paths_to_verify = [
                (self.doc_splits_path, "문서 데이터"),
                (self.faiss_index_path, "FAISS 인덱스"),
                (self.bm25_retriever_path, "BM25 리트리버"),
            ]

            for path, desc in paths_to_verify:
                try:
                    if not self.security_manager.verify_cache_trust(path):
                        raise CacheTrustError(f"불신 경로: {path}")
                    if os.path.isfile(path):
                        if os.path.exists(path + ".meta"):
                            self.security_manager.verify_cache_integrity(path)
                        else:
                            logger.debug(
                                f"[Cache] .meta 없음 - 무결성 검증 생략 (legacy): {path}"
                            )
                    elif os.path.isdir(path):
                        for f in os.listdir(path):
                            f_path = os.path.join(path, f)
                            if os.path.exists(f_path + ".meta"):
                                self.security_manager.verify_cache_integrity(f_path)
                            else:
                                logger.debug(
                                    f"[Cache] .meta 없음 - 무결성 검증 생략 (legacy): {f_path}"
                                )
                except (CacheTrustError, CacheIntegrityError) as e:
                    logger.critical(
                        f"[Security] 캐시 무결성 위반 감지 ({desc}: {type(e).__name__}). "
                        "캐시를 삭제하지 않고 재구축을 유도합니다."
                    )
                    self._cache_invalid = True
                    return None, None, None

            # 1. 문서 조각 로드
            with open(self.doc_splits_path, "rb") as file_handle:
                doc_dicts = orjson.loads(file_handle.read())
            doc_splits = _deserialize_docs(doc_dicts)

            # 2. FAISS 인덱스 수동 로드 (Pickle/index.pkl 무시)
            index_file = os.path.join(self.faiss_index_path, "index.faiss")
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")

            # FAISS 버전 호환성 검사
            version_path = os.path.join(self.faiss_index_path, "VERSION.json")
            if os.path.exists(version_path):
                try:
                    with open(version_path) as _f:
                        manifest = json.load(_f)
                    stored_faiss_version = manifest.get("faiss_version", "unknown")
                    if stored_faiss_version != faiss.__version__:
                        logger.warning(
                            f"[CACHE] FAISS version mismatch: cached={stored_faiss_version}, "
                            f"runtime={faiss.__version__}. 캐시를 삭제하지 않고 재구축합니다."
                        )
                        self._cache_invalid = True
                        return None, None, None
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.warning(
                        f"[CACHE] VERSION.json read failed ({e}). "
                        "캐시를 삭제하지 않고 재구축합니다."
                    )
                    self._cache_invalid = True
                    return None, None, None
            else:
                # Legacy cache without version info — assume compatible
                logger.warning(
                    "[CACHE] No VERSION.json found (legacy cache). Proceeding with compatibility mode."
                )

            index = faiss.read_index(index_file)

            import uuid

            doc_ids = [str(uuid.uuid4()) for _ in range(len(doc_splits))]
            new_docstore_docs = dict(zip(doc_ids, doc_splits, strict=False))
            docstore = InMemoryDocstore(new_docstore_docs)
            index_to_docstore_id = dict(enumerate(doc_ids))

            vector_store = FAISS(
                embedding_function=embedder,
                index=index,
                docstore=docstore,
                index_to_docstore_id=index_to_docstore_id,
                distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
            )

            # 3. BM25 로드
            with open(self.bm25_retriever_path, "rb") as file_handle:
                bm25_doc_dicts = orjson.loads(file_handle.read())
            bm25_docs = _deserialize_docs(bm25_doc_dicts)

            bm25_retriever = BM25Retriever.from_documents(
                bm25_docs, preprocess_func=bm25_tokenizer
            )
            bm25_retriever.k = RETRIEVER_CONFIG.get("search_kwargs", {}).get("k", 5)

            logger.info(f"RAG 캐시 안전 로드 완료 (Pickle-free): '{self.cache_dir}'")
            return doc_splits, vector_store, bm25_retriever

        except Exception as e:
            logger.warning(
                f"캐시 로드 중 예외 발생: {e}. 캐시를 삭제하지 않고 재구축을 유도합니다."
            )
            self._cache_invalid = True
            return None, None, None

    def save(
        self,
        doc_splits: list[Document],
        vector_store: Any,
        bm25_retriever: Any,
    ) -> None:
        if os.path.exists(self.cache_dir) and not self._cache_invalid:
            logger.info(f"[Cache] 캐시가 이미 존재함: {self.cache_dir}")
            return

        staging_dir = f"{self.cache_dir}.tmp.{uuid.uuid4().hex[:8]}"
        stg_doc_splits_path = os.path.join(staging_dir, "doc_splits.json")
        stg_faiss_index_path = os.path.join(staging_dir, "faiss_index")
        stg_bm25_retriever_path = os.path.join(staging_dir, "bm25_docs.json")

        try:
            os.makedirs(staging_dir, exist_ok=True)
            self.security_manager.enforce_directory_permissions(staging_dir)

            bm25_docs = getattr(bm25_retriever, "docs", None)
            if bm25_docs is None:
                bm25_docs = doc_splits

            payloads = _build_cache_payloads(doc_splits, bm25_docs)
            with open(stg_doc_splits_path, "wb") as f:
                f.write(payloads["doc_splits_payload"])
            self.security_manager.enforce_file_permissions(stg_doc_splits_path)

            doc_meta = self.security_manager.create_metadata_for_file(
                stg_doc_splits_path, description="Document splits cache (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_doc_splits_path + ".meta", doc_meta
            )

            vector_store.save_local(stg_faiss_index_path)
            self.security_manager.enforce_directory_permissions(stg_faiss_index_path)

            # FAISS 버전 메타데이터 저장 (버전 불일치 시 자동 캐시 재생성)
            import sys as _sys
            import time as _time

            import faiss as _faiss

            version_manifest = {
                "faiss_version": _faiss.__version__,
                "schema_version": 2,
                "python_version": _sys.version,
                "created_at": _time.time(),
            }
            version_path = os.path.join(stg_faiss_index_path, "VERSION.json")
            with open(version_path, "w") as _f:
                json.dump(version_manifest, _f, indent=2)

            # load()는 .meta가 존재하는 파일만 무결성 검증하므로 모든 아티팩트에 생성.
            for artifact in os.listdir(stg_faiss_index_path):
                artifact_path = os.path.join(stg_faiss_index_path, artifact)
                if os.path.isfile(artifact_path):
                    meta = self.security_manager.create_metadata_for_file(
                        artifact_path,
                        description=f"FAISS cache artifact ({artifact})",
                    )
                    self.security_manager.save_cache_metadata(
                        artifact_path + ".meta", meta
                    )

            # BM25 저장
            with open(stg_bm25_retriever_path, "wb") as f:
                f.write(payloads["bm25_payload"])
            self.security_manager.enforce_file_permissions(stg_bm25_retriever_path)
            bm25_meta = self.security_manager.create_metadata_for_file(
                stg_bm25_retriever_path, description="BM25 retriever cache (JSON)"
            )
            self.security_manager.save_cache_metadata(
                stg_bm25_retriever_path + ".meta", bm25_meta
            )

            # 스테이징 디렉토리를 최종 위치로 이동
            if self._cache_invalid and os.path.exists(self.cache_dir):
                # 비파괴 재구축: 무결성 위반으로 판정된 캐시만 교체 대상.
                shutil.rmtree(self.cache_dir, ignore_errors=True)
            try:
                os.rename(staging_dir, self.cache_dir)
            except FileExistsError:
                # 다른 프로세스가 먼저 저장한 경우: 스테이징만 정리하고 종료.
                logger.info(
                    "[Cache] 동시 저장 감지 (다른 프로세스가 먼저 저장). 스테이징 제거."
                )
                if os.path.exists(staging_dir):
                    shutil.rmtree(staging_dir)
                return
            self._cache_invalid = False
            logger.info(f"[Cache] 벡터 캐시 저장 완료: {self.cache_dir}")

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
