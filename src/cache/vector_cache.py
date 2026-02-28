"""
벡터 저장소 캐싱을 담당하는 모듈.
"""

import logging
import os
import shutil
import uuid
from typing import Any

import orjson
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import RETRIEVER_CONFIG, VECTOR_STORE_CACHE_DIR
from common.text_utils import bm25_tokenizer
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
        self.cache_dir = os.path.join(
            cache_dir, f"{self.file_hash}_{embedding_model_name[:10]}"
        )
        self.doc_splits_path = os.path.join(self.cache_dir, "doc_splits.json")
        self.faiss_index_path = os.path.join(self.cache_dir, "faiss_index")
        self.bm25_retriever_path = os.path.join(self.cache_dir, "bm25_docs.json")
        from security.cache_security import get_security_manager

        self.security_manager = get_security_manager()

    def _get_cache_paths(self):
        cache_dir = self.cache_dir
        return (
            cache_dir,
            os.path.join(cache_dir, "doc_splits.json"),
            os.path.join(cache_dir, "faiss_index"),
            os.path.join(cache_dir, "bm25_docs.json"),
        )

    def _purge_cache(self, reason: str):
        if os.path.exists(self.cache_dir):
            try:
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
                    self.security_manager.verify_cache_trust(path)
                    if os.path.isfile(path):
                        self.security_manager.verify_cache_integrity(path)
                    elif os.path.isdir(path):
                        for f in os.listdir(path):
                            f_path = os.path.join(path, f)
                            self.security_manager.verify_cache_integrity(f_path)
                except (CacheTrustError, CacheIntegrityError) as e:
                    self._purge_cache(
                        reason=f"Security Violation in {desc}: {type(e).__name__}"
                    )
                    return None, None, None

            # 1. 문서 조각 로드
            with open(self.doc_splits_path, "rb") as file_handle:
                doc_dicts = orjson.loads(file_handle.read())
            doc_splits = _deserialize_docs(doc_dicts)

            # 2. FAISS 인덱스 수동 로드 (Pickle/index.pkl 무시)
            index_file = os.path.join(self.faiss_index_path, "index.faiss")
            if not os.path.exists(index_file):
                raise FileNotFoundError(f"FAISS index file not found: {index_file}")

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
            logger.warning(f"캐시 로드 중 예외 발생: {e}. 캐시를 폐기합니다.")
            self._purge_cache(reason=f"Load Error: {str(e)}")
            return None, None, None

    def save(
        self,
        doc_splits: list[Document],
        vector_store: Any,
        bm25_retriever: Any,
    ) -> None:
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

            # BM25 저장
            bm25_docs = bm25_retriever.docs
            serialized_bm25 = _serialize_docs(bm25_docs)
            with open(stg_bm25_retriever_path, "wb") as f:
                f.write(orjson.dumps(serialized_bm25))
            self.security_manager.enforce_file_permissions(stg_bm25_retriever_path)

            # 스테이징 디렉토리를 최종 위치로 이동
            os.rename(staging_dir, self.cache_dir)
            logger.info(f"[Cache] 벡터 캐시 저장 완료: {self.cache_dir}")

        except Exception as e:
            logger.error(f"캐시 저장 실패: {e}")
            if os.path.exists(staging_dir):
                shutil.rmtree(staging_dir)
