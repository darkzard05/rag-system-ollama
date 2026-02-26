"""
벡터 저장소 및 리트리버 컴포넌트 생성을 담당하는 팩토리 모듈.
"""

import logging
from typing import Any

import numpy as np
import torch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import RETRIEVER_CONFIG
from common.text_utils import bm25_tokenizer
from core.session import SessionManager

logger = logging.getLogger(__name__)


def create_vector_store(
    docs: list[Document],
    embedder: Embeddings,
    vectors: Any = None,
) -> Any:
    """FAISS 벡터 저장소를 생성합니다."""
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

    # GPU 자동 감지 및 설정
    use_gpu = False
    gpu_device = 0
    try:
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

    # L2 정규화
    if len(vectors) > 1000:
        faiss.normalize_L2(vectors)
        logger.debug("[FAISS] L2 정규화 완료")

    chunk_count = len(docs)
    d = vectors.shape[1]

    # 계층형 인덱스 전략
    if chunk_count < 5000:
        index_type = "Flat"
        ef_search = 0
    elif chunk_count < 50000:
        index_type = "HNSW32,Flat"
        ef_search = 128
    else:
        index_type = "HNSW32,SQ8"
        ef_search = 256

    logger.info(f"[FAISS] 인덱스 타입 결정: {index_type} (Chunks: {chunk_count})")
    SessionManager.add_status_log(f"고속 검색 인덱스 구축 중 ({index_type})")
    index = faiss.index_factory(d, index_type, faiss.METRIC_INNER_PRODUCT)

    # GPU 인덱스로 전환
    if use_gpu and chunk_count > 5000:
        try:
            from core.model_loader import ModelManager

            gpu_res = ModelManager.get_faiss_gpu_resources()
            if gpu_res:
                co = faiss.GpuMultipleClonerOptions()
                co.shard = True
                co.usePrecomputed = False

                gpu_index = faiss.index_cpu_to_gpu_multiple(gpu_res, index, co)
                logger.info(
                    "[FAISS GPU] CPU 인덱스를 GPU로 복사 완료 (Singleton Resources)"
                )
                index = gpu_index
            else:
                use_gpu = False
        except Exception as e:
            logger.warning(f"[FAISS GPU] 전환 실패, CPU 사용: {e}")
            use_gpu = False

    if "SQ" in index_type or "IVF" in index_type:
        logger.info(f"[FAISS] 인덱스 훈련 중: {index_type}")
        index.train(vectors)

    index.add(vectors)

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
    """BM25 리트리버를 생성합니다."""
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(docs, preprocess_func=bm25_tokenizer)
    retriever.k = RETRIEVER_CONFIG["search_kwargs"]["k"]
    return retriever
