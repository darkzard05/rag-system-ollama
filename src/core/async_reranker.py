"""
비동기 네이티브 리랭커: 스레드 풀 없이 async/await로 직접 처리.
내부적으로 배치 임베딩 + 코사인 유사도 벡터 연산만 수행하므로 GIL 해제 불필요.
"""

import asyncio
import hashlib
import logging
from collections import OrderedDict

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.config import RERANKER_ENGINE

logger = logging.getLogger(__name__)

# Module-level document embedding cache: text_hash → embedding_vector list
# LRU eviction at 1024 entries; one chunk is ~1000 tokens → ~250KB for 1024 entries
_doc_emb_cache: OrderedDict[str, list[float]] = OrderedDict()
_DOC_EMB_CACHE_MAX = 1024


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_cached_emb(text: str) -> list[float] | None:
    h = _text_hash(text)
    if h in _doc_emb_cache:
        _doc_emb_cache.move_to_end(h)  # LRU refresh
        return _doc_emb_cache[h]
    return None


def _set_cached_emb(text: str, vec: list[float]) -> None:
    h = _text_hash(text)
    _doc_emb_cache[h] = vec
    _doc_emb_cache.move_to_end(h)
    if len(_doc_emb_cache) > _DOC_EMB_CACHE_MAX:
        _doc_emb_cache.popitem(last=False)


class AsyncSemanticReranker:
    """비동기 임베딩 기반 시맨틱 리랭커 (스레드 풀 미사용)."""

    def __init__(self, embedder: Embeddings, batch_size: int = 32):
        self.embedder = embedder
        self.batch_size = batch_size
        self._query_emb_cache: dict[str, np.ndarray] = {}

    async def rerank(
        self,
        documents: list[Document],
        query: str,
        top_k: int = 10,
    ) -> tuple[list[Document], list[float]]:
        """문서들을 쿼리와의 의미적 유사도로 재순위화합니다."""
        if not documents:
            return [], []

        query_vec = await self._get_query_embedding(query)
        doc_vecs = await self._get_doc_embeddings(documents)
        similarities = self._cosine_similarity_batch(query_vec, doc_vecs)

        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        ranked_docs = [documents[i] for i in ranked_indices]
        ranked_scores = [float(similarities[i]) for i in ranked_indices]

        for doc, score in zip(ranked_docs, ranked_scores, strict=False):
            doc.metadata["rerank_score"] = score

        return ranked_docs, ranked_scores

    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """쿼리 임베딩을 생성합니다 (캐시 활용)."""
        if query in self._query_emb_cache:
            return self._query_emb_cache[query]

        vec = await asyncio.to_thread(self.embedder.embed_query, query)
        vec_np = np.array(vec, dtype="float32")
        self._query_emb_cache[query] = vec_np
        return vec_np

    async def _get_doc_embeddings(self, documents: list[Document]) -> np.ndarray:
        """문서 임베딩 배치를 생성합니다 (메타데이터 캐시 재사용)."""
        texts: list[str] = []
        indices_needing_emb: list[int] = []
        vecs: list[np.ndarray | None] = [None] * len(documents)

        for i, doc in enumerate(documents):
            # 1. Check module-level persistent cache first (survives across queries)
            cached_vec = _get_cached_emb(doc.page_content)
            if cached_vec is not None:
                vecs[i] = np.array(cached_vec, dtype="float32")
                # Also store on metadata for faster access within this batch
                doc.metadata["embedding_vector"] = cached_vec
                continue
            # 2. Check metadata-level cache (within current query)
            cached_vec = doc.metadata.get("embedding_vector")
            if cached_vec is not None:
                vecs[i] = np.array(cached_vec, dtype="float32")
                continue
            # 3. Need to compute
            texts.append(doc.page_content)
            indices_needing_emb.append(i)

        if texts:
            batch_vecs = await asyncio.to_thread(self.embedder.embed_documents, texts)
            for idx, vec in zip(indices_needing_emb, batch_vecs, strict=False):
                vec_np = np.array(vec, dtype="float32")
                vecs[idx] = vec_np
                vec_list = vec_np.tolist()
                documents[idx].metadata["embedding_vector"] = vec_list
                _set_cached_emb(
                    documents[idx].page_content, vec_list
                )  # Module-level cache

        # 모든 벡터가 None이 아닌지 확인
        result_vecs = [v if v is not None else np.zeros_like(vecs[0]) for v in vecs]
        return np.stack(result_vecs)

    @staticmethod
    def _cosine_similarity_batch(
        query_vec: np.ndarray, doc_vecs: np.ndarray
    ) -> np.ndarray:
        """벡터화된 코사인 유사도 계산 (NumPy einsum 활용)."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-9)
        return np.dot(doc_norms, query_norm)


class AsyncCrossEncoderReranker:
    """FlashRank 크로스-인코더 리랭커 (ONNX, CPU).

    쿼리-문서 쌍을 직접 스코어링해 bi-encoder 임베딩 재정렬보다 정확도가 높습니다.
    FlashRank 로드/추론 중 예외 발생 시 AsyncSemanticReranker로 폴백합니다.
    """

    async def rerank(
        self,
        documents: list[Document],
        query: str,
        top_k: int = 10,
    ) -> tuple[list[Document], list[float]]:
        """FlashRank로 문서들을 재순위화합니다 (실패 시 bi-encoder 폴백)."""
        if not documents:
            return [], []

        try:
            from flashrank import RerankRequest

            from core.model_loader import ModelManager

            ranker = await ModelManager.get_flashranker()
            request = RerankRequest(
                query=query,
                passages=[
                    {"id": i, "text": doc.page_content}
                    for i, doc in enumerate(documents)
                ],
            )
            results = await asyncio.to_thread(ranker.rerank, request)
            ranked = sorted(
                ((documents[p["id"]], float(p["score"])) for p in results),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            ranked_docs = [doc for doc, _ in ranked]
            ranked_scores = [score for _, score in ranked]
            for doc, score in zip(ranked_docs, ranked_scores, strict=False):
                doc.metadata["rerank_score"] = score
            return ranked_docs, ranked_scores
        except Exception:
            logger.warning(
                "[RERANK] FlashRank 리랭킹 실패 — AsyncSemanticReranker로 폴백",
                exc_info=True,
            )
            from core.model_loader import ModelManager

            embedder = await ModelManager.get_embedder()
            semantic_reranker = AsyncSemanticReranker(embedder)
            return await semantic_reranker.rerank(documents, query, top_k)


# 전역 인스턴스 (지연 초기화)
_async_reranker: AsyncSemanticReranker | AsyncCrossEncoderReranker | None = None


async def get_async_reranker() -> AsyncSemanticReranker | AsyncCrossEncoderReranker:
    """RERANKER_ENGINE 설정에 따라 전역 리랭커 인스턴스를 반환합니다.

    - "semantic": bi-encoder (AsyncSemanticReranker, 기존 동작)
    - "auto"/"flashrank": FlashRank 크로스-인코더 (실패 시 semantic 폴백)
    """
    global _async_reranker  # noqa: PLW0603
    if _async_reranker is None:
        if RERANKER_ENGINE == "semantic":
            from core.model_loader import ModelManager

            embedder = await ModelManager.get_embedder()
            _async_reranker = AsyncSemanticReranker(embedder)
            engine = "semantic"
        else:
            _async_reranker = AsyncCrossEncoderReranker()
            engine = "flashrank"
        logger.info(f"[RERANK] engine={engine}")
    return _async_reranker
