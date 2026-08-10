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
# LRU eviction at 1024 entries; 768-dim float32 ≈ 3KB/entry → ~3.1MB at cap.
_doc_emb_cache: OrderedDict[str, list[float]] = OrderedDict()
_DOC_EMB_CACHE_MAX = 1024

# 쿼리 임베딩 캐시 상한 (고유 쿼리 수만큼 무한 누적 방지 — R3b-07)
_QUERY_EMB_CACHE_MAX = 512

# get_async_reranker가 마지막으로 활성화한 리랭커 엔진. rerank_score의 스케일
# (FlashRank sigmoid vs bi-encoder 코사인)을 결정하므로 grade short-circuit 임계값 분기에 사용된다 (R3b-02).
_rerank_engine_active: str = (
    "semantic" if RERANKER_ENGINE == "semantic" else "flashrank"
)


def get_active_rerank_engine() -> str:
    """현재 활성 리랭커 엔진을 반환합니다 ("semantic" | "flashrank")."""
    return _rerank_engine_active


def _set_active_engine(engine: str) -> None:
    global _rerank_engine_active  # noqa: PLW0603
    _rerank_engine_active = engine


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
        self.batch_size = max(1, batch_size)
        self._query_emb_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    async def rerank(
        self,
        documents: list[Document],
        query: str,
        top_k: int = 10,
    ) -> tuple[list[Document], list[float]]:
        """문서들을 쿼리와의 의미적 유사도로 재순위화합니다."""
        _set_active_engine("semantic")
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
        """쿼리 임베딩을 생성합니다 (LRU 캐시 활용)."""
        cached = self._query_emb_cache.get(query)
        if cached is not None:
            self._query_emb_cache.move_to_end(query)
            return cached

        vec = await asyncio.to_thread(self.embedder.embed_query, query)
        vec_np = np.array(vec, dtype="float32")
        self._query_emb_cache[query] = vec_np
        self._query_emb_cache.move_to_end(query)
        if len(self._query_emb_cache) > _QUERY_EMB_CACHE_MAX:
            self._query_emb_cache.popitem(last=False)
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
            # [R3b-06] batch_size 단위로 분할해 embed_documents 호출 — 후보 풀 확대 시
            # 단일 대형 배열 요청(Ollama /api/embed 한도) 회귀를 방지한다.
            embedded: list[list[float]] = []
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                embedded.extend(
                    await asyncio.to_thread(self.embedder.embed_documents, batch)
                )
            for idx, vec in zip(indices_needing_emb, embedded, strict=False):
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
        """FlashRank로 문서들을 재순위화합니다 (실패 시 bi-encoder 폴백).

        실패 분류 (R3b-03):
        - 의존성 부재(ImportError) → semantic 폴백 고정 (error 로그)
        - 로드/추론 실패(회로 차단 ResourceBuildError·네트워크·파일 포함) → semantic 폴백.
          재시도 정책은 resource_manager의 실패 네거티브 캐시(회로 차단기)가 관리한다.
        - 그 외(자체 점수 매핑 버그 등 프로그래밍 오류) → 은닉하지 않고 재발생.
        """
        if not documents:
            return [], []

        _set_active_engine("flashrank")

        # (1) 의존성 부재 → 영구 폴백
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
            # (2) 로드/추론 실패 → 폴백 (재시도 정책은 회로 차단기가 담당)
            results = await asyncio.to_thread(ranker.rerank, request)
        except (ImportError, ModuleNotFoundError) as e:
            logger.error(
                f"[RERANK] FlashRank 의존성 부재 ({e}) — semantic 폴백 고정",
                exc_info=True,
            )
            return await (await _get_semantic_fallback_reranker()).rerank(
                documents, query, top_k
            )
        except Exception as e:
            logger.warning(
                f"[RERANK] FlashRank 로드/추론 실패 ({type(e).__name__}: {e}) — "
                "semantic 폴백 (재시도는 네거티브 캐시 회로 차단기가 관리)",
                exc_info=True,
            )
            return await (await _get_semantic_fallback_reranker()).rerank(
                documents, query, top_k
            )

        # (3) 우리 모듈의 점수 매핑/정렬 — 프로그래밍 오류는 은닉하지 않고 재발생.
        try:
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
            logger.error(
                "[RERANK] FlashRank 결과 점수 매핑 중 오류 — 폴백하지 않고 재발생",
                exc_info=True,
            )
            raise


# FlashRank 실패 폴백용 semantic 리랭커 싱글턴 — 쿼리마다 새 인스턴스가 생성돼
# _query_emb_cache 웜업 이점이 사라지는 것을 방지한다 (R3b-03).
_semantic_fallback_reranker: AsyncSemanticReranker | None = None


async def _get_semantic_fallback_reranker() -> AsyncSemanticReranker:
    global _semantic_fallback_reranker  # noqa: PLW0603
    if _semantic_fallback_reranker is None:
        from core.model_loader import ModelManager

        embedder = await ModelManager.get_embedder()
        _semantic_fallback_reranker = AsyncSemanticReranker(embedder)
    return _semantic_fallback_reranker


# 전역 인스턴스 (지연 초기화)
_async_reranker: AsyncSemanticReranker | AsyncCrossEncoderReranker | None = None

# get_async_reranker 초기화 잠금 — 이벤트 루프 변경(테스트 등) 시 재생성한다 (R3b-04).
_reranker_init_lock: asyncio.Lock | None = None
_reranker_init_lock_loop: asyncio.AbstractEventLoop | None = None


def _get_reranker_init_lock() -> asyncio.Lock:
    global _reranker_init_lock, _reranker_init_lock_loop  # noqa: PLW0603
    loop = asyncio.get_running_loop()
    if _reranker_init_lock is None or _reranker_init_lock_loop is not loop:
        _reranker_init_lock = asyncio.Lock()
        _reranker_init_lock_loop = loop
    return _reranker_init_lock


async def get_async_reranker() -> AsyncSemanticReranker | AsyncCrossEncoderReranker:
    """RERANKER_ENGINE 설정에 따라 전역 리랭커 인스턴스를 반환합니다.

    - "semantic": bi-encoder (AsyncSemanticReranker, 기존 동작)
    - "auto"/"flashrank": FlashRank 크로스-인코더 (실패 시 semantic 폴백)

    await 지점에서의 태스크 전환에도 단일 인스턴스를 보장하기 위해
    double-checked locking(asyncio.Lock)으로 초기화를 직렬화한다 (R3b-04).
    """
    global _async_reranker  # noqa: PLW0603
    if _async_reranker is None:
        async with _get_reranker_init_lock():
            if _async_reranker is None:
                if RERANKER_ENGINE == "semantic":
                    from core.model_loader import ModelManager

                    embedder = await ModelManager.get_embedder()
                    _async_reranker = AsyncSemanticReranker(embedder)
                    _set_active_engine("semantic")
                    engine = "semantic"
                else:
                    _async_reranker = AsyncCrossEncoderReranker()
                    _set_active_engine("flashrank")
                    engine = "flashrank"
                logger.info(f"[RERANK] engine={engine}")
    return _async_reranker
