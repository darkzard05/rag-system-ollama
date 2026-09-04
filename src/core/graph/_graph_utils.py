"""Graph 유틸리티 함수 모듈.

graph_builder, grade_memo 등 여러 모듈에서 공유하는 헬퍼 함수를 모은다.
순환 의존성 방지를 위해 이 모듈은 graph_builder를 절대 import하지 않는다.
"""

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    QUERY_CACHE_ENABLED,
)
from common.utils import doc_stable_id
from core.model_loader import ModelManager
from services.optimization.caching_optimizer import get_cache_manager

logger = logging.getLogger(__name__)


def get_state_attr(state: Any, key: str, default: Any = None) -> Any:
    """dict와 object(GraphState) 모두에서 속성을 안전하게 가져옵니다."""
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


class _AsyncEmbeddingWrapper:
    """동기형 embed_query를 비동기 인터페이스로 감싸는 어댑터.

    SemanticCache._embed()는 self.embedding_model.embed_query(text)를
    무조건 await한다. Ollama 임베더(_NoTruncateOllamaEmbeddings)의 embed_query는
    동기형(list 반환)이라 await 시 crash한다. 캐시 소스는 건드리지 않고, 쿼리
    경로에서 캐시에 주입하는 임베더만 비동기 퍼사드로 감싸 호환시킨다.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    async def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self._embedder, "embed_documents"):
            return self._embedder.embed_documents(texts)
        return [self._embedder.embed_query(t) for t in texts]


async def _ensure_query_cache_embedder() -> None:
    """전역 캐시 싱글톤의 세맨틱 캐시에 임베더를 주입한다.

    SemanticCache는 self.embedding_model이 None이면 get/set 시 항상 None을
    반환한다(임베딩 불가). 청커는 별도 CacheManager 인스턴스를 만들므로 전역
    싱글톤의 semantic_cache.embedding_model은 기본 비어 있다 — 쿼리 경로에서
    캐시를 쓰려면 런타임에 주입해야 한다. (캐시 클래스는 수정하지 않음)
    동기형 Ollama 임베더는 _AsyncEmbeddingWrapper로 감싸 await 호환을 맞춘다.
    """
    if not QUERY_CACHE_ENABLED:
        return
    cm = get_cache_manager()
    if cm.semantic_cache is None:
        return
    if cm.semantic_cache.embedding_model is not None:
        return
    try:
        embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)
        cm.semantic_cache.embedding_model = _AsyncEmbeddingWrapper(embedder)
    except Exception as e:  # noqa: BLE001 - 캐시 누락은 치명적이지 않음
        logger.warning(f"[RAG] [CACHE] 임베더 주입 실패 — 쿼리 캐시 비활성화: {e}")


def _sanitize_channel_value(value: Any) -> Any:
    """상태 채널 값을 msgpack 직렬화 가능한 순수 타입으로 위생화합니다.

    R1a-05: JsonPlusSerializer(pickle_fallback=False) 전환에 따라 상태에 저장되는
    값은 int/str/float/bool/None/list/dict만 허용한다. 그 외 객체(커스텀 클래스
    인스턴스 등)를 만나면 조용히 pickle로 강등하지 않고 명시적 예외를 던진다.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_sanitize_channel_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_channel_value(item) for item in value)
    if isinstance(value, dict):
        return {k: _sanitize_channel_value(v) for k, v in value.items()}
    raise ValueError(
        f"직렬화 불가 객체 감지 (pickle 강등 금지): {type(value).__name__} — "
        "채널에는 순수 타입만 저장할 수 있습니다."
    )


async def _safe_invoke(
    llm: Any, prompt: str, config: dict, model_name: str = DEFAULT_OLLAMA_MODEL
) -> Any:
    """Safely invoke an LLM with async fallback, pinning it for the call duration.

    Pins the model via the coordinator so it cannot be evicted mid-inference
    (use-after-free defect) while the underlying ainvoke is in flight.
    """
    from core.resource_manager import get_resource_manager

    coordinator = get_resource_manager()
    async with coordinator.use_llm(model_name=model_name):
        res = llm.ainvoke(prompt, config=config)
        if asyncio.iscoroutine(res):
            return await res
        return res


def _doc_stable_id(doc: Document) -> str:
    """문서의 안정(stable) 식별자를 반환합니다.

    `format_context`, `_estimate_ctx_tokens`, verify 노드 모두 동일한 식별자를
    사용해야 하므로 단일 진원(single source of truth)으로 추출합니다.
    `doc_id` 메타데이터가 있으면 그것을, 없으면 page_content 해시를 사용합니다.
    공용 구현은 `common.utils.doc_stable_id` 를 참조합니다 (R: 중복 통합).
    """
    return doc_stable_id(doc)
