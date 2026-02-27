"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

import contextlib

from common.config import (
    CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    MAX_CACHED_MODELS,
    MAX_CONCURRENT_INFERENCE,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TOP_P,
)
from common.exceptions import EmbeddingModelError
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()


class ModelManager:
    """
    시스템 전체의 모델 인스턴스를 관리하는 중앙 클래스 (LRU 캐시 적용).
    UI와 API가 공동으로 사용하여 중복 로딩 및 VRAM 낭비를 방지합니다.
    [Modernized] asyncio 네이티브 동기화 도구 사용으로 전환됨.
    """

    # 비동기 락 (지연 로딩)
    _locks: dict[str, asyncio.Lock] = {}
    _inference_semaphore: asyncio.Semaphore | None = None

    # [수정] LRU 캐시로 변경
    _instances: OrderedDict[str, Any] = OrderedDict()
    MAX_CACHED_MODELS = MAX_CACHED_MODELS

    _sync_client = None
    _async_client = None
    _client_loop = None
    _faiss_gpu_resources = None

    @classmethod
    def _get_lock(cls, name: str) -> asyncio.Lock:
        """이름에 해당하는 비동기 락을 반환합니다. (지연 로딩)"""
        if name not in cls._locks:
            cls._locks[name] = asyncio.Lock()
        return cls._locks[name]

    @classmethod
    def _get_semaphore(cls) -> asyncio.Semaphore:
        """전역 추론 세마포어를 반환합니다. (지연 로딩)"""
        if cls._inference_semaphore is None:
            cls._inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCE)
        return cls._inference_semaphore

    @classmethod
    def get_faiss_gpu_resources(cls):
        """FAISS GPU 리소스를 싱글톤으로 반환합니다."""
        if cls._faiss_gpu_resources is None:
            # 이 부분은 FAISS 내부 로직이므로 동기 락 없이 초기화
            import faiss

            try:
                cls._faiss_gpu_resources = faiss.StandardGpuResources()
                logger.info("[ModelManager] FAISS GPU 리소스 초기화 완료")
            except Exception as e:
                logger.warning(f"[ModelManager] FAISS GPU 리소스 생성 실패: {e}")
        return cls._faiss_gpu_resources

    @classmethod
    async def acquire_inference_lock(cls):
        """비동기 세마포어를 획득합니다."""
        await cls._get_semaphore().acquire()

    @classmethod
    def release_inference_lock(cls):
        """세마포어를 해제합니다."""
        cls._get_semaphore().release()

    @classmethod
    def _get_from_cache(cls, key: str) -> Any | None:
        """LRU 캐시에서 인스턴스를 가져오고 순서를 갱신합니다."""
        if key in cls._instances:
            cls._instances.move_to_end(key)
            return cls._instances[key]
        return None

    @classmethod
    async def _add_to_cache(cls, key: str, instance: Any):
        """LRU 캐시에 인스턴스를 추가합니다. 필요 시 가장 오래된 것을 방출합니다."""
        if key in cls._instances:
            cls._instances.move_to_end(key)
            cls._instances[key] = instance
        else:
            if len(cls._instances) >= cls.MAX_CACHED_MODELS:
                await cls._evict_oldest_model()
            cls._instances[key] = instance
            cls._instances.move_to_end(key)

    @classmethod
    async def _evict_oldest_model(cls):
        """가장 오래된 모델을 방출하고 메모리를 정리합니다."""
        if not cls._instances:
            return
        key, instance = cls._instances.popitem(last=False)
        logger.info(f"[ModelManager] 가장 오래된 모델 방출: {key}")
        del instance
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("[ModelManager] GPU 캐시 비우기 완료 (torch.cuda.empty_cache)")

    @classmethod
    async def get_flashranker(cls, model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Any:
        """FlashRank 리랭커 모델을 가져오거나 로드합니다 (고속 CPU 리랭킹)"""
        async with cls._get_lock("flashrank"):
            cache_key = f"flashrank_{model_name}"
            instance = cls._get_from_cache(cache_key)
            if instance:
                return instance

            from flashrank import Ranker

            logger.info(f"[MODEL] [LOAD] FlashRank 리랭커 로드 중: {model_name}")
            instance = Ranker(model_name=model_name, cache_dir=CACHE_DIR)
            await cls._add_to_cache(cache_key, instance)
            return instance

    @classmethod
    def get_inference_semaphore(cls) -> Any:
        """[DEPRECATED] 하위 호환성을 위해 유지하되 사용을 지양합니다."""
        return cls._get_semaphore()

    @classmethod
    def get_client(cls, host: str):
        """캐싱된 동기 Ollama 클라이언트를 가져옵니다."""
        import ollama

        # 동기 클라이언트는 락 없이 체크 (가벼움)
        needs_new = (
            cls._sync_client is None
            or str(getattr(cls._sync_client, "base_url", "")) != host
        )
        if needs_new:
            cls._sync_client = ollama.Client(host=host)
            logger.info(f"[ModelManager] 새 동기 클라이언트 생성 (Host: {host})")
        return cls._sync_client

    @classmethod
    async def get_async_client(cls, host: str):
        """현재 이벤트 루프에 맞는 비동기 클라이언트를 가져옵니다."""
        from urllib.parse import urlparse

        import ollama

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            return ollama.AsyncClient(host=host)

        async with cls._get_lock("client"):
            # 호스트 정규화 (비교용)
            def normalize_url(url):
                if not url:
                    return ""
                p = urlparse(str(url))
                return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")

            target_host = normalize_url(host)
            current_host = normalize_url(getattr(cls._async_client, "base_url", ""))

            needs_new = (
                cls._async_client is None
                or cls._client_loop != current_loop
                or current_host != target_host
            )

            if needs_new:
                if cls._async_client:
                    with contextlib.suppress(Exception):
                        await cls._async_client.close()
                cls._async_client = ollama.AsyncClient(host=host)
                cls._client_loop = current_loop
                logger.info(
                    f"[ModelManager] 새 비동기 클라이언트 생성 (Host: {host}, Loop: {id(current_loop)})"
                )

            return cls._async_client

    @classmethod
    async def get_embedder(cls, model_name: str | None = None) -> Embeddings:
        """임베딩 모델을 가져오거나 로드합니다 (Thread-safe, LRU 캐시 적용)"""
        name = model_name or DEFAULT_EMBEDDING_MODEL
        async with cls._get_lock("embedder"):
            instance = cls._get_from_cache(name)
            if instance:
                return instance

            instance = load_embedding_model(name)
            await cls._add_to_cache(name, instance)
            return instance

    @classmethod
    async def get_llm(cls, model_name: str, **kwargs) -> Any:
        """LLM 클라이언트 인스턴스를 가져오거나 생성합니다 (Single-instance per model, LRU 캐시 적용)."""
        async with cls._get_lock("llm"):
            cache_key = f"llm_{model_name}"
            base_llm = cls._get_from_cache(cache_key)

            if not base_llm:
                base_llm = load_llm(model_name)
                await cls._add_to_cache(cache_key, base_llm)

            if not kwargs:
                return base_llm

            return base_llm.bind(**kwargs)

    @classmethod
    async def clear_vram(cls):
        """[위험] 모든 모델 인스턴스를 제거하여 VRAM을 강제로 비웁니다."""
        async with (
            cls._get_lock("llm"),
            cls._get_lock("embedder"),
            cls._get_lock("flashrank"),
        ):
            cls._instances.clear()
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[System] [VRAM] 모든 모델 인스턴스 해제 및 캐시 클리어")


def _fetch_available_models_cached() -> list[str]:
    """Ollama 모델 목록을 가져옵니다. (UI 종속성 제거)"""
    try:
        import ollama

        ollama_response = ollama.list()
        models = []
        if hasattr(ollama_response, "models"):
            for model in ollama_response.models:
                name = getattr(model, "model", None) or (
                    model.get("model") if isinstance(model, dict) else None
                )
                if name:
                    models.append(name)
        elif isinstance(ollama_response, dict) and "models" in ollama_response:
            for model in ollama_response["models"]:
                name = model.get("model") or model.get("name")
                if name:
                    models.append(name)
        models.sort()
        return models
    except Exception as e:
        logger.warning(f"Ollama 모델 목록 조회 실패: {e}")
        return []


def load_embedding_model(
    embedding_model_name: str | None = None,
) -> Any:
    """
    임베딩 모델을 로드합니다. (HuggingFace 및 Ollama 지원)
    HuggingFace 모델은 VRAM 보호를 위해 기본적으로 CPU에서 작동하도록 설정합니다.
    """
    model_key = embedding_model_name or DEFAULT_EMBEDDING_MODEL

    # [최적화] Ollama 임베딩 여부 판별
    is_ollama_embedding = "/" not in model_key or model_key.startswith("ollama:")
    clean_model_name = (
        model_key.replace("ollama:", "") if "ollama:" in model_key else model_key
    )

    # [최적화] CI/유닛 테스트 환경에서는 실제 모델 로드 없이 가짜 임베딩 모델 반환
    if os.getenv("IS_CI_TEST") == "true" or os.getenv("IS_UNIT_TEST") == "true":
        from langchain_core.embeddings import FakeEmbeddings

        logger.info(f"[TEST] [MOCK] 가짜 임베딩 모델 로드됨 (모델명: {model_key})")
        return FakeEmbeddings(size=1536)  # nomic-embed-text 등 주요 모델 크기에 맞춤

    try:
        result: Embeddings
        if is_ollama_embedding:
            # [지연 로딩] 무거운 라이브러리는 실제 사용 시점에 임포트
            from langchain_ollama import OllamaEmbeddings

            from core.session import SessionManager

            logger.info(
                f"[MODEL] [LOAD] Ollama 임베딩 엔진 사용 | 모델: {clean_model_name}"
            )
            result = OllamaEmbeddings(
                model=clean_model_name,
                base_url=OLLAMA_BASE_URL,
            )

            SessionManager.set("current_embedding_device", "Ollama Backend")
        else:
            # --- HuggingFace 로직 (지연 로딩) ---
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings

            from core.session import SessionManager

            target_device = EMBEDDING_DEVICE.lower()
            if target_device == "auto":
                target_device = "cpu"

            display_device = "GPU" if target_device == "cuda" else "CPU"
            SessionManager.set("current_embedding_device", display_device)
            batch_size = 32 if target_device == "cuda" else 16

            # [최적화] ONNX 백엔드 활성화 (CPU/GPU 모두 지원)
            backend = "default"
            try:
                import importlib.util

                if importlib.util.find_spec("optimum") and importlib.util.find_spec(
                    "onnxruntime"
                ):
                    backend = "onnx"
                    logger.info("[MODEL] [LOAD] Optimum/ONNX 백엔드 가용 확인")
            except ImportError:
                pass

            model_kwargs: dict[str, Any] = {"device": target_device}
            if backend == "onnx":
                model_kwargs["backend"] = "onnx"

            if target_device == "cuda":
                model_kwargs["torch_dtype"] = torch.float16

            result = HuggingFaceEmbeddings(
                model_name=model_key,
                model_kwargs=model_kwargs,
                encode_kwargs={"device": target_device, "batch_size": batch_size},
                cache_folder=CACHE_DIR,
            )

            logger.info(
                f"[MODEL] [LOAD] HF 임베딩 모델 로드 성공 | 엔진: {display_device} (Backend: {backend})"
            )

        return result

    except Exception as e:
        logger.error(f"임베딩 모델 로드 실패: {e}")
        raise EmbeddingModelError(model=model_key, reason=str(e)) from e


def get_available_models() -> list[str]:
    models = _fetch_available_models_cached()
    from common.config import DEFAULT_OLLAMA_MODEL

    return models or [DEFAULT_OLLAMA_MODEL, MSG_ERROR_OLLAMA_NOT_RUNNING]


def load_llm(model_name: str) -> Any:
    # [최적화] CI/유닛 테스트 환경에서는 Ollama 서버 없이도 동작하도록 가짜 LLM 반환
    if os.getenv("IS_CI_TEST") == "true" or os.getenv("IS_UNIT_TEST") == "true":
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        logger.info(f"[TEST] [MOCK] 가짜 LLM 로드됨 (모델명: {model_name})")
        return GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="안녕하세요! RAG 시스템 테스트 응답입니다. <thinking>테스트 생각 중...</thinking> 질문에 답변해 드릴게요."
                    ),
                    "이것은 두 번째 테스트 스트리밍 조각입니다.",
                ]
            )
        )

    with monitor.track_operation(OperationType.PDF_LOADING, {"model": model_name}):
        from core.custom_ollama import DeepThinkingChatOllama

        return DeepThinkingChatOllama(
            model=model_name,
            num_predict=OLLAMA_NUM_PREDICT,
            top_p=OLLAMA_TOP_P,
            num_ctx=OLLAMA_NUM_CTX,
            temperature=OLLAMA_TEMPERATURE,
            base_url=OLLAMA_BASE_URL,
            keep_alive=OLLAMA_KEEP_ALIVE,
            # timeout 및 기타 추가 인자는 딕셔너리로 안전하게 전달하거나
            # ChatOllama 규격에 맞게 조정
        )


def is_embedding_model_cached(model_name: str) -> bool:
    cache_path = os.path.join(CACHE_DIR, f"models--{model_name.replace('/', '--')}")
    return os.path.exists(cache_path)
