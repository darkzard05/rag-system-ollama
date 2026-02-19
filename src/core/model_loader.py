"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

from common.config import (
    CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_NUM_THREAD,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    OLLAMA_TOP_P,
)
from common.exceptions import EmbeddingModelError
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

logger = logging.getLogger(__name__)
monitor = get_performance_monitor()

# [추가] 다중 스레드 중복 로딩 방지를 위한 전역 락 및 공유 저장소
_model_init_lock = threading.RLock()
_loaded_model_instances: dict[str, Any] = {}


class ModelManager:
    """
    시스템 전체의 모델 인스턴스를 관리하는 중앙 클래스.
    UI와 API가 공동으로 사용하여 중복 로딩 및 VRAM 낭비를 방지합니다.
    """

    # [최적화] 재진입 가능 락(RLock)으로 통일하여 데드락 위험 방지
    _llm_lock = threading.RLock()
    _embedder_lock = threading.RLock()
    _reranker_lock = threading.RLock()
    _flashrank_lock = threading.RLock()
    _client_lock = threading.RLock()
    _semaphore_lock = threading.RLock()

    # [복구] 누락된 클래스 속성들
    _instances: dict[str, Any] = {}
    _sync_client = None
    _async_client = None
    _client_loop = None
    _inference_semaphore: asyncio.Semaphore | None = None

    @classmethod
    def get_flashranker(cls, model_name: str = "ms-marco-TinyBERT-L-2-v2") -> Any:
        """FlashRank 리랭커 모델을 가져오거나 로드합니다 (고속 CPU 리랭킹)"""
        with cls._flashrank_lock:
            cache_key = f"flashrank_{model_name}"
            if cache_key not in cls._instances:
                from flashrank import Ranker

                logger.info(f"[MODEL] [LOAD] FlashRank 리랭커 로드 중: {model_name}")
                cls._instances[cache_key] = Ranker(
                    model_name=model_name, cache_dir=CACHE_DIR
                )
            return cls._instances[cache_key]

    @classmethod
    def get_inference_semaphore(cls) -> asyncio.Semaphore:
        """추론 제어를 위한 세마포어를 반환합니다. (Thread & Loop safe)"""
        if cls._inference_semaphore is None:
            with cls._semaphore_lock:
                if cls._inference_semaphore is None:
                    try:
                        # 실행 중인 루프에서 생성 시도
                        asyncio.get_running_loop()
                        cls._inference_semaphore = asyncio.Semaphore(1)
                    except RuntimeError:
                        # 루프가 없는 환경이면 임시 세마포어 반환
                        return asyncio.Semaphore(1)
        return cls._inference_semaphore

    @classmethod
    def get_client(cls, host: str):
        """캐싱된 동기 Ollama 클라이언트를 가져옵니다."""
        import ollama

        with cls._client_lock:
            needs_new = (
                cls._sync_client is None
                or str(getattr(cls._sync_client, "base_url", "")) != host
            )
            if needs_new:
                cls._sync_client = ollama.Client(host=host)
                logger.info(f"[ModelManager] 새 동기 클라이언트 생성 (Host: {host})")
            return cls._sync_client

    @classmethod
    def get_async_client(cls, host: str):
        """현재 이벤트 루프에 맞는 비동기 클라이언트를 가져옵니다."""
        import ollama

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # 루프가 없는 경우 (동기 환경에서 비동기 클라이언트 요청 시)
            return ollama.AsyncClient(host=host)

        with cls._client_lock:
            # 루프가 바뀌었거나 클라이언트가 없거나 호스트가 다르면 새로 생성
            needs_new = (
                cls._async_client is None
                or cls._client_loop != current_loop
                or str(getattr(cls._async_client, "base_url", "")) != host
            )

            if needs_new:
                cls._async_client = ollama.AsyncClient(host=host)
                cls._client_loop = current_loop
                logger.info(
                    f"[ModelManager] 새 비동기 클라이언트 생성 (Host: {host}, Loop: {id(current_loop)})"
                )

            return cls._async_client

    @classmethod
    def get_embedder(cls, model_name: str | None = None) -> Embeddings:
        """임베딩 모델을 가져오거나 로드합니다 (Thread-safe)"""
        name = model_name or DEFAULT_EMBEDDING_MODEL
        with cls._embedder_lock:
            if name not in cls._instances or cls._instances[name] is None:
                cls._instances[name] = load_embedding_model(name)
            return cls._instances[name]

    @classmethod
    def get_llm(cls, model_name: str, **kwargs) -> Any:
        """LLM 클라이언트 인스턴스를 가져오거나 생성합니다 (Single-instance per model)."""
        with cls._llm_lock:
            # [최적화] 모델명만 키로 사용하여 중복 인스턴스 생성 방지
            cache_key = f"base_llm_{model_name}"

            if cache_key not in cls._instances:
                # 기본 인스턴스는 한 번만 로드
                cls._instances[cache_key] = load_llm(model_name)

            base_llm = cls._instances[cache_key]

            # [핵심] 설정값(온도 등)은 bind()를 통해 동적으로 전달 (메모리 절감)
            if not kwargs:
                return base_llm

            # ChatOllama의 bind()는 가벼운 RunnableBinding을 반환함
            return base_llm.bind(**kwargs)

    @classmethod
    def clear_vram(cls):
        """[위험] 모든 모델 인스턴스를 제거하여 VRAM을 강제로 비웁니다."""
        with cls._llm_lock, cls._embedder_lock, cls._reranker_lock:
            cls._instances.clear()
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
    global _loaded_model_instances
    model_key = embedding_model_name or DEFAULT_EMBEDDING_MODEL

    with _model_init_lock:
        if model_key in _loaded_model_instances:
            return _loaded_model_instances[model_key]

        # [최적화] Ollama 임베딩 여부 판별
        is_ollama_embedding = "/" not in model_key or model_key.startswith("ollama:")
        clean_model_name = (
            model_key.replace("ollama:", "") if "ollama:" in model_key else model_key
        )

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

                model_kwargs = {"device": target_device}
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

            _loaded_model_instances[model_key] = result
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
            num_thread=OLLAMA_NUM_THREAD,
            temperature=OLLAMA_TEMPERATURE,
            timeout=OLLAMA_TIMEOUT,
            base_url=OLLAMA_BASE_URL,
            keep_alive=OLLAMA_KEEP_ALIVE,
            streaming=True,
        )


def is_embedding_model_cached(model_name: str) -> bool:
    cache_path = os.path.join(CACHE_DIR, f"models--{model_name.replace('/', '--')}")
    return os.path.exists(cache_path)
