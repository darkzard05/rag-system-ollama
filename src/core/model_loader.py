"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import streamlit as st
from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

import threading

from common.config import (
    CACHE_DIR,
    EMBEDDING_DEVICE,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_TIMEOUT,
    OLLAMA_TOP_P,
)
from common.exceptions import EmbeddingModelError, LLMInferenceError
from common.utils import log_operation
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
    _llm_lock = threading.Lock()
    _embedder_lock = threading.Lock()
    _reranker_lock = threading.Lock()
    _client_lock = threading.Lock()

    _instances: dict[str, Any] = {}
    _async_client = None  # 싱글톤 비동기 클라이언트

    @classmethod
    def get_async_client(cls, host: str):
        """커넥션 풀을 유지하는 비동기 클라이언트를 가져옵니다."""
        with cls._client_lock:
            if cls._async_client is None:
                import ollama
                cls._async_client = ollama.AsyncClient(host=host)
            return cls._async_client

    @classmethod
    def get_embedder(cls, model_name: str | None = None) -> HuggingFaceEmbeddings:
        """임베딩 모델을 가져오거나 로드합니다 (Thread-safe)"""
        name = model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        with cls._embedder_lock:
            if name not in cls._instances or cls._instances[name] is None:
                cls._instances[name] = load_embedding_model(name)
            return cls._instances[name]

    @classmethod
    def get_llm(cls, model_name: str, **kwargs) -> ChatOllama:
        """LLM 클라이언트 인스턴스를 가져오거나 생성합니다."""
        # ChatOllama는 가벼운 클라이언트이므로 캐싱하되,
        # 설정(temperature 등)이 다를 수 있으므로 키에 설정을 포함하거나
        # 호출 시마다 새로 생성하는 것이 안전할 수 있습니다.
        # 여기서는 모델명 기반으로 캐싱하여 재사용합니다.
        with cls._llm_lock:
            # 설정값 변화를 반영하기 위해 튜플 형태의 키 사용
            temp = kwargs.get("temperature", OLLAMA_TEMPERATURE)
            ctx = kwargs.get("num_ctx", OLLAMA_NUM_CTX)
            cache_key = f"llm_{model_name}_{temp}_{ctx}"

            if cache_key not in cls._instances:
                cls._instances[cache_key] = load_llm(model_name, **kwargs)
            return cls._instances[cache_key]

    @classmethod
    def get_reranker(cls, model_name: str) -> CrossEncoder | None:
        """리랭커 모델을 가져오거나 로드합니다."""
        with cls._reranker_lock:
            if model_name not in cls._instances:
                cls._instances[model_name] = load_reranker_model(model_name)
            return cls._instances[model_name]

    @classmethod
    def clear_vram(cls):
        """[위험] 모든 모델 인스턴스를 제거하여 VRAM을 강제로 비웁니다."""
        with cls._llm_lock, cls._embedder_lock, cls._reranker_lock:
            cls._instances.clear()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("[System] [VRAM] 모든 모델 인스턴스 해제 및 캐시 클리어")


@st.cache_data(ttl=600)  # 모델 목록 캐시 10분
def _fetch_available_models_cached() -> list[str]:
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


@st.cache_resource(show_spinner=False)
def load_embedding_model(
    embedding_model_name: str | None = None,
) -> HuggingFaceEmbeddings:
    """
    임베딩 모델을 로드합니다. (Thread-safe & VRAM Optimized)
    """
    global _loaded_model_instances
    model_key = (
        embedding_model_name
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # [수정] 중복 진입 방지를 위한 이중 확인 잠금 (Double-Checked Locking)
    with _model_init_lock:
        if model_key in _loaded_model_instances:
            # 이미 다른 스레드에서 로드를 완료했으므로 공유 인스턴스 반환
            return _loaded_model_instances[model_key]

        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(f">>> load_embedding_model 진입 (Model: {model_key})")
        logger.info(f"[Debug] EMBEDDING_DEVICE 설정값: {EMBEDDING_DEVICE}")
        logger.info(f"[Debug] torch.cuda.is_available(): {torch.cuda.is_available()}")

        # 1. 디바이스 결정 로직 (설정값 최우선)
        target_device = EMBEDDING_DEVICE.lower()

        if target_device == "auto":
            if torch.cuda.is_available():
                target_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                target_device = "mps"
            else:
                target_device = "cpu"

        logger.info(f"[MODEL] [DEVICE] 최종 결정된 디바이스 | {target_device}")

        # UI 표시를 위해 세션 상태에 기록
        from core.session import SessionManager
        display_device = "GPU" if target_device.lower() == "cuda" else target_device.upper()
        SessionManager.set("current_embedding_device", display_device)

        batch_size = 32 if "cuda" in target_device else 4
        logger.info(f"[MODEL] [LOAD] 임베딩 모델 로드 시작 | 모델: {model_key} | 장치: {target_device}")

        try:
            # [최적화] ONNX 가속 사용 시도
            try:
                from langchain_huggingface import HuggingFaceOptimumEmbeddings
                logger.info(f"[MODEL] [LOAD] ONNX 가속 임베딩 로드 시도 | 모델: {model_key}")
                result = HuggingFaceOptimumEmbeddings(
                    model_name=model_key,
                    cache_folder=CACHE_DIR,
                    model_kwargs={"device": target_device},
                    encode_kwargs={"device": target_device, "batch_size": batch_size},
                )
                logger.info(f"[MODEL] [LOAD] ONNX 가속 임베딩 로드 성공 | 모델: {model_key}")
            except (ImportError, Exception) as e:
                logger.info(f"[MODEL] [LOAD] 표준 임베딩 로드로 전환합니다 | 원인: {type(e).__name__}")
                result = HuggingFaceEmbeddings(
                    model_name=model_key,
                    model_kwargs={"device": target_device},
                    encode_kwargs={"device": target_device, "batch_size": batch_size},
                    cache_folder=CACHE_DIR,
                )

            _loaded_model_instances[model_key] = result
            logger.info(f"[MODEL] [LOAD] 임베딩 모델 로드 완료 | 모델: {model_key}")
            return result

        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise EmbeddingModelError(model=model_key, reason=str(e)) from e


@st.cache_resource(show_spinner=False)
def load_reranker_model(model_name: str) -> CrossEncoder | None:
    """
    [최적화] 리랭커 모델 로드
    - VRAM 경합 방지를 위해 디바이스 최적화 적용
    """
    try:
        import torch
        from sentence_transformers import CrossEncoder

        # [최적화] 리랭커는 상대적으로 가볍고 LLM과 VRAM을 공유하면 성능이 저하되므로,
        # GPU 메모리가 넉넉하지 않은 경우 CPU 사용을 고려
        is_gpu, total_mem = 0, 0
        try:
            is_gpu = torch.cuda.is_available()
            if is_gpu:
                total_mem = torch.cuda.get_device_properties(0).total_memory // (
                    1024**2
                )
        except Exception:
            pass

        # [수정] 6GB GPU 환경에서도 리랭커 가속을 위해 기준을 4GB로 하향
        device = "cuda" if (is_gpu and total_mem > 4000) else "cpu"

        logger.info(
            f"[MODEL] [LOAD] 리랭커 모델 로드 시작 | 모델: {model_name} | 장치: {device}"
        )
        return CrossEncoder(model_name, device=device)
    except Exception as e:
        logger.error(f"[MODEL] [LOAD] Reranker 로드 실패 | {e}")
        raise EmbeddingModelError(
            model=model_name,
            reason="Reranker 모델 로드 실패",
            details={"error": str(e)},
        ) from e


def get_available_models() -> list[str]:
    models = _fetch_available_models_cached()
    return models if models else [MSG_ERROR_OLLAMA_NOT_RUNNING]


@log_operation("Ollama LLM 로드")
def load_llm(
    model_name: str,
    temperature: float = OLLAMA_TEMPERATURE,
    num_predict: int = OLLAMA_NUM_PREDICT,
    top_p: float = OLLAMA_TOP_P,
    num_ctx: int = OLLAMA_NUM_CTX,
    timeout: float = OLLAMA_TIMEOUT,
) -> ChatOllama:
    with monitor.track_operation(
        OperationType.PDF_LOADING, {"model": model_name, "timeout": timeout}
    ):
        if not model_name or model_name == MSG_ERROR_OLLAMA_NOT_RUNNING:
            raise LLMInferenceError(
                model=model_name,
                reason="model_not_selected",
                details={"msg": "Ollama 모델이 선택되지 않았습니다."},
            )

        from core.custom_ollama import DeepThinkingChatOllama

        logger.info(
            f"[MODEL] [LOAD] LLM 모델 로드 | 모델: {model_name} | 타임아웃: {timeout}s"
        )
        logger.debug(
            f"[MODEL] [CONFIG] Ollama 설정 | predict: {num_predict} | ctx: {num_ctx} | temp: {temperature}"
        )

        # ChatOllama 사용으로 사고 과정(thinking) 필드 지원 강화
        # [Custom] DeepThinkingChatOllama를 사용하여 Ollama의 thinking 필드 캡처 지원
        result = DeepThinkingChatOllama(
            model=model_name,
            num_predict=num_predict,
            top_p=top_p,
            num_ctx=num_ctx,
            temperature=temperature,
            timeout=timeout,
            keep_alive="60m",
            base_url=OLLAMA_BASE_URL,
            streaming=True,
        )
        return result


def is_embedding_model_cached(model_name: str) -> bool:
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)
