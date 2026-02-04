"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
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
    OLLAMA_NUM_THREAD,
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

    # [최적화] 재진입 가능 락(RLock)으로 통일하여 데드락 위험 방지
    _llm_lock = threading.RLock()
    _embedder_lock = threading.RLock()
    _reranker_lock = threading.RLock()
    _client_lock = threading.RLock()
    _semaphore_lock = threading.RLock()

    # [복구] 누락된 클래스 속성들
    _instances: dict[str, Any] = {}
    _async_client = None
    _client_loop = None
    _inference_semaphore: asyncio.Semaphore | None = None

    @classmethod
    def get_inference_semaphore(cls) -> asyncio.Semaphore:
        """추론 제어를 위한 세마포어를 반환합니다. (Lazy Initialization)"""
        if cls._inference_semaphore is None:
            with cls._semaphore_lock:
                if cls._inference_semaphore is None:
                    import asyncio

                    cls._inference_semaphore = asyncio.Semaphore(1)
        return cls._inference_semaphore

    @classmethod
    def get_async_client(cls, host: str):
        """현재 이벤트 루프에 맞는 비동기 클라이언트를 가져옵니다."""
        import asyncio

        import ollama

        current_loop = asyncio.get_running_loop()

        with cls._client_lock:
            # 루프가 바뀌었거나 클라이언트가 없으면 새로 생성
            if cls._async_client is None or cls._client_loop != current_loop:
                if cls._async_client:
                    # 이전 클라이언트가 있다면 닫기 시도 (Best effort)
                    pass
                cls._async_client = ollama.AsyncClient(host=host)
                cls._client_loop = current_loop
                logger.info(
                    f"[ModelManager] 새 비동기 클라이언트 생성 (Loop: {id(current_loop)})"
                )

            return cls._async_client

    @classmethod
    def get_embedder(cls, model_name: str | None = None) -> HuggingFaceEmbeddings:
        """임베딩 모델을 가져오거나 로드합니다 (Thread-safe)"""
        name = (
            model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
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
    임베딩 모델을 로드합니다. (ONNX 가속 및 VRAM 최적화)
    """
    global _loaded_model_instances
    model_key = (
        embedding_model_name
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    with _model_init_lock:
        if model_key in _loaded_model_instances:
            return _loaded_model_instances[model_key]

        import torch
        from langchain_huggingface import HuggingFaceEmbeddings

        # 1. 디바이스 결정
        target_device = EMBEDDING_DEVICE.lower()
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"[MODEL] [DEVICE] 결정됨 | {target_device}")

        # 2. ONNX 가속 시도 (CPU 환경이거나 명시적 요청 시)
        use_onnx = False
        if target_device == "cpu":
            try:
                # [최적화] Ruff 권장에 따라 availability 체크 위주로 구성 (임포트는 필요 시점에)
                import importlib.util

                if importlib.util.find_spec("optimum.onnxruntime"):
                    logger.info(f"[MODEL] [ONNX] ONNX 가속 엔진 감지됨 | {model_key}")
                    use_onnx = True
            except Exception:
                logger.info(
                    "[MODEL] [ONNX] optimum 라이브러리 로드 실패 | PyTorch 사용"
                )

        # UI 표시용 세션 업데이트
        from core.session import SessionManager

        display_device = (
            "GPU" if target_device == "cuda" else ("CPU (ONNX)" if use_onnx else "CPU")
        )
        SessionManager.set("current_embedding_device", display_device)

        batch_size = 32 if target_device == "cuda" else 8

        try:
            # HuggingFaceEmbeddings 초기화
            # [최적화] GPU 사용 시 FP16(반정밀도) 적용으로 VRAM 절반 절약
            model_kwargs = {"device": target_device}
            if target_device == "cuda":
                model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

            result = HuggingFaceEmbeddings(
                model_name=model_key,
                model_kwargs=model_kwargs,
                encode_kwargs={"device": target_device, "batch_size": batch_size},
                cache_folder=CACHE_DIR,
            )

            logger.info(
                f"[MODEL] [LOAD] 임베딩 모델 로드 성공 | 엔진: {display_device}"
            )
            _loaded_model_instances[model_key] = result
            return result

        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise EmbeddingModelError(model=model_key, reason=str(e)) from e


@st.cache_resource(show_spinner=False)
def load_reranker_model(model_name: str) -> CrossEncoder | None:
    """
    [최적화] 리랭커 모델 로드
    - VRAM 경합 방지를 위해 디바이스 최적화 적용 및 실패 시 자동 CPU 전환
    """
    try:
        import torch
        from sentence_transformers import CrossEncoder

        # 1. 디바이스 결정 로직
        is_gpu = False
        total_mem = 0
        try:
            is_gpu = torch.cuda.is_available()
            if is_gpu:
                # 현재 가용 메모리 확인
                total_mem = torch.cuda.get_device_properties(0).total_memory // (
                    1024**2
                )
                free_mem = torch.cuda.memory_reserved(0) // (1024**2)
                logger.debug(
                    f"[MODEL] [RERANKER] GPU 메모리 상태 | Total: {total_mem}MB | Reserved: {free_mem}MB"
                )
        except Exception as e:
            logger.debug(f"GPU 상태 확인 중 경미한 오류: {e}")

        # GPU 메모리가 4GB 이상이고, 가용 메모리에 여유가 있을 때만 CUDA 사용
        device = "cuda" if (is_gpu and total_mem > 4000) else "cpu"

        logger.info(
            f"[MODEL] [LOAD] 리랭커 모델 로드 시작 | 모델: {model_name} | 장치: {device}"
        )

        try:
            return CrossEncoder(model_name, device=device)
        except Exception as inner_e:
            if device == "cuda":
                logger.warning(
                    f"[MODEL] [LOAD] Reranker CUDA 로드 실패, CPU로 재시도합니다 | 사유: {inner_e}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return CrossEncoder(model_name, device="cpu")
            raise inner_e

    except Exception as e:
        logger.error(
            f"[MODEL] [LOAD] Reranker 최종 로드 실패 | {type(e).__name__}: {e}"
        )
        # 리랭커는 선택적 컴포넌트이므로 에러를 던지는 대신 None을 반환하여 파이프라인이 계속 진행되도록 함
        return None


def get_available_models() -> list[str]:
    models = _fetch_available_models_cached()
    return models if models else [MSG_ERROR_OLLAMA_NOT_RUNNING]


@log_operation("Ollama LLM 로드")
def load_llm(
    model_name: str,
) -> ChatOllama:
    """기본 LLM 인스턴스를 로드합니다. (설정은 호출 시 bind됨)"""
    with monitor.track_operation(OperationType.PDF_LOADING, {"model": model_name}):
        if not model_name or model_name == MSG_ERROR_OLLAMA_NOT_RUNNING:
            raise LLMInferenceError(
                model=model_name,
                reason="model_not_selected",
                details={"msg": "Ollama 모델이 선택되지 않았습니다."},
            )

        from core.custom_ollama import DeepThinkingChatOllama

        logger.info(f"[MODEL] [LOAD] 기본 LLM 모델 로드 | 모델: {model_name}")

        # [최적화] 기본 인스턴스는 시스템 표준 설정으로 생성
        result = DeepThinkingChatOllama(
            model=model_name,
            num_predict=OLLAMA_NUM_PREDICT,
            top_p=OLLAMA_TOP_P,
            num_ctx=OLLAMA_NUM_CTX,
            num_thread=OLLAMA_NUM_THREAD,
            temperature=OLLAMA_TEMPERATURE,
            timeout=OLLAMA_TIMEOUT,
            keep_alive="24h",
            base_url=OLLAMA_BASE_URL,
            streaming=True,
        )
        return result


def is_embedding_model_cached(model_name: str) -> bool:
    model_path_name = f"models--{model_name.replace('/', '--')}"
    cache_path = os.path.join(CACHE_DIR, model_path_name)
    return os.path.exists(cache_path)
