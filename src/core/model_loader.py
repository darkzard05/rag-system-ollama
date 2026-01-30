"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import streamlit as st
from langchain_ollama import ChatOllama

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

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

# [추가] 동시 로딩 방지를 위한 글로벌 비동기 락
_loading_lock = asyncio.Lock()


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
    import torch
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(">>> load_embedding_model 진입")

    # 1. 디바이스 결정 로직 (VRAM 8GB 미만 감지 포함)
    target_device = EMBEDDING_DEVICE

    if target_device == "auto":
        if torch.cuda.is_available():
            try:
                # RTX 2060(6GB) 등 저사양 GPU 환경에서는 LLM 속도 유지를 위해 CPU 강제 할당
                props = torch.cuda.get_device_properties(0)
                total_vram_gb = props.total_memory / (1024**3)

                if total_vram_gb < 7.5:  # 8GB 미만 (여유값 0.5GB)
                    logger.info(
                        f"[Model] 저사양 VRAM 감지({total_vram_gb:.1f}GB): 임베딩을 CPU로 오프로딩합니다."
                    )
                    target_device = "cpu"
                else:
                    target_device = "cuda"
            except Exception as e:
                logger.warning(f"CUDA 속성 조회 실패(CPU로 전환): {e}")
                target_device = "cpu"
        else:
            target_device = "cpu"

    # [최적화] 배치 크기 설정
    batch_size = 32 if target_device == "cuda" else 4

    logger.info(
        f"[System] [Model] 임베딩 모델 로드 시작: {embedding_model_name} (Device: {target_device})"
    )

    try:
        # HuggingFaceEmbeddings는 내부적으로 스레드 세이프하지 않을 수 있으므로 주의 필요
        result = HuggingFaceEmbeddings(
            model_name=embedding_model_name
            or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={"device": target_device},
            encode_kwargs={"device": target_device, "batch_size": batch_size},
            cache_folder=CACHE_DIR,
        )
        logger.info("[System] [Model] 임베딩 모델 로드 성공")
        return result
    except Exception as e:
        logger.error(f"임베딩 모델 로드 실패: {e}")
        raise EmbeddingModelError(model=embedding_model_name, reason=str(e)) from e


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
            f"[System] [Model] 리랭커 모델 로드: {model_name} (디바이스: {device})"
        )
        return CrossEncoder(model_name, device=device)
    except Exception as e:
        logger.error(f"Reranker 로드 실패: {e}")
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
            f"[System] [Model] LLM 모델 로드: {model_name} (타임아웃: {timeout}s)"
        )
        logger.debug(
            f"Ollama 로드 설정: predict={num_predict}, ctx={num_ctx}, temp={temperature}"
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
