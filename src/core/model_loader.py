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
from langchain_ollama import ChatOllama, OllamaEmbeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from sentence_transformers import CrossEncoder

import threading

from common.config import (
    CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
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
) -> Embeddings:
    """
    임베딩 모델을 로드합니다. (HuggingFace 및 Ollama 지원)
    HuggingFace 모델은 VRAM 보호를 위해 기본적으로 CPU에서 작동하도록 설정합니다.
    """
    global _loaded_model_instances
    model_key = embedding_model_name or DEFAULT_EMBEDDING_MODEL

    with _model_init_lock:
        if model_key in _loaded_model_instances:
            return _loaded_model_instances[model_key]

        from core.session import SessionManager

        # [최적화] Ollama 임베딩 여부 판별
        # 슬래시(/)가 없거나 명시적으로 'ollama:'로 시작하면 Ollama 모델로 간주
        is_ollama_embedding = "/" not in model_key or model_key.startswith("ollama:")
        clean_model_name = (
            model_key.replace("ollama:", "") if "ollama:" in model_key else model_key
        )

        try:
            if is_ollama_embedding:
                logger.info(
                    f"[MODEL] [LOAD] Ollama 임베딩 엔진 사용 | 모델: {clean_model_name}"
                )
                # OllamaEmbeddings는 내부적으로 원격 호출을 수행하므로 별도의 디바이스 설정이 필요 없음
                result = OllamaEmbeddings(
                    model=clean_model_name,
                    base_url=OLLAMA_BASE_URL,
                    # 최신 langchain-ollama는 전역 설정을 잘 따름
                )

                # 가용성 체크 (첫 호출 시도)
                try:
                    # 빈 텍스트로 테스트하여 모델 존재 및 임베딩 지원 여부 확인
                    result.embed_query("test")
                    logger.info(
                        f"[MODEL] [LOAD] Ollama 임베딩 모델 가용성 확인 완료: {clean_model_name}"
                    )
                except Exception as ollama_e:
                    logger.warning(
                        f"Ollama 임베딩 가용성 확인 실패 (무시 가능): {ollama_e}"
                    )

                SessionManager.set("current_embedding_device", "Ollama Backend")
                _loaded_model_instances[model_key] = result
                return result

            # --- 기존 HuggingFace 로직 ---
            import torch
            from langchain_huggingface import HuggingFaceEmbeddings

            # [최적화] 임베딩 디바이스 결정 로직 복구
            target_device = EMBEDDING_DEVICE.lower()
            if target_device == "auto":
                target_device = "cpu"

            display_device = "GPU" if target_device == "cuda" else "CPU"
            SessionManager.set("current_embedding_device", display_device)
            batch_size = 32 if target_device == "cuda" else 16

            # [최적화] ONNX 백엔드 활성화 (CPU/GPU 모두 지원)
            model_kwargs = {"device": target_device, "backend": "onnx"}
            if target_device == "cuda":
                model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

            # 표준 HuggingFaceEmbeddings 로드
            result = HuggingFaceEmbeddings(
                model_name=model_key,
                model_kwargs=model_kwargs,
                encode_kwargs={"device": target_device, "batch_size": batch_size},
                cache_folder=CACHE_DIR,
            )

            logger.info(
                f"[MODEL] [LOAD] HF 임베딩 모델 로드 성공 | 엔진: {display_device} (ONNX) | 배치: {batch_size}"
            )
            _loaded_model_instances[model_key] = result
            return result

        except Exception as e:
            logger.error(f"임베딩 모델 로드 실패: {e}")
            raise EmbeddingModelError(model=model_key, reason=str(e)) from e


class OllamaReranker:
    """
    Ollama의 리랭킹 모델을 호출하는 래퍼 클래스.
    sentence-transformers의 CrossEncoder.predict와 유사한 인터페이스를 제공합니다.
    """

    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url

    async def apredict(
        self, pairs: list[list[str]], batch_size: int = 32
    ) -> list[float]:
        """
        [최적화] 비동기 병렬 Ollama 리랭킹.
        LLM 기반의 지능형 채점을 병렬로 수행하여 속도를 극대화합니다.
        """
        import re

        if not pairs:
            return []

        query = pairs[0][0]
        documents = [p[1] for p in pairs]

        # ModelManager의 비동기 클라이언트 사용
        client = ModelManager.get_async_client(host=self.base_url)

        async def _score_batch(batch: list[str], start_idx: int) -> list[float]:
            # LLM에게 리랭킹 점수를 요구하는 최적화된 프롬프트
            scoring_prompt = (
                f"Task: Evaluate document relevance to the query.\n"
                f"Query: {query}\n\n"
                "For each document, provide a relevance score between 0.0 and 1.0.\n"
                "Output ONLY a list of numbers. Example: [0.9, 0.1, 0.5]\n\n"
            )
            for j, doc in enumerate(batch):
                scoring_prompt += f"Document {j + 1}: {doc[:800]}\n\n"

            try:
                # Chat API 비동기 호출
                response = await client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": scoring_prompt}],
                    options={"temperature": 0.0, "num_predict": 50},
                )
                content = response["message"]["content"]

                # 숫자 추출
                found = re.findall(r"0\.\d+|1\.0|\d\.\d+", content)
                batch_scores = [float(s) for s in found][: len(batch)]

                # 안전장치: 숫자를 못 찾았을 경우 기본 순차 점수 부여
                while len(batch_scores) < len(batch):
                    batch_scores.append(0.5 - ((start_idx + len(batch_scores)) * 0.02))

                return batch_scores
            except Exception as batch_e:
                logger.error(f"[Reranker] 배치 채점 실패: {batch_e}")
                return [0.5] * len(batch)

        # 5개씩 배치화하여 병렬 실행
        batch_tasks = []
        for i in range(0, len(documents), 5):
            batch_tasks.append(_score_batch(documents[i : i + 5], i))

        all_batch_results = await asyncio.gather(*batch_tasks)
        all_scores = []
        for batch_res in all_batch_results:
            all_scores.extend(batch_res)

        return all_scores

    def predict(self, pairs: list[list[str]], batch_size: int = 32) -> list[float]:
        """
        동기 방식의 리랭킹 호출 (호환성 유지).
        """
        try:
            # 현재 루프가 있는지 확인
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    return self._predict_sync(pairs, batch_size)
                # 루프는 있으나 작동 중이 아니면 run 호출
                return asyncio.run(self.apredict(pairs, batch_size))
            except RuntimeError:
                # 루프가 없는 경우
                return asyncio.run(self.apredict(pairs, batch_size))
        except Exception as e:
            logger.error(f"[Reranker] predict 오류: {e}")
            return [0.5] * len(pairs)

    def _predict_sync(
        self, pairs: list[list[str]], batch_size: int = 32
    ) -> list[float]:
        """기존 동기 처리 로직 (Fallback)"""
        import re

        import ollama

        query = pairs[0][0]
        documents = [p[1] for p in pairs]
        client = ollama.Client(host=self.base_url)

        try:
            all_scores = []
            for i in range(0, len(documents), 5):
                batch = documents[i : i + 5]
                scoring_prompt = (
                    f"Task: Evaluate document relevance to the query.\n"
                    f"Query: {query}\n\n"
                    "For each document, provide a relevance score between 0.0 and 1.0.\n"
                    "Output ONLY a list of numbers. Example: [0.9, 0.1, 0.5]\n\n"
                )
                for j, doc in enumerate(batch):
                    scoring_prompt += f"Document {j + 1}: {doc[:800]}\n\n"

                response = client.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": scoring_prompt}],
                    options={"temperature": 0.0, "num_predict": 50},
                )
                content = response["message"]["content"]
                found = re.findall(r"0\.\d+|1\.0|\d\.\d+", content)
                batch_scores = [float(s) for s in found][: len(batch)]
                while len(batch_scores) < len(batch):
                    batch_scores.append(0.5 - (i * 0.02))
                all_scores.extend(batch_scores)
            return all_scores
        except Exception as e:
            logger.error(f"[Reranker] LLM 채점 실패: {e}")
            return [1.0 - (idx * 0.01) for idx in range(len(documents))]


@st.cache_resource(show_spinner=False)
def load_reranker_model(model_name: str) -> Any:
    """
    [최적화] 리랭커 모델 로드
    - Ollama 모델인 경우 OllamaReranker 반환
    - HF 모델인 경우 CrossEncoder 반환
    """
    if not model_name:
        return None

    # Ollama 리랭커 감지 (슬래시가 없거나 명시적 접두사)
    is_ollama_reranker = "/" not in model_name or "reranker" in model_name.lower()

    if is_ollama_reranker:
        logger.info(f"[MODEL] [LOAD] Ollama 리랭킹 엔진 사용 | 모델: {model_name}")
        return OllamaReranker(model_name=model_name, base_url=OLLAMA_BASE_URL)

    try:
        import torch
        from sentence_transformers import CrossEncoder

        # 1. 디바이스 결정 로직 (EMBEDDING_DEVICE 설정 준수)
        target_device = EMBEDDING_DEVICE.lower()

        is_gpu = torch.cuda.is_available()
        if target_device == "auto":
            # auto일 때만 메모리 상황에 따라 결정
            total_mem = 0
            try:
                if is_gpu:
                    total_mem = torch.cuda.get_device_properties(0).total_memory // (
                        1024**2
                    )
            except Exception:
                pass
            device = "cuda" if (is_gpu and total_mem > 4000) else "cpu"
        else:
            # 명시적 설정이 있으면 그에 따름 (예: cpu)
            device = target_device

        logger.info(
            f"[MODEL] [LOAD] 리랭커 모델 로드 시도 | 모델: {model_name} | 장치: {device}"
        )

        try:
            # [수정] device 인자를 명시적으로 전달
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
        return None


def get_available_models() -> list[str]:
    """사용 가능한 Ollama 모델 목록을 반환합니다. 실패 시 기본 모델 포함."""
    models = _fetch_available_models_cached()
    from common.config import DEFAULT_OLLAMA_MODEL

    # 서버가 꺼져 있거나 목록이 비어있으면 기본 모델이라도 반환
    if not models or (len(models) == 1 and MSG_ERROR_OLLAMA_NOT_RUNNING in models):
        return [DEFAULT_OLLAMA_MODEL, MSG_ERROR_OLLAMA_NOT_RUNNING]

    return models


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
