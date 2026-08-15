"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

import contextlib

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    ENABLE_OLLAMA_PRESSURE_FALLBACK,
    MAX_CACHED_MODELS,
    MAX_CONCURRENT_INFERENCE,
    MODEL_CACHE_DIR,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_THINKING,
    OLLAMA_TIMEOUT,
    OLLAMA_TOP_P,
)
from common.exceptions import EmbeddingModelError
from common.system_pressure import (
    host_pressure_exceeded,
    ollama_backend_active,
)
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        try:
            import torch as _torch_module

            _torch = _torch_module
        except ImportError:
            _torch = None
    return _torch


_psutil = None


def _get_psutil():
    global _psutil
    if _psutil is None:
        try:
            import psutil as _psutil_module

            _psutil = _psutil_module
        except ImportError:
            _psutil = None
    return _psutil


logger = logging.getLogger(__name__)


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
    def get_filtered_models(cls, available_models: list[str]) -> dict[str, list[str]]:
        """모델 목록을 LLM과 임베딩 모델로 분류하여 반환합니다."""
        from common.config import (
            DEFAULT_EMBEDDING_MODEL,
            DEFAULT_OLLAMA_MODEL,
        )

        safe_models = [m for m in available_models if m and "---" not in str(m)]
        embed_keywords = ["embed", "bge", "nomic", "mxbai", "snowflake"]

        embedding_candidates = [
            m for m in safe_models if any(kw in str(m).lower() for kw in embed_keywords)
        ]
        actual_embeddings = sorted(set(embedding_candidates))
        if DEFAULT_EMBEDDING_MODEL not in actual_embeddings:
            actual_embeddings.append(DEFAULT_EMBEDDING_MODEL)
        actual_embeddings.sort()

        llm_candidates = [m for m in safe_models if m not in embedding_candidates]
        # 중복 제거
        actual_llms = (
            sorted(set(llm_candidates)) if llm_candidates else [DEFAULT_OLLAMA_MODEL]
        )
        if DEFAULT_OLLAMA_MODEL not in actual_llms:
            actual_llms.append(DEFAULT_OLLAMA_MODEL)
        actual_llms.sort()

        return {"llm": actual_llms, "embedding": actual_embeddings}

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
    @contextlib.asynccontextmanager
    async def inference_session(cls):
        """추론 세마포어를 안전하게 관리하는 비동기 컨텍스트 매니저."""
        from .resource_manager import get_resource_manager

        async with get_resource_manager().inference_session():
            yield

    @classmethod
    async def acquire_inference_lock(cls):
        """비동기 세마포어를 획득합니다."""
        from .resource_manager import get_resource_manager

        await get_resource_manager().acquire_inference_lock()

    @classmethod
    def release_inference_lock(cls):
        """세마포어를 해제합니다."""
        from .resource_manager import get_resource_manager

        get_resource_manager().release_inference_lock()

    @classmethod
    def _get_from_cache(cls, key: str) -> Any | None:
        """LRU 캐시에서 인스턴스를 가져오고 순서를 갱신합니다."""
        if key in cls._instances:
            cls._instances.move_to_end(key)
            return cls._instances[key]
        return None

    @classmethod
    async def _check_memory_pressure(cls):
        """현재 VRAM/RAM 사용량을 확인하고 압박 시 가장 오래된 모델을 방출합니다."""

        # 1. GPU VRAM 체크 (사용 가능한 경우)
        _torch = _get_torch()
        if _torch and _torch.cuda.is_available():
            try:
                # 현재 디바이스의 메모리 정보 (MB 단위)
                device = _torch.cuda.current_device()
                total_mem = _torch.cuda.get_device_properties(device).total_memory / (
                    1024**2
                )
                reserved_mem = _torch.cuda.memory_reserved(device) / (1024**2)

                # 실질 점유율 (Reserved 기준)
                usage_pct = (reserved_mem / total_mem) * 100

                if usage_pct > 90:  # 90% 이상 사용 시
                    logger.warning(
                        f"[ModelManager] VRAM 압박 감지 ({usage_pct:.1f}%). 자원 방출을 시작합니다."
                    )
                    await cls._evict_oldest_model()
                    return True
            except Exception as e:
                logger.debug(f"VRAM 체크 실패 (무시): {e}")

        # 2. Ollama 압력 폴백 (torch.cuda 미사용 기본 배포)
        if (
            ENABLE_OLLAMA_PRESSURE_FALLBACK
            and ollama_backend_active()
            and host_pressure_exceeded(threshold=90.0)
        ):
            logger.warning(
                "[ModelManager] 호스트 RAM 압박 감지 (Ollama 폴백, >90%). "
                "자원 방출을 시작합니다."
            )
            await cls._evict_oldest_model()
            return True

        # 3. 시스템 RAM 체크 (폴백)
        _psutil = _get_psutil()
        if _psutil:
            mem = _psutil.virtual_memory()
            if mem.percent > 95:
                logger.warning(
                    f"[ModelManager] 시스템 RAM 부족 ({mem.percent}%). 자원 방출을 시작합니다."
                )
                await cls._evict_oldest_model()
                return True
        return False

    @classmethod
    async def _add_to_cache(cls, key: str, instance: Any):
        """LRU 캐시에 인스턴스를 추가합니다. 필요 시 가장 오래된 것을 방출합니다."""
        # 추가 전 메모리 상태 확인
        await cls._check_memory_pressure()

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
    async def get_flashranker(cls, model_name: str | None = None) -> Any:
        """FlashRank 리랭커 모델을 가져오거나 로드합니다 (고속 CPU 리랭킹)"""
        from .resource_manager import get_resource_manager

        return await get_resource_manager().get_flashranker(model_name)

    @classmethod
    def get_client(cls, host: str):
        """캐싱된 동기 Ollama 클라이언트를 가져옵니다."""
        from .resource_manager import get_resource_manager

        return get_resource_manager().get_client(host)

    @classmethod
    async def get_async_client(cls, host: str):
        """현재 이벤트 루프에 맞는 비동기 클라이언트를 가져옵니다."""
        from .resource_manager import get_resource_manager

        return await get_resource_manager().get_async_client(host)

    @classmethod
    async def get_embedder(cls, model_name: str | None = None) -> Embeddings:
        """임베딩 모델을 가져오거나 로드합니다 (Thread-safe, LRU 캐시 적용)"""
        from .resource_manager import get_resource_manager

        return await get_resource_manager().get_embedder(model_name)

    @classmethod
    async def get_llm(cls, model_name: str, **kwargs) -> Any:
        """LLM 클라이언트 인스턴스를 가져오거나 생성합니다 (Single-instance per model, LRU 캐시 적용)."""
        from .resource_manager import get_resource_manager

        return await get_resource_manager().get_llm(model_name, **kwargs)

    @classmethod
    async def clear_vram(cls):
        """[위험] 모든 모델 인스턴스를 제거하고 Ollama 모델을 GPU에서 강제로 내립니다."""
        from .resource_manager import get_resource_manager

        await get_resource_manager().clear_vram()


def _fetch_available_models_cached() -> list[str]:
    """Ollama 모델 목록을 가져옵니다. (UI 종속성 제거)"""
    try:
        import ollama

        client = ollama.Client(host=OLLAMA_BASE_URL, timeout=5)
        ollama_response = client.list()
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


def _keep_alive_seconds() -> int:
    """OLLAMA_KEEP_ALIVE(기본 "30m")를 초 단위 정수로 변환.

    OllamaEmbeddings.keep_alive는 ``int | None`` 타입이므로 문자열을 그대로
    넘기면 pydantic 검증 오류가 발생한다. 파싱 실패 시 1800초(30분)로 폴백한다.
    """
    match = re.fullmatch(r"(\d+)m", OLLAMA_KEEP_ALIVE.strip())
    if match:
        return int(match.group(1)) * 60
    return 1800


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

            # [R2-07] Ollama `/api/embed` truncate 기본값(true)은 임베딩 모델
            # 컨텍스트 초과 입력을 무음 잘라낸다. langchain_ollama 0.3.x는
            # truncate를 생성자로 받지도 않으므로, 서브클래스에서 명시적으로
            # truncate=False를 전달해 과잉 입력을 에러로 표면화한다.
            class _NoTruncateOllamaEmbeddings(OllamaEmbeddings):
                """Ollama 임베딩 — `/api/embed` truncate=False 명시."""

                truncate: bool = False

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    if not self._client:
                        msg = (
                            "Ollama client is not initialized. "
                            "Please ensure Ollama is running and the model is loaded."
                        )
                        raise ValueError(msg)
                    return self._client.embed(
                        self.model,
                        texts,
                        truncate=self.truncate,
                        options=self._default_params,
                        keep_alive=self.keep_alive,
                    )["embeddings"]

                async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                    if not self._async_client:
                        msg = (
                            "Ollama client is not initialized. "
                            "Please ensure Ollama is running and the model is loaded."
                        )
                        raise ValueError(msg)
                    return (
                        await self._async_client.embed(
                            self.model,
                            texts,
                            truncate=self.truncate,
                            options=self._default_params,
                            keep_alive=self.keep_alive,
                        )
                    )["embeddings"]

            logger.info(
                f"[MODEL] [LOAD] Ollama 임베딩 엔진 사용 | 모델: {clean_model_name}"
            )
            result = _NoTruncateOllamaEmbeddings(
                model=clean_model_name,
                base_url=OLLAMA_BASE_URL,
                keep_alive=_keep_alive_seconds(),
            )

            SessionManager.set("current_embedding_device", "Ollama Backend")
        else:
            # --- HuggingFace 로직 (지연 로딩) ---
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
            encode_kwargs: dict[str, Any] = {
                "device": target_device,
                "batch_size": batch_size,
                # [R3a-07] HF 임베더 출력 L2 정규화 — 코사인 일관성 계약.
                # 인덱스(use_l2_norm)와 쿼리 양쪽이 단위벡터일 때 IP(내적)=코사인.
                # ONNX+CPU 가속 경로에만 국한하지 않고 전 백엔드(CPU/CUDA)에 적용.
                "normalize_embeddings": True,
            }

            if target_device == "cuda":
                # [최적화] GPU 사용 시 fp16 정밀도 적용하여 VRAM 절약 및 가속
                import torch

                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["trust_remote_code"] = True
                logger.info(f"[MODEL] [VRAM] {model_key} 로드 시 fp16 정밀도 적용")

            if backend == "onnx":
                model_kwargs["backend"] = "onnx"

            if target_device == "cuda":
                # [수정] SentenceTransformer 직접 생성 시 torch_dtype 관련 오류 방지를 위해 일단 제외
                # 필요한 경우 model_kwargs 대신 별도 최적화 경로 사용
                pass

            result = HuggingFaceEmbeddings(
                model_name=model_key,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder=MODEL_CACHE_DIR,
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
            messages=itertools.cycle(
                [
                    AIMessage(
                        content="안녕하세요! RAG 시스템 테스트 응답입니다. <thinking>테스트 생각 중...</thinking> 질문에 답변해 드릴게요."
                    ),
                    "이것은 두 번째 테스트 스트리밍 조각입니다.",
                ]
            )
        )

    with get_performance_monitor().track_operation(
        OperationType.PDF_LOADING, {"model": model_name}
    ):
        from core.custom_ollama import DeepThinkingChatOllama

        return DeepThinkingChatOllama(
            model=model_name,
            num_predict=OLLAMA_NUM_PREDICT,
            top_p=OLLAMA_TOP_P,
            num_ctx=OLLAMA_NUM_CTX,
            temperature=OLLAMA_TEMPERATURE,
            reasoning=OLLAMA_THINKING,
            base_url=OLLAMA_BASE_URL,
            keep_alive=OLLAMA_KEEP_ALIVE,
            timeout=OLLAMA_TIMEOUT,
        )
