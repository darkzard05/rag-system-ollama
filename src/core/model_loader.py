"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    ENABLE_OLLAMA_PRESSURE_FALLBACK,
    MODEL_CACHE_DIR,
    OLLAMA_BASE_URL,
    is_test_env,
)
from common.exceptions import EmbeddingModelError
from common.system_pressure import eviction_allowed
from core._loaders import (  # noqa: F401 — re-exports for backward compat
    _EMBED_PAYLOAD_CAP_CHARS,
    _EMBED_REQUEST_TIMEOUT_SECONDS,
    _EMBED_RETRY_BACKOFF_SECONDS,
    _EMBED_RETRY_MAX_ATTEMPTS,
    T,
    _aembed_retry_loop,
    _build_offloop,
    _fetch_available_models_cached,
    _get_psutil,
    _get_torch,
    _host_pressure_exceeded,
    _is_embed_transient_failure,
    _keep_alive_seconds,
    _memo_wrap,
    _ollama_backend_active,
    _PayloadCappedEmbeddings,
    _split_by_payload_size,
    _warmup_models,
    get_available_models,
    load_llm,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Module layout (model-loader split)
# ----------------------------------------------------------------------------
# 개별 모델 로더/보조 계층은 ``core._loaders`` 로 분리되었다 (load_embedding_model
# 의 페이로드 분할·재시도 보조, load_llm, get_available_models, _warmup_models,
# _PayloadCappedEmbeddings, _get_torch/_get_psutil, _host_pressure_exceeded/
# _ollama_backend_active 등). 위 import 는 분리 이전 import 경로
# (``core.model_loader.<symbol>``)에 대한 하위 호환 재수출이다.
# 본 파일에는 ModelManager 퍼사드와, 테스트가 ``core.model_loader`` 네임스페이스
# 를 패치(patch)하는 계약에 묶인 요소(_resolve_embedding_batch_size, 그리고
# load_embedding_model 내부의 _NoTruncateOllamaEmbeddings — logger/
# EMBEDDING_BATCH_SIZE 를 모듈 전역으로 참조)를 유지한다.
# ============================================================================


class ModelManager:
    """
    시스템 전체의 모델 인스턴스를 관리하는 중앙 클래스 (LRU 캐시 적용).
    UI와 API가 공동으로 사용하여 중복 로딩 및 VRAM 낭비를 방지합니다.
    [Modernized] asyncio 네이티브 동기화 도구 사용으로 전환됨.
    """

    # 비동기 락 (지연 로딩)
    _locks: dict[str, asyncio.Lock] = {}
    _inference_semaphore: asyncio.Semaphore | None = None
    _inference_semaphore_bound: int | None = None

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
    async def _check_memory_pressure(cls):
        """현재 VRAM/RAM 사용량을 확인하고 압박 여부를 반환합니다 (퇴출은 ModelPool 소관)."""
        # ENABLE_OLLAMA_PRESSURE_FALLBACK 는 모듈 레벨 import 를 사용하므로,
        # 테스트는 core.model_loader.ENABLE_OLLAMA_PRESSURE_FALLBACK 를 패치해
        # 동작을 격리할 수 있다 (runtime 재import 는 conftest 별칭으로 인해
        # 서로 다른 모듈 객체를 볼 수 있어 패치가 누수된다).

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
                    return True
            except Exception as e:
                logger.debug(f"VRAM 체크 실패 (무시): {e}")

        # 2. Ollama 압력 폴백 (torch.cuda 미사용 기본 배포)
        if (
            ENABLE_OLLAMA_PRESSURE_FALLBACK
            and _ollama_backend_active()
            and _host_pressure_exceeded()
        ):
            if not eviction_allowed("model_manager"):
                return False
            logger.warning(
                "[ModelManager] 호스트 RAM 압박 감지 (Ollama 폴백, >90%). "
                "자원 방출을 시작합니다."
            )
            return True

        # 3. 시스템 RAM 체크 (폴백)
        _psutil = _get_psutil()
        if _psutil:
            mem = _psutil.virtual_memory()
            if mem.percent > 95:
                logger.warning(
                    f"[ModelManager] 시스템 RAM 부족 ({mem.percent}%). 자원 방출을 시작합니다."
                )
                return True
        return False

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


def _resolve_embedding_batch_size(target_device: str) -> int:
    """HF 내부 ``encode_kwargs.batch_size`` 값을 config 엔진에서 해석한다.

    ``EMBEDDING_BATCH_SIZE``(config.yml의 ``embedding_batch_size``, 기본 "auto")
    를 그대로 소비한다: "auto" → device 기본값(cuda=32, 그 외=16), 양의 정수 →
    그대로, 숫자 문자열("8") → 파싱, 그 외(0/음수/bool/비숫자) → 기본 16 + 경고.
    """
    raw = EMBEDDING_BATCH_SIZE
    invalid = 16
    if isinstance(raw, bool) or not isinstance(raw, (int, str)):
        logger.warning(
            "[MODEL] [EMBED] EMBEDDING_BATCH_SIZE 유효하지 않음, 기본 16 사용: %r",
            raw,
        )
        return invalid
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized == "auto":
            return 32 if target_device == "cuda" else 16
        if normalized.isdigit() and int(normalized) >= 1:
            return int(normalized)
        logger.warning(
            "[MODEL] [EMBED] EMBEDDING_BATCH_SIZE 유효하지 않음, 기본 16 사용: %r",
            raw,
        )
        return invalid
    if raw >= 1:
        return raw
    logger.warning(
        "[MODEL] [EMBED] EMBEDDING_BATCH_SIZE 유효하지 않음, 기본 16 사용: %r",
        raw,
    )
    return invalid


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
    if is_test_env():
        from langchain_core.embeddings import FakeEmbeddings

        logger.info(f"[TEST] [MOCK] 가짜 임베딩 모델 로드됨 (모델명: {model_key})")
        return _memo_wrap(
            FakeEmbeddings(size=1536)
        )  # nomic-embed-text 등 주요 모델 크기에 맞춤

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
                """Ollama 임베딩 — `/api/embed` truncate=False 명시.

                [B11] 동기·비동기 공용 단일 시도 코어(``_embed_once``)를 기준으로
                재시도 루프를 두 갈래로 갖는다: 동기(``_embed_with_retry``)는
                기존대로 ``time.sleep`` 백오프, 비동기(``_aembed_with_retry``)는
                시도별 ``asyncio.wait_for`` 타임아웃 + ``await asyncio.sleep``.
                양쪽 모두 페이로드 크기 선분할을 적용한다.
                """

                truncate: bool = False

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    return self._embed_with_retry(texts)

                async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
                    return await self._aembed_with_retry(texts)

                async def aembed_query(self, text: str) -> list[float]:
                    return (await self._aembed_with_retry([text]))[0]

                def _embed_once(self, texts: list[str]) -> list[list[float]]:
                    """단일 시도: 페이로드 크기 분할 → 그룹별 1회 모델 호출."""
                    if not self._client:
                        msg = (
                            "Ollama client is not initialized. "
                            "Please ensure Ollama is running and the model is loaded."
                        )
                        raise ValueError(msg)
                    results: list[list[float]] = []
                    for group in _split_by_payload_size(texts):
                        results.extend(
                            self._client.embed(
                                self.model,
                                group,
                                truncate=self.truncate,
                                options=self._default_params,
                                keep_alive=self.keep_alive,
                            )["embeddings"]
                        )
                    return results

                def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
                    """동기 재시도 루프 (기존 시맨틱 유지, 분할 추가)."""
                    last_exc: Exception | None = None
                    for attempt in range(_EMBED_RETRY_MAX_ATTEMPTS):
                        try:
                            return self._embed_once(texts)
                        except Exception as exc:
                            last_exc = exc
                            if not _is_embed_transient_failure(exc):
                                raise
                            if attempt + 1 >= _EMBED_RETRY_MAX_ATTEMPTS:
                                break
                            wait = _EMBED_RETRY_BACKOFF_SECONDS[
                                min(attempt, len(_EMBED_RETRY_BACKOFF_SECONDS) - 1)
                            ]
                            logger.info(
                                "[MODEL] [EMBED] 임베딩 콜드스타트 감지, "
                                "재시도 %d/%d (대기 %ds)",
                                attempt + 2,
                                _EMBED_RETRY_MAX_ATTEMPTS,
                                wait,
                            )
                            time.sleep(wait)
                    assert last_exc is not None
                    raise last_exc

                async def _aembed_with_retry(
                    self, texts: list[str]
                ) -> list[list[float]]:
                    """비동기 재시도 루프 — 시도별 wait_for(1회 모델 왕복 상한)."""
                    return await _aembed_retry_loop(
                        lambda: asyncio.wait_for(
                            asyncio.to_thread(self._embed_once, texts),
                            timeout=_EMBED_REQUEST_TIMEOUT_SECONDS,
                        ),
                        "임베딩 콜드스타트 감지",
                    )

            logger.info(
                f"[MODEL] [LOAD] Ollama 임베딩 엔진 사용 | 모델: {clean_model_name}"
            )

            def _build_ollama() -> Embeddings:
                return _NoTruncateOllamaEmbeddings(
                    model=clean_model_name,
                    base_url=OLLAMA_BASE_URL,
                    keep_alive=_keep_alive_seconds(),
                )

            result = _build_offloop(_build_ollama)

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
            # [B11] config 엔진(EMBEDDING_BATCH_SIZE)에서 batch_size 해석
            batch_size = _resolve_embedding_batch_size(target_device)

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

            def _build_hf() -> Embeddings:
                # [B11] 페이로드 크기 선분할+비동기 타임아웃 래퍼로 감싼다.
                return _PayloadCappedEmbeddings(
                    HuggingFaceEmbeddings(
                        model_name=model_key,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs,
                        cache_folder=MODEL_CACHE_DIR,
                    )
                )

            result = _build_offloop(_build_hf)

            logger.info(
                f"[MODEL] [LOAD] HF 임베딩 모델 로드 성공 | 엔진: {display_device} (Backend: {backend})"
            )

        return _memo_wrap(result)

    except Exception as e:
        logger.error(f"임베딩 모델 로드 실패: {e}")
        raise EmbeddingModelError(model=model_key, reason=str(e)) from e


# ============================================================================
# memo 섹션 (Phase 3A: core.embedding_memo 에서 verbatim 이동, 의미 동결)
# ----------------------------------------------------------------------------
# 쿼리 임베딩 중복 계산 제거용 LRU+TTL+단일-flight 메모이징 래퍼.
# 동일한 쿼리 원문이 한 번의 RAG 질의 사이클에서 임베더로 여러 번 직렬
# 임베딩될 수 있는 문제를 제거한다. 단일-flight(동일 텍스트 동시 중복 계산
# 차단) + LRU 캐시 + TTL 만료 + 실패 폭포 전파 + 타임아웃 단일 승계를 동기
# ``embed_query`` 시그니처 그대로 제공한다. ``embed_documents``는 순수
# passthrough다(메모이제이션 대상 아님).
# 구 core.embedding_memo 경로는 pure re-export shim으로 유지한다.
# ============================================================================

# waiter가 계산자 완료를 기다리는 최대 시간(초). 초과 시 단일 승계(takeover)로
# 새 계산자가 되어 재계산한다. 0 이하의 값은 즉시 승계로 해석된다.
_WAIT_BOUND = 30.0

# LRU 캐시 기본 상한 및 TTL(초). 상한 초과 시 가장 오래된 항목부터 퇴출된다.
_DEFAULT_MAXSIZE = 1024
_DEFAULT_TTL = 300.0


def _current_wait_bound() -> float:
    """waiter 대기 상한의 호출-시점 판독 (shim 패치 가시성).

    ``tests/unit/test_embedding_memo.py`` 가 구 경로
    (``core.embedding_memo._WAIT_BOUND``)를 패치하므로, shim 네임스페이스에
    명시적 오버라이드가 있으면 그것을 우선한다. 없으면 정규 바인딩을 쓴다.
    (resource_pools→manager Live-forwarder와 같은 방향의 패치-심 계약.)
    """
    for modname in ("core.embedding_memo", "src.core.embedding_memo"):
        mod = sys.modules.get(modname)
        if mod is not None and "_WAIT_BOUND" in mod.__dict__:
            try:
                return float(mod.__dict__["_WAIT_BOUND"])
            except (TypeError, ValueError):
                break
    return _WAIT_BOUND


@dataclass
class _InFlight:
    """진행 중 계산 마커 + 결과/실패 전파 구조체.

    ``vec``/``error``는 단계 2에서 포착한 객체에서 직접 읽어 락 재획득 없이
    waiter에게 배달된다. ``event.set()``은 성공/실패 단계(3/4)에서만 호출된다.
    """

    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    vec: np.ndarray | None = None


class MemoizingEmbedding(Embeddings):
    """동기 임베더 데코레이터 — 쿼리 임베딩 중복 계산 제거.

    같은 텍스트에 대한 ``embed_query`` 호출이 TTL 창 안에서는 캐시를 재사용하고,
    동시 요청은 단일 계산자에게 병합(단일-flight)된다. 계산은 반드시 락 밖에서
    수행되고(락은 구조 접근에만 사용), waiter의 ``Event.wait``도 락 밖에서
    수행되어 교착/루프 역전이 없다.
    """

    def __init__(
        self,
        inner: Embeddings,
        *,
        maxsize: int = _DEFAULT_MAXSIZE,
        ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.RLock()
        # key(쿼리 원문) -> (monotonic 등록 시각, float32 벡터)
        self._cache: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()
        # key -> 진행 중 계산 마커(단일-flight)
        self._in_flight: dict[str, _InFlight] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "calls": 0,
            "in_flight": 0,
        }
        _memo_instances.add(self)

    @property
    def model(self) -> Any:
        """내부 임베더의 모델 식별자를 위임합니다.

        데코레이터 래퍼는 모델명을 노출하지 않으면 자신이 감싼 임베더의
        ``model``/``model_name`` 이 없으면 ``embedding_memo`` 캐시 키의 기준이
        ``default_model`` 로 폴백되어, 서로 다른 임베더(가짜/실제)가 같은 캐시
        키 공간을 공유하게 된다. (FAISS 차원 불일치 사고의 근본 원인)
        """
        return getattr(self._inner, "model", None)

    @property
    def model_name(self) -> Any:
        """내부 임베더의 모델명을 위임합니다 (``model`` 미보유 임베더 대비)."""
        return getattr(self._inner, "model_name", None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """메모이제이션 없이 내부 임베더로 순수 위임한다."""
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """단일-flight 메모이징 쿼리 임베딩.

        단계 1(캐시) / 단계 2(단일-flight 판정)는 락 임계구간에서, 실제 계산과
        waiter 대기는 락 밖에서 수행된다. 타임아웃 시 항등 비교로 단일 승계자가
        되어 재계산하고, 나머지는 후임 체인을 따라 다시 대기한다.
        """
        with self._lock:
            entry = self._cache.get(text)
            if entry is not None:
                timestamp, vec = entry
                if time.monotonic() - timestamp <= self._ttl:
                    # LRU: 접근 시 최근 사용 위치로 이동
                    self._cache.move_to_end(text)
                    self._stats["hits"] += 1
                    return vec.tolist()
                del self._cache[text]
            self._stats["misses"] += 1
            existing = self._in_flight.get(text)
            if existing is not None:
                holder = existing
                is_registrant = False
            else:
                holder = _InFlight()
                self._in_flight[text] = holder
                self._stats["in_flight"] += 1
                is_registrant = True
        if is_registrant:
            # 이 스레드가 계산자 → 락 밖 compute
            return self._compute(text, holder)
        # 이 스레드는 waiter → 락 밖 wait + 타임아웃 시 단일 승계
        return self._await_query(text, holder)

    def _compute(self, text: str, in_flight: _InFlight) -> list[float]:
        """계산자 경로(단계 3/4): 락 미보유 상태에서만 호출된다."""
        with self._lock:
            self._stats["calls"] += 1
        try:
            vec = self._inner.embed_query(text)
        except BaseException as exc:
            # 단계 4: 실패 → error 기록 + 이벤트 set + 항등 가드 pop(캐시 미저장)
            with self._lock:
                in_flight.error = exc
                if self._in_flight.get(text) is in_flight:
                    self._in_flight.pop(text, None)
                    self._stats["in_flight"] -= 1
                in_flight.event.set()
            raise
        # 단계 3: 성공 → vec 기록 + 캐시 저장 + 항등 가드 pop + 이벤트 set
        arr = np.asarray(vec, dtype=np.float32)
        with self._lock:
            in_flight.vec = arr
            self._store_locked(text, arr)
            if self._in_flight.get(text) is in_flight:
                self._in_flight.pop(text, None)
                self._stats["in_flight"] -= 1
            in_flight.event.set()
        return in_flight.vec.tolist()

    def _await_query(self, text: str, captured: _InFlight) -> list[float]:
        """waiter 경로(단계 2): event.wait → set이면 직접 배달, timeout이면 승계."""
        while True:
            # _WAIT_BOUND 직접 참조가 아닌 호출-시점 판독 (shim 패치 가시성).
            if captured.event.wait(timeout=_current_wait_bound()):
                break
            # timeout — 단일 승계 시도 (항등 비교)
            successor: _InFlight | None = None
            with self._lock:
                current = self._in_flight.get(text)
                if current is captured:
                    # 승계 승자: stale 엔트리 교체 후 이 스레드가 새 계산자
                    self._in_flight.pop(text, None)
                    successor = _InFlight()
                    self._in_flight[text] = successor
                elif current is not None:
                    # 다른 waiter가 이미 승계 → 후임 체인으로 갱신 후 재대기
                    captured = current
            if successor is not None:
                return self._compute(text, successor)
        # 이벤트 set — 포착 객체에서 직접 배달(락 재획득 없음)
        if captured.error is not None:
            raise captured.error
        assert captured.vec is not None
        return captured.vec.tolist()

    def _store_locked(self, text: str, vec: np.ndarray) -> None:
        """락 보유 상태에서만 호출. TTL 만료 스캔 후 put, maxsize 초과 시 퇴출."""
        now = time.monotonic()
        for key in [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl]:
            del self._cache[key]
        self._cache[text] = (now, vec)
        self._cache.move_to_end(text)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    def clear(self) -> None:
        """캐시·단일-flight·통계 전부 초기화.

        진행 중인 계산자는 단계 3/4의 항등 가드 pop으로 후임자/신규 엔트리를
        건드리지 않는다.
        """
        with self._lock:
            self._cache.clear()
            self._in_flight.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "calls": 0,
                "in_flight": 0,
            }

    def stats(self) -> dict[str, int]:
        """현재 통계 사본을 반환한다."""
        with self._lock:
            return dict(self._stats)


# weakref 레지스트리: 생성된 모든 래퍼를 약참조로 추적한다(테스트 격리용).
_memo_instances: weakref.WeakSet[MemoizingEmbedding] = weakref.WeakSet()


def clear_memo_instances() -> None:
    """등록된 모든 래퍼 인스턴스의 상태를 초기화한다.

    pytest 프로세스에서는 모델 풀 공유로 동일 래퍼가 테스트 간 잔존하므로,
    ``tests/unit/conftest.py``의 autouse 픽스처가 각 테스트 후 이 함수를 호출해
    테스트 간 메모이 오염을 차단한다.
    """
    for instance in list(_memo_instances):
        instance.clear()
