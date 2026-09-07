"""
LLM 및 임베딩 모델 로딩을 담당하는 파일.
Optimized: 타임아웃 강화 및 로컬 Ollama 통신 안정성 확보.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from langchain_core.embeddings import Embeddings

T = TypeVar("T")


def _build_offloop(builder: Callable[[], T]) -> T:
    """무거운 동기 모델 생성자를 안전하게 실행합니다.

    ``load_embedding_model``은 동기 함수이므로, 비동기 경로에서는
    ``ResourceCoordinator.get_or_build``가 이미 ``asyncio.to_thread``로 이
    함수를 워커 스레드에서 실행합니다(이때 실행 중인 루프가 없음). Streamlit
    시작점 등 동기 진입부에서는 인라인으로 실행됩니다. 본 헬퍼는 두 경로 모두
    동일하게 동작하도록 모델 생성부를 한 곳으로 모읍니다.
    """
    return builder()


# Ollama 임베딩 콜드스타트 레이스 대응 재시도 파라미터.
# Ollama Go 서버는 임베딩 백엔드(llama-server)가 아직 스폰 중/재시작 중일 때
# 내부 연결 실패(연결 거부)를 HTTP 400 + 연결 거부 문자열로 반환한다. 이는
# 일시적 상태이므로 멱등한 임베딩 호출에 한해 지수 백오프로 재시도한다.
_EMBED_RETRY_MAX_ATTEMPTS = 3
_EMBED_RETRY_BACKOFF_SECONDS = (1, 3, 9)

# [B11] 비동기 임베딩 1회 모델 왕복(그룹) 타임아웃(초). 사이클 단위가 아닌
# 시도(attempt) 단위로 적용한다. 사이클 단위로 걸면 3회 재시도+백오프(1+3+9s
# 수면)가 한 번의 wait_for에 몰려 60s 같은 타이트한 값은 콜드스타트를 못견디고,
# 느슨한 값은 1회 호출 hang을 사실상 무제한 방치한다. 시도 내부에서 to_thread
# 호출을 개별 상한(120s — Ollama 콜드스타트/백엔드 스폰 허용)하면 매 시도가
# 유계(bounded)가 되어 총 지연 ≤ 3×(120s + 백오프)로 확정된다.
_EMBED_REQUEST_TIMEOUT_SECONDS = 120.0

# [B11/§NB5] 페이로드 크기 기준 상한(문자 수). 개수 기반 배칭으로는 컨텍스트
# 초과를 막을 수 없으므로(truncate=False에서 오버사이즈 입력은 에러 표면화),
# 한 번에 모델로 보내는 텍스트의 총 char 수를 여기로 캡한다. CJK는 대략
# 문자당 1토큰이므로 12_000자 ≈ 8~9k 토큰으로 안전하다. 단일 텍스트가 상한을
# 넘으면 문자열을 쪼개지 않고 단독 그룹으로 보낸다(모델이 판단).
_EMBED_PAYLOAD_CAP_CHARS = 12_000


def _is_embed_transient_failure(exc: BaseException) -> bool:
    """연결 거부형 오류 본문만 재시도 대상으로 한정한다.

    ``asyncio.TimeoutError``는 유형(type)으로도 재시도 대상으로 인정한다 —
    비동기 경로의 wait_for가 1회 호출 hang을 중단시켰을 때(콜드스타트일 수
    있음) 재시도해야 하기 때문이다. 문자열 "timeout"은 판별하지 않는다.
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    text = str(exc)
    return any(
        marker in text
        for marker in ("actively refused", "connection refused", "connectex")
    )


def _split_by_payload_size(
    texts: list[str], max_chars: int = _EMBED_PAYLOAD_CAP_CHARS
) -> list[list[str]]:
    """페이로드 크기 기준 욕심(greedy) 선분할.

    ``sum(len(t)) <= max_chars`` 가 유지되도록 문자 누적으로 그룹을 만들고,
    원본 순서를 보존한다. 단일 텍스트가 상한을 넘으면 문자열을 중간에서
    쪼개지 않고 단독 그룹으로 반환한다(모델 컨텍스트 판단에 위임).
    """
    groups: list[list[str]] = []
    current: list[str] = []
    current_chars = 0
    for text in texts:
        text_chars = len(text)
        if current and current_chars + text_chars > max_chars:
            groups.append(current)
            current = []
            current_chars = 0
        current.append(text)
        current_chars += text_chars
    if current:
        groups.append(current)
    return groups


def _memo_wrap(embedder: Embeddings) -> Embeddings:
    """쿼리 임베딩 메모이제이션 래퍼. env 토글(기본 on), 생성 시 1회 평가.

    FAISS 검색·시맨틱 리랭커·세맨틱 쿼리 캐시 등 모든 소비 경로가 풀을 통해
    동일 래퍼 인스턴스를 받도록 반환 지점에서 래핑한다. off면 순수 위임.
    """
    if os.getenv("MEMOIZE_EMBEDDING_QUERY", "1") != "1":
        return embedder
    from core.embedding_memo import MemoizingEmbedding  # 순환 방지 lazy import

    return MemoizingEmbedding(embedder)


from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE,
    ENABLE_OLLAMA_PRESSURE_FALLBACK,
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
    eviction_allowed,
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

            # 서브모듈을 명시적으로 로드해야 테스트의
            # patch("torch.cuda.is_available") 가 일관되게 적용된다
            # (lazy 로딩 시 패치가 누수됨).
            import torch.cuda  # noqa: F401

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


async def _aembed_retry_loop(
    attempt_fn: Callable[[], Awaitable[list[list[float]]]],
    log_prefix: str,
) -> list[list[float]]:
    """비동기 임베딩 재시도 루프 공용 코어.

    매 시도는 ``attempt_fn``(=wait_for로 감싼 to_thread 단일 모델 호출)이며,
    ``asyncio.TimeoutError``와 ``_is_embed_transient_failure`` 대상 오류만
    재시도한다. 비일시 오류는 즉시 전파, 소진 시 마지막 예외를 재발생한다.
    """
    last_exc: Exception | None = None
    for attempt in range(_EMBED_RETRY_MAX_ATTEMPTS):
        try:
            return await attempt_fn()
        except asyncio.TimeoutError as exc:
            last_exc = exc
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
            "[MODEL] [EMBED] %s, 재시도 %d/%d (대기 %ds)",
            log_prefix,
            attempt + 2,
            _EMBED_RETRY_MAX_ATTEMPTS,
            wait,
        )
        await asyncio.sleep(wait)
    assert last_exc is not None
    raise last_exc


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


class _PayloadCappedEmbeddings(Embeddings):
    """HF/ONNX 임베딩 — 페이로드 크기 기준 선분할 + 비동기 타임아웃 래퍼.

    ``HuggingFaceEmbeddings``(=SentenceTransformer)의 내부 배칭은 개수 기준
    (``batch_size``)이라 컨텍스트 초과(payload 크기)를 막지 못한다. 호출자
    경계에서 ``_EMBED_PAYLOAD_CAP_CHARS`` 단위로 선분할해 과잉 입력을 차단하고,
    비동기 경로는 그룹별 ``asyncio.wait_for``로 hang을 1회 모델 왕복당 상한시킨다.
    동기 경로는 분할만 적용(기존 동기 시맨틱 유지).
    """

    def __init__(self, delegate: Embeddings) -> None:
        self._delegate = delegate

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_groups(_split_by_payload_size(texts))

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        groups = _split_by_payload_size(texts)
        return await _aembed_retry_loop(
            lambda: self._aembed_groups(groups),
            "HF 임베딩 일시 오류 감지",
        )

    async def aembed_query(self, text: str) -> list[float]:
        return (await self.aembed_documents([text]))[0]

    def _embed_groups(self, groups: list[list[str]]) -> list[list[float]]:
        results: list[list[float]] = []
        for group in groups:
            results.extend(self._delegate.embed_documents(group))
        return results

    async def _aembed_groups(self, groups: list[list[str]]) -> list[list[float]]:
        results: list[list[float]] = []
        for group in groups:
            results.extend(
                await asyncio.wait_for(
                    asyncio.to_thread(self._delegate.embed_documents, group),
                    timeout=_EMBED_REQUEST_TIMEOUT_SECONDS,
                )
            )
        return results


def _host_pressure_exceeded() -> bool:
    """동적 참조로 호스트 RAM 압력을 확인합니다.

    ``sys.modules`` 에서 모듈을 조회해 호출하므로, ``common.system_pressure`` 와
    ``src.common.system_pressure`` 별칭이 다른 객체로 매핑된 환경(conftest)에서도
    테스트 패치가 일관되게 적용됩니다 (로컬 import 는 패치 무효화).
    """
    import common.system_pressure as sp

    return sp.host_pressure_exceeded()


def _ollama_backend_active() -> bool:
    """동적 참조로 Ollama 백엔드 여부를 확인합니다.

    ``sys.modules`` 에서 모듈을 조회해 호출하므로, ``common.system_pressure`` 와
    ``src.common.system_pressure`` 별칭이 다른 객체로 매핑된 환경(conftest)에서도
    테스트 패치가 일관되게 적용됩니다 (로컬 import 는 패치 무효화).
    """
    import common.system_pressure as sp

    return sp.ollama_backend_active()


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


async def _warmup_models() -> None:
    """[WARMUP] 시작 시 LLM+임베더를 1회 프리웜하여 첫 쿼리 TTFT를 제거한다.

    - LLM / 임베더 각각 한 번씩만 로드(캐시 히트 보장).
    - 임베더는 ``None`` 전달로 설정 기본 모델 사용(앱 정상 로드 경로와 동일).
    - LLM은 ``keep_alive=OLLAMA_KEEP_ALIVE``(로더 기본값)로 빌드되어 즉시
      축출되지 않는다.
    - 호출부에서 비치명적으로 감싸야 하므로 여기선 예외를 잡지 않는다.
    - 모델 로드/쿼리 경로는 건드리지 않고 프리웜 전용 throwaway 토큰만 소비.
    """
    from common.config import DEFAULT_OLLAMA_MODEL
    from core.resource_manager import get_resource_manager

    coordinator = get_resource_manager()

    # Throwaway 토큰: 임베더 1회, LLM minimal 스트리밍 → 즉시 중단.
    async with coordinator.use_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
        embedder = await ModelManager.get_embedder(None)
        await embedder.aembed_query("warmup")

    async with coordinator.use_llm(model_name=DEFAULT_OLLAMA_MODEL):
        llm = await ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
        async for _ in llm.astream("warmup"):
            break

    logger.info("[WARMUP] LLM+임베더 프리웜 완료")


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
