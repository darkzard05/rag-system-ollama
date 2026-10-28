"""
LLM 및 임베딩 모델 로더 구현 계층.

model_loader 모듈에서 분리된 개별 모델 로더/보조 구현이다. load_embedding_model
에서 사용하던 페이로드 분할·재시도·배치 크기 해석 보조 요소, load_llm,
get_available_models, 프리웜(_warmup_models) 등이 위치한다. 기존 import 경로
(``core.model_loader.<symbol>``)는 model_loader 모듈의 재수출로 유지된다.

[분리 경계] model_loader(부모) → _loaders(하위) 단방향. _loaders 는 model_loader
를 모듈 레벨에서 import 하지 않는다(_warmup_models 는 런타임 지연 import 로
ModelManager 를 참조한다).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import re
import socket
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = object  # type: ignore


from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    OLLAMA_TEMPERATURE,
    OLLAMA_THINKING,
    OLLAMA_TIMEOUT,
    OLLAMA_TOP_P,
    is_test_env,
)
from services.monitoring.performance_monitor import (
    OperationType,
    get_performance_monitor,
)

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
_TEST_EMBED_RETRY_BACKOFF_SECONDS = (0.001, 0.002, 0.003)


def _get_embed_retry_backoff() -> tuple[float, ...]:
    """임베딩 재시도 백오프 정책을 반환합니다.

    테스트 환경에서는 1ms 단위로 축소해 재시도 검증이 실제 수 초를 대기하지
    않도록 한다. 프로덕션에서는 기존 (1, 3, 9)초 정책을 그대로 유지한다.
    """
    if is_test_env():
        return _TEST_EMBED_RETRY_BACKOFF_SECONDS
    return _EMBED_RETRY_BACKOFF_SECONDS


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


logger = logging.getLogger("core.model_loader")


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
        backoff_seconds = _get_embed_retry_backoff()
        wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
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


class _PayloadCappedEmbeddings(Embeddings):
    """HF/ONNX 임베딩 — 페이로드 크기 기준 선분할 + 비동기 타임아웃 래퍼.

    ``HuggingFaceEmbeddings``(=SentenceTransformer)의 내부 배칭은 개수 기준
    (``batch_size``)이라 컨텍스트 초과(payload 크기)를 막지 못한다. 호출자
    경계에서 ``_EMBED_PAYLOAD_CAP_CHARS`` 단위로 선분할해 과잉 입력을 차단하고,
    비동기 경로는 그룹별 ``asyncio.wait_for``로 hang을 중단한다.
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


class _NoTruncateOllamaEmbeddings(OllamaEmbeddings):  # type: ignore
    """Ollama 임베딩 — `/api/embed` truncate=False 명시."""

    truncate: bool = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed_with_retry(texts)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._aembed_with_retry(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return (await self._aembed_with_retry([text]))[0]

    @staticmethod
    def _is_context_length_error(err: Any) -> bool:
        if getattr(err, "status_code", None) != 400:
            return False
        return "context length" in str(err).lower()

    def _embed_one_group(self, client: Any, group: list[str]) -> list[list[float]]:
        try:
            resp = client.embed(
                self.model,
                group,
                truncate=self.truncate,
                options=self._default_params,
                keep_alive=self.keep_alive,
            )
            return list(resp["embeddings"])
        except Exception as err:
            import ollama

            if isinstance(err, ollama.ResponseError):
                if self.truncate or not self._is_context_length_error(err):
                    raise
                logger.warning(
                    "[MODEL] [EMBED] 컨텍스트 초과로 truncate=True 폴백 적용"
                )
                resp = client.embed(
                    self.model,
                    group,
                    truncate=True,
                    options=self._default_params,
                    keep_alive=self.keep_alive,
                )
                return list(resp["embeddings"])
            raise

    def _embed_once(self, texts: list[str]) -> list[list[float]]:
        if not getattr(self, "_client", None):
            raise ValueError("Ollama client is not initialized.")
        import concurrent.futures

        client = self._client
        groups = _split_by_payload_size(texts)
        if len(groups) <= 1:
            results: list[list[float]] = []
            for group in groups:
                results.extend(self._embed_one_group(client, group))
            return results

        ordered: list[list[float]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(4, len(groups))
        ) as executor:
            for embeddings in executor.map(
                lambda g: self._embed_one_group(client, g), groups
            ):
                ordered.extend(embeddings)
        return ordered

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
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
                backoff_seconds = _get_embed_retry_backoff()
                wait = backoff_seconds[min(attempt, len(backoff_seconds) - 1)]
                import core.model_loader

                core.model_loader.logger.info(
                    "[MODEL] [EMBED] 임베딩 콜드스타트 감지, 재시도 %d/%d (대기 %ds)",
                    attempt + 2,
                    _EMBED_RETRY_MAX_ATTEMPTS,
                    wait,
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    async def _aembed_with_retry(self, texts: list[str]) -> list[list[float]]:
        return await _aembed_retry_loop(
            lambda: asyncio.wait_for(
                asyncio.to_thread(self._embed_once, texts),
                timeout=_EMBED_REQUEST_TIMEOUT_SECONDS,
            ),
            "임베딩 콜드스타트 감지",
        )


def _quick_probe_port(url_str: str, timeout_seconds: float = 0.05) -> bool:
    """초고속 소켓 연결 검사로 Ollama 데몬 포트 개방 여부 확인."""
    try:
        parsed = urlparse(url_str)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 11434)
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except (OSError, TimeoutError):
        return False


def _is_ollama_server_up() -> bool:
    """캐시 상태 또는 고속 소켓 검사를 통해 Ollama 가용성을 판정 (Fail-Closed)."""
    global _models_cache
    if _models_cache is not None:
        models = _models_cache[1]
        return bool(models) and (MSG_ERROR_OLLAMA_NOT_RUNNING not in models)
    return _quick_probe_port(OLLAMA_BASE_URL, timeout_seconds=0.05)


async def _warmup_models() -> None:
    """시스템 추론 세마포어를 보호하며 자원 누수 없이 실행되는 안정적 프리워밍."""
    from common.config import DEFAULT_OLLAMA_MODEL
    from core.model_loader import ModelManager
    from core.resource_manager import get_resource_manager
    from core.session import SessionManager

    if not _is_ollama_server_up():
        logger.debug("[WARMUP] Ollama 미연결 상태 — 프리워밍 건너뜀")
        return

    coordinator = get_resource_manager()

    if SessionManager.get("is_generating_answer") or SessionManager.get(
        "is_building_rag"
    ):
        logger.info("[WARMUP] 사용자 작업 우선 — 임베더 워밍업 양보")
        return

    # 1. 임베더 웜업
    try:
        async with coordinator.use_embedder(model_name=DEFAULT_EMBEDDING_MODEL):
            embedder = await ModelManager.get_embedder(None)
            if hasattr(embedder, "aembed_query"):
                await embedder.aembed_query("warmup")
            else:
                await asyncio.to_thread(embedder.embed_query, "warmup")
    except Exception as e:
        logger.debug(f"[WARMUP] 임베더 프리웜 실패 (무시): {e}")

    if SessionManager.get("is_generating_answer") or SessionManager.get(
        "is_building_rag"
    ):
        logger.info("[WARMUP] 사용자 작업 우선 — LLM 워밍업 양보")
        return

    # 2. LLM 웜업 (단일 1-토큰 invoke로 안전 완결)
    try:
        async with (
            coordinator.inference_session(timeout=2.0),
            coordinator.use_llm(model_name=DEFAULT_OLLAMA_MODEL),
        ):
            llm = await ModelManager.get_llm(DEFAULT_OLLAMA_MODEL)
            warm_llm = (
                llm.bind(options={"num_predict": 1, "think": False, "temperature": 0.0})
                if hasattr(llm, "bind")
                else llm
            )
            await warm_llm.ainvoke("warmup")
        logger.info("[WARMUP] LLM+임베더 프리웜 정상 완료")
    except (asyncio.TimeoutError, TimeoutError) as e:
        logger.debug(f"[WARMUP] LLM 프리웜 건너뜀 (타임아웃 또는 양보): {e}")
    except Exception as e:
        logger.debug(f"[WARMUP] LLM 프리웜 실패: {e}")


def _fetch_available_models() -> list[str]:
    """Ollama 모델 목록을 실제로 조회합니다. (캐시 없음)"""
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


# 프로세스 전역 TTL 캐시. UI(Streamlit)뿐 아니라 API 등 비-Streamlit 호출자도
# Ollama ``list()`` HTTP 호출(최대 5s)을 반복하지 않도록 한다. 빈 결과(조회
# 실패)는 짧은 TTL로만 캐시해, Ollama가 늦게 기동해도 빠르게 복구되면서
# 동시에 매 호출마다 5s 타임아웃을 물지 않도록 한다.
_MODELS_CACHE_TTL_SECONDS = 300.0
_MODELS_FAILURE_TTL_SECONDS = 15.0
_models_cache: tuple[float, list[str]] | None = None
_models_cache_lock = threading.Lock()
_models_fetch_schedule_lock = threading.Lock()
_models_fetching_event = threading.Event()


def _models_cache_ttl(models: list[str]) -> float:
    return _MODELS_CACHE_TTL_SECONDS if models else _MODELS_FAILURE_TTL_SECONDS


def _bg_fetch_models_task() -> None:
    """백그라운드에서 모델 목록을 조회해 전역 TTL 캐시를 채웁니다."""
    global _models_cache

    try:
        models = _fetch_available_models()
        _models_cache = (time.monotonic(), models)
    finally:
        _models_fetching_event.clear()


def _schedule_background_models_fetch() -> None:
    """진행 중인 조회가 없을 때만 단일 백그라운드 조회를 예약합니다."""
    with _models_fetch_schedule_lock:
        if _models_fetching_event.is_set():
            return
        _models_fetching_event.set()
        threading.Thread(
            target=_bg_fetch_models_task,
            name="AsyncModelListFetcher",
            daemon=True,
        ).start()


def _ollama_client_is_stubbed() -> bool:
    """테스트가 ``ollama.Client`` 를 mock으로 교체했는지 판별합니다.

    명시적으로 클라이언트를 스텁한 테스트(캐시/TTL 계약 검증)는 테스트 환경
    fast-path를 우회해 실제 코드 경로를 검증해야 한다. mock이 아니면 로컬
    Ollama 데몬 부재 시 5초 소켓 타임아웃을 물지 않도록 fast-path를 탄다.
    """
    try:
        import ollama
    except ImportError:
        return False
    return hasattr(getattr(ollama, "Client", None), "mock_calls")


def _fetch_available_models_cached(
    non_blocking_fallback: bool = False,
) -> list[str]:
    """Ollama 모델 목록을 TTL 캐시로 반환합니다.

    ``non_blocking_fallback=True`` (UI 메인 스레드 경로)에서는 캐시 미스 시
    ``ollama.Client.list()`` 를 직접 호출하지 않고 백그라운드 조회만 예약한 뒤
    이전 캐시(또는 기본 모델)를 즉시 반환합니다. Streamlit 메인 렌더 스레드가
    최대 5s의 동기 소켓 I/O에 블로킹되는 것을 막습니다.

    ``non_blocking_fallback=False`` (CLI/테스트/명시적 새로고침)에서는 기존처럼
    동기 조회로 최신 값을 반환합니다.
    """
    global _models_cache

    # [최적화] 단위 테스트 환경에서 실제 Ollama 데몬이 없을 때 5초 소켓
    # 타임아웃에 블로킹되지 않도록 즉시 기본 모델 목록을 반환한다. 클라이언트를
    # 명시적으로 스텁한 테스트는 실제 경로를 검증할 수 있도록 통과시킨다.
    if is_test_env() and not _ollama_client_is_stubbed():
        from common.config import DEFAULT_OLLAMA_MODEL

        return [DEFAULT_OLLAMA_MODEL]

    cached = _models_cache
    now = time.monotonic()
    if cached is not None and (now - cached[0] < _models_cache_ttl(cached[1])):
        return list(cached[1])

    if non_blocking_fallback:
        _schedule_background_models_fetch()
        if cached is not None:
            return list(cached[1])
        from common.config import DEFAULT_OLLAMA_MODEL

        # fallback 반환 시 방어적으로 복사본 전달 (호출자의 mutate 방지)
        return [DEFAULT_OLLAMA_MODEL]

    with _models_cache_lock:
        # 락 획득 후 더블 체크 (다른 스레드가 먼저 갱신했을 수 있음)
        cached = _models_cache
        now = time.monotonic()
        if cached is not None and (now - cached[0] < _models_cache_ttl(cached[1])):
            return list(cached[1])

        models = _fetch_available_models()
        _models_cache = (time.monotonic(), models)
        return list(models)


def clear_available_models_cache() -> None:
    """모델 목록 캐시를 무효화합니다. (사이드바 "모델 새로고침" 경로)"""
    global _models_cache

    with _models_cache_lock:
        _models_cache = None


def _keep_alive_seconds() -> int:
    """OLLAMA_KEEP_ALIVE(기본 "5m")를 초 단위 정수로 변환.

    OllamaEmbeddings.keep_alive는 ``int | None`` 타입이므로 문자열을 그대로
    넘기면 pydantic 검증 오류가 발생한다. 파싱 실패 시 300초(5분)로 폴백한다.
    """
    match = re.fullmatch(r"(\d+)m", OLLAMA_KEEP_ALIVE.strip())
    if match:
        return int(match.group(1)) * 60
    return 300


def get_available_models(non_blocking_fallback: bool = False) -> list[str]:
    models = _fetch_available_models_cached(non_blocking_fallback=non_blocking_fallback)
    from common.config import DEFAULT_OLLAMA_MODEL

    return models or [DEFAULT_OLLAMA_MODEL, MSG_ERROR_OLLAMA_NOT_RUNNING]


def load_llm(model_name: str) -> Any:
    # [최적화] CI/유닛 테스트 환경에서는 Ollama 서버 없이도 동작하도록 가짜 LLM 반환
    if is_test_env():
        from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
        from langchain_core.messages import AIMessage

        logger.info(f"[TEST] [MOCK] 가짜 LLM 로드됨 (모델명: {model_name})")
        return GenericFakeChatModel(
            messages=itertools.cycle(
                [
                    AIMessage(
                        content='{"final_answer": "안녕하세요! RAG 시스템 테스트 응답입니다. <thinking>테스트 생각 중...</thinking> 질문에 답변해 드릴게요.", "citations": []}'
                    ),
                    AIMessage(
                        content='{"final_answer": "이것은 두 번째 테스트 스트리밍 조각입니다.", "citations": []}'
                    ),
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
