"""
T14 검증: warmup + concurrency 보장.

세 가지를 입증한다:
  (a) WARMUP_NONFATAL — ModelManager._warmup_models() 가 모델 로드/throwaway
      토큰 경로를 모의(fake)로 돌려도 예외 없이 완료된다. 실제 Ollama 호출 없이
      CI 에서 deterministic 하게 green 이어야 함.
  (b) CONCURRENCY_CLAMP — effective bound 계산이 VRAM 압력(host_pressure_exceeded)
      시 MAX_CONCURRENT_INFERENCE>1 을 1 로 강등하고, 기본값(==1)에서는
      host_pressure_exceeded 를 호출조차 하지 않는 short-circuit 을 가진다.
  (c) RELEASE_ON_EXC — ModelManager.inference_session() 이 내부 블록 예외 시에도
      추론 세마포어를 반드시 해제(release)한다.

주의: effective bound 는 별도 헬퍼가 아니라 ResourceCoordinator.acquire_inference_lock
(str208 resource_manager.py:407) 및 ModelManager._get_semaphore(model_loader.py:144)
내부에 인라인으로 계산된다. 여기서는 실제 구현 경로를 통해 관측한다.
"""

from __future__ import annotations

from asyncio import Semaphore
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import core.model_loader as model_loader
import core.resource_manager as resource_manager_mod
from core.model_loader import ModelManager, _warmup_models
from core.resource_manager import get_resource_manager

# ---------------------------------------------------------------------------
# (a) WARMUP_NONFATAL — 모의 모델로 _warmup_models() 가 예외 없이 완료
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """get_embedder() 가 반환하는 가짜 임베더 — aembed_query 모의."""

    def __init__(self) -> None:
        self.aembed_query = AsyncMock(return_value=[0.1])


class _FakeLLM:
    """get_llm() 이 반환하는 가짜 LLM — astream() 은 즉시 1토큰 후 종료."""

    def __init__(self) -> None:
        async def _gen():
            yield "warmup-token"

        self.astream = MagicMock(side_effect=lambda _text: _gen())


@pytest.mark.asyncio
async def test_warmup_runs_without_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ollama 없이 get_llm/get_embedder 를 페이크로 대체해도 프리웜이
    (LLM 로드 → 임베더 로드 → throwaway aembed_query → minimal astream) 경로를
    예외 없이 통과한다."""
    fake_embedder = _FakeEmbedder()
    fake_llm = _FakeLLM()

    get_embedder_mock = AsyncMock(return_value=fake_embedder)
    get_llm_mock = AsyncMock(return_value=fake_llm)
    monkeypatch.setattr(ModelManager, "get_embedder", get_embedder_mock)
    monkeypatch.setattr(ModelManager, "get_llm", get_llm_mock)

    # 예외가 전파되지 않아야 함 (호출부에서 비치명적으로 감싸는 전제).
    await _warmup_models()

    # 프리웜이 실제로 두 경로를 소비했는지도 확인.
    get_embedder_mock.assert_awaited_once_with(None)
    get_llm_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_warmup_invokes_throwaway_token_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """프리웜이 임베더 aembed_query 와 LLM astream(break) throwaway 경로를
    실제로 탐 — 모의 객체가 호출되었음을 assert."""
    fake_embedder = _FakeEmbedder()
    fake_llm = _FakeLLM()

    get_embedder_mock = AsyncMock(return_value=fake_embedder)
    get_llm_mock = AsyncMock(return_value=fake_llm)
    monkeypatch.setattr(ModelManager, "get_embedder", get_embedder_mock)
    monkeypatch.setattr(ModelManager, "get_llm", get_llm_mock)

    await _warmup_models()

    fake_embedder.aembed_query.assert_awaited_once_with("warmup")
    fake_llm.astream.assert_called_once_with("warmup")


# ---------------------------------------------------------------------------
# (b) CONCURRENCY_CLAMP — effective bound 계산 + VRAM 압력 강등 + short-circuit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_clamp_pressure_forces_bound_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """VRAM 압력(host_pressure_exceeded=True) + MAX_CONCURRENT_INFERENCE=4 인데도
    effective bound 가 1 로 강등되는지 — 이게 명시적 QA 실패 요구사항."""
    coordinator = get_resource_manager()
    # 상태 격리를 위해 세마포어 강제 재생성.
    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None

    monkeypatch.setattr(resource_manager_mod, "host_pressure_exceeded", lambda: True)
    monkeypatch.setattr(resource_manager_mod, "MAX_CONCURRENT_INFERENCE", 4)

    await coordinator.acquire_inference_lock(timeout=0)
    try:
        # 압력 시 >1 동시 추론은 1 로 강등.
        assert coordinator._inference_semaphore_bound == 1
        assert coordinator.inference_semaphore is not None
    finally:
        coordinator.release_inference_lock()
        # 다른 테스트 격리를 위해 세마포어 해제.
        coordinator._inference_semaphore = None
        coordinator._inference_semaphore_bound = None


@pytest.mark.asyncio
async def test_concurrency_default_one_shortcircuits_pressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """기본값 MAX_CONCURRENT_INFERENCE==1 에서는 host_pressure_exceeded() 를
    호출조차 하지 않는 short-circuit — 압력과 무관하게 bound==1."""
    coordinator = get_resource_manager()
    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None

    pressure_mock: MagicMock = MagicMock(return_value=True)
    monkeypatch.setattr(resource_manager_mod, "host_pressure_exceeded", pressure_mock)
    monkeypatch.setattr(resource_manager_mod, "MAX_CONCURRENT_INFERENCE", 1)

    await coordinator.acquire_inference_lock(timeout=0)
    try:
        assert coordinator._inference_semaphore_bound == 1
        # `> 1` 단락 평가로 인해 압력 감지 함수 자체가 호출되지 않아야 함.
        pressure_mock.assert_not_called()
    finally:
        coordinator.release_inference_lock()
        coordinator._inference_semaphore = None
        coordinator._inference_semaphore_bound = None


# ---------------------------------------------------------------------------
# (c) RELEASE_ON_EXC — inference_session 이 예외 시에도 세마포어 해제
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inference_session_releases_on_exception() -> None:
    """async with ModelManager.inference_session(): raise RuntimeError("boom")
    이후에도 추론 세마포어가 해제(획득 전 value 복원)되는지."""
    coordinator = get_resource_manager()
    # 기본 bound(==1) 세마포어를 확실히 생성하기 위해 리셋.
    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None

    with pytest.raises(RuntimeError, match="boom"):
        async with ModelManager.inference_session():
            raise RuntimeError("boom")

    sem: Semaphore | None = coordinator.inference_semaphore
    assert sem is not None
    # 예외 후에도 1 획득분이 해제되어 세마포어가 full bound 로 복원되었는지.
    assert sem._value == 1

    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None


@pytest.mark.asyncio
async def test_inference_session_releases_on_normal_exit() -> None:
    """정상 종료 시에도 세마포어가 해제되는지 (대조군)."""
    coordinator = get_resource_manager()
    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None

    async with ModelManager.inference_session():
        pass

    sem: Semaphore | None = coordinator.inference_semaphore
    assert sem is not None
    assert sem._value == 1

    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None


# ---------------------------------------------------------------------------
# (a) 보강 — warmup 이 모델 로드 실패를 삼키지 않고 전파(propagate) 하는지.
#     비치명적 래핑은 호출부(api_server/main) 책임이므로, 헬퍼 자체는 반드시
#     예외를 재발생시켜야 한다.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_warmup_propagates_llm_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_llm() 이 실패하면 _warmup_models 가 예외를 잡지 않고 그대로 전파."""
    get_embedder_mock = AsyncMock(return_value=_FakeEmbedder())
    get_llm_mock = AsyncMock(side_effect=RuntimeError("llm load boom"))
    monkeypatch.setattr(ModelManager, "get_embedder", get_embedder_mock)
    monkeypatch.setattr(ModelManager, "get_llm", get_llm_mock)

    with pytest.raises(RuntimeError, match="llm load boom"):
        await _warmup_models()


# ---------------------------------------------------------------------------
# (b) 보강 — 명시적 2-case: MAX=3 + 압력 없음 → bound==3 (강등 없음).
#     위 clamp 테스트(압력 시 bound==1)와 짝을 이뤄 "honors config + pressure
#     clamp" 를 결정적으로 입증한다.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrency_no_pressure_keeps_configured_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MAX_CONCURRENT_INFERENCE=3 + host_pressure_exceeded()==False 면
    effective bound 가 3 (강등 없음) 이 되는지."""
    coordinator = get_resource_manager()
    coordinator._inference_semaphore = None
    coordinator._inference_semaphore_bound = None

    monkeypatch.setattr(resource_manager_mod, "host_pressure_exceeded", lambda: False)
    monkeypatch.setattr(resource_manager_mod, "MAX_CONCURRENT_INFERENCE", 3)

    await coordinator.acquire_inference_lock(timeout=0)
    try:
        # 압력 없음 → 설정된 동시성 그대로 적용.
        assert coordinator._inference_semaphore_bound == 3
    finally:
        coordinator.release_inference_lock()
        coordinator._inference_semaphore = None
        coordinator._inference_semaphore_bound = None


# ---------------------------------------------------------------------------
# (c) 보강 — AsyncMock 세마포어 주입: 본문 예외 시에도 release 가 정확히 1회.
#     주입은 공개 API(ResourceManager.inference_semaphore 세터)를 통해 수행.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inference_session_releases_mock_semaphore_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """본문에서 예외가 나도 주입된 세마포어의 release 가 정확히 1회 호출된다."""
    from core.resource_manager import ResourceManager

    rc = ResourceManager()
    mock_sem = MagicMock()
    mock_sem.acquire = AsyncMock()  # acquire_inference_lock 내에서 await 됨
    mock_sem.release = MagicMock()  # release_inference_lock 은 동기 호출
    # 공개 세터로 주입(주입 세마포어는 bound=None 이므로 재생성되지 않음).
    rc.inference_semaphore = mock_sem

    with patch.object(resource_manager_mod, "get_resource_manager", return_value=rc):
        with pytest.raises(RuntimeError, match="boom"):
            async with ModelManager.inference_session():
                raise RuntimeError("boom")

    # finally 경로로 예외와 무관하게 release 가 정확히 1회.
    mock_sem.release.assert_called_once()
    # 진입 시 acquire 도 1회.
    mock_sem.acquire.assert_awaited_once()
