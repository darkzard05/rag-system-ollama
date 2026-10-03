"""
P2 검증: _preload_model의 LLM·FlashRank 실질 워밍 추론.

프리로드 step3(get_llm)가 단순 클라이언트 생성에 그치면 첫 쿼리가 Ollama 콜드
모델 로드(~20s)를 지불하므로, 실질 워밍 추론(invoke("warmup"))이 정확히 1회
발생하는지 검증합니다.

step2(get_flashranker)는 래퍼 획득만으로는 ONNX InferenceSession이 뜨지 않아
첫 쿼리가 리랭킹 콜드 스타트(측정 rerank_ms=4744.8)를 지불합니다. 실제로
스코어링이 1회 발생하는지, 그리고 그 실패가 LLM 워밍을 막지 않는지 검증합니다.

헤르메틱: get_embedder/get_flashranker/get_llm 3개를 모두 스텁하고 flashrank는
가짜 모듈로 주입 — 실 Ollama/모델/네트워크 호출 없이 CI(Ollama 부재)에서도
green이어야 합니다.
"""

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.pipeline_builder as pb
from core.model_loader import ModelManager


class _RecordingLLM:
    """invoke 호출을 기록하는 동기 fake (프리로드는 to_thread로 실행)."""

    def __init__(self) -> None:
        self.invoke_calls: list[tuple[str, dict]] = []

    def invoke(self, prompt: str, **kwargs) -> str:
        self.invoke_calls.append((prompt, kwargs))
        return "warmup ok"


class _FakeEmbedder:
    """embed_query가 즉시 리스트를 반환하는 동기 fake (to_thread로 실행)."""

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2]


class _RecordingReranker:
    """rerank 호출을 기록하는 동기 fake (워밍은 to_thread로 실행)."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.rerank_calls: list[tuple[str, list[dict]]] = []
        self._error = error

    def rerank(self, request) -> list[dict]:
        self.rerank_calls.append((request.query, request.passages))
        if self._error is not None:
            raise self._error
        return [{"id": 0, "score": 0.0}]


def _install_fake_flashrank(monkeypatch) -> None:
    """flashrank 미설치 환경에서도 RerankRequest를 쓸 수 있도록 가짜 모듈을 주입합니다."""

    class _Request:
        def __init__(self, query: str, passages: list[dict]) -> None:
            self.query = query
            self.passages = passages

    module = types.ModuleType("flashrank")
    module.__dict__["RerankRequest"] = _Request
    monkeypatch.setitem(sys.modules, "flashrank", module)


def _patch_model_manager(monkeypatch, llm_fake, reranker=None) -> None:
    """step 1(임베딩)/step 2(FlashRank)/step 3(LLM)을 모두 스텁합니다."""
    monkeypatch.setattr(
        ModelManager,
        "get_embedder",
        AsyncMock(return_value=_FakeEmbedder()),
    )
    monkeypatch.setattr(
        ModelManager,
        "get_flashranker",
        AsyncMock(return_value=reranker if reranker is not None else MagicMock()),
    )
    monkeypatch.setattr(ModelManager, "get_llm", AsyncMock(return_value=llm_fake))


@pytest.mark.asyncio
async def test_preload_warmup_invokes_llm_exactly_once_with_warmup(monkeypatch):
    """프리로드 step3에서 LLM.invoke가 정확히 1회, "warmup" 인자로 호출되는지 검증합니다."""
    recording_llm = _RecordingLLM()
    _patch_model_manager(monkeypatch, recording_llm)

    await pb._preload_model()

    assert len(recording_llm.invoke_calls) == 1
    prompt, _kwargs = recording_llm.invoke_calls[0]
    assert prompt == "warmup"


@pytest.mark.asyncio
async def test_preload_warmup_swallows_get_llm_failure(monkeypatch):
    """get_llm이 예외를 던져도 _preload_model이 재발생 없이 조용히 종료되는지 검증합니다."""
    _patch_model_manager(monkeypatch, llm_fake=MagicMock())
    monkeypatch.setattr(
        ModelManager,
        "get_llm",
        AsyncMock(side_effect=RuntimeError("ollama down")),
    )

    # 재발생하면 테스트가 실패합니다.
    await pb._preload_model()


@pytest.mark.asyncio
async def test_preload_warmup_scores_flashranker_exactly_once(monkeypatch):
    """프리로드 step2가 FlashRank ONNX 세션을 실제 스코어링 1회로 확정하는지 검증합니다.

    get_flashranker()만으로는 파이썬 래퍼만 얻고 ONNX InferenceSession은 뜨지
    않으므로, 첫 쿼리가 리랭킹 콜드 스타트(측정 rerank_ms=4744.8)를 지불합니다.
    빈 passages는 즉시 반환되어 세션을 확정하지 못하므로 1건 이상을 넘겨야 합니다.
    """
    _install_fake_flashrank(monkeypatch)
    recording_ranker = _RecordingReranker()
    _patch_model_manager(monkeypatch, llm_fake=MagicMock(), reranker=recording_ranker)

    await pb._preload_model()

    assert len(recording_ranker.rerank_calls) == 1
    query, passages = recording_ranker.rerank_calls[0]
    assert query == "warmup"
    assert len(passages) >= 1


@pytest.mark.asyncio
async def test_preload_flashrank_failure_still_warms_llm(monkeypatch):
    """FlashRank 워밍이 실패해도 step3 LLM 워밍까지 진행되어야 합니다.

    FlashRank 워밍은 실제 스코어링을 수행하므로 이제 예외를 던질 수 있습니다.
    이것이 바깥 try/except로만 처리되면 LLM 워밍이 건너뛰어져 첫 쿼리 지연이
    재발하므로, step2가 자체 try/except로 격리되었는지 검증합니다.
    """
    _install_fake_flashrank(monkeypatch)
    recording_llm = _RecordingLLM()
    failing_ranker = _RecordingReranker(error=RuntimeError("onnx session failed"))
    _patch_model_manager(monkeypatch, llm_fake=recording_llm, reranker=failing_ranker)

    await pb._preload_model()

    assert len(failing_ranker.rerank_calls) == 1
    assert len(recording_llm.invoke_calls) == 1
