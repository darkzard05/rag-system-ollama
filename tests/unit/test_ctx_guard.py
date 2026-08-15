"""
Todo 5 검증: 컨텍스트 토큰 가드 + 모델 프리로드 + config 파서.

- (a) generate 노드: num_ctx 85% 초과 시 rerank_score 낮은 문서부터 제거, 최소 2문서 유지
- (b) 모델 프리로드: get_llm 예외 삼킴 + 1회성 스케줄링
- (c) config.yml: num_ctx=8192 / ollama_num_predict=2048 로드 검증
"""

import asyncio
import importlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

import common.config as config_module
import core.pipeline_builder as pb
from core.graph_builder import generate
from core.model_loader import ModelManager
from core.pipeline_builder import PipelineBuilder

_NUM_CTX = 8192
_NUM_PREDICT = 2048
_BUDGET = int((_NUM_CTX - _NUM_PREDICT) * 0.85)

_DOC_MARKERS = ("doc_A_low", "doc_B_mid", "doc_C_high")


def _make_docs() -> list[Document]:
    return [
        Document(
            page_content="doc_A_low rerank content",
            metadata={"rerank_score": 0.1, "page": 1},
        ),
        Document(
            page_content="doc_B_mid rerank content",
            metadata={"rerank_score": 0.5, "page": 2},
        ),
        Document(
            page_content="doc_C_high rerank content",
            metadata={"rerank_score": 0.9, "page": 3},
        ),
    ]


def _reset_preload_state() -> None:
    pb._preload_lock = None
    pb._preload_loop = None
    pb._preload_scheduled = False


@pytest.fixture(autouse=True)
def _reset_preload():
    _reset_preload_state()
    yield
    # pytest는 테스트마다 이벤트 루프를 교체하므로, 파일 종료 후에는 1회성 플래그를
    # 유지해 이후 테스트 파일의 build()에서 preload 태스크가 abandon되지 않게 합니다.
    # (운영 단일 루프에서는 발생하지 않는 테스트 환경 전용 보호)
    pb._preload_scheduled = True


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


async def _run_generate(mock_llm: MagicMock, docs: list[Document], fake_count) -> tuple:
    sent_messages: list = []

    # AnswerStructure 스키마에 맞는 유효한 JSON 응답
    valid_json_response = {
        "reasoning": "테스트 추론 과정",
        "final_answer": "최종 답변",
        "citations": [
            {
                "doc_id": 0,
                "text_span": "테스트",
                "section": "테스트",
                "page": 1,
                "score": 0.9,
            }
        ],
        "confidence": 0.9,
    }
    import json

    json_str = json.dumps(valid_json_response, ensure_ascii=False)

    async def mock_astream(messages, config=None):
        sent_messages.append(messages)
        yield AIMessageChunk(
            content=json_str, response_metadata={"prompt_eval_count": 5}
        )

    mock_llm.astream = mock_astream
    mock_llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    state = {"input": "질문입니다", "relevant_docs": docs, "is_cached": False}
    config = {"configurable": {"llm": mock_llm}}

    with (
        patch("core.graph_builder.OLLAMA_NUM_CTX", _NUM_CTX),
        patch("core.graph_builder.OLLAMA_NUM_PREDICT", _NUM_PREDICT),
        patch("core.graph_builder.count_tokens_rough", side_effect=fake_count),
        patch("core.graph_builder.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        result = await generate(state, config, writer=MagicMock())
    return result, sent_messages, state


@pytest.mark.asyncio
async def test_generate_trims_low_rerank_docs_when_over_budget():
    """3문서 중 2문서만 예산 이내일 때 rerank_score가 낮은 문서부터 제거되는지 검증합니다."""

    def fake_count(text: str) -> int:
        n_docs = sum(m in text for m in _DOC_MARKERS)
        return _BUDGET + 5000 if n_docs >= 3 else _BUDGET - 5000

    mock_llm = MagicMock()
    result, sent_messages, _state = await _run_generate(
        mock_llm, _make_docs(), fake_count
    )

    assert result["performance"]["relevant_docs_count"] == 2
    # R1a-03: 최종 상태는 노드 반환 dict로 반영된다 (in-place state 변이 제거)
    assert [d.metadata["rerank_score"] for d in result["relevant_docs"]] == [0.9, 0.5]
    human_content = sent_messages[0][0].content
    assert "doc_A_low" not in human_content
    assert "doc_B_mid" in human_content
    assert "doc_C_high" in human_content


@pytest.mark.asyncio
async def test_generate_keeps_minimum_two_docs_even_when_over_budget():
    """항상 예산 초과여도 최소 2문서는 유지하고 2문서로 진행하는지 검증합니다."""

    def fake_count(text: str) -> int:
        return _BUDGET + 5000

    mock_llm = MagicMock()
    result, sent_messages, _state = await _run_generate(
        mock_llm, _make_docs(), fake_count
    )

    assert result["performance"]["relevant_docs_count"] == 2
    # R1a-03: 최종 상태는 노드 반환 dict로 반영된다 (in-place state 변이 제거)
    assert [d.metadata["rerank_score"] for d in result["relevant_docs"]] == [0.9, 0.5]
    human_content = sent_messages[0][0].content
    assert "doc_A_low" not in human_content
    assert "doc_B_mid" in human_content
    assert "doc_C_high" in human_content


@pytest.mark.asyncio
async def test_generate_preserves_descending_rerank_order_after_trim():
    """trim 후 남은 문서가 rerank_score 내림차순(최상위 우선)으로 컨텍스트/상태에 배치되는지 검증합니다."""
    docs = [
        Document(
            page_content="doc_A_low rerank content",
            metadata={"rerank_score": 0.1, "page": 1},
        ),
        Document(
            page_content="doc_B_mid rerank content",
            metadata={"rerank_score": 0.5, "page": 2},
        ),
        Document(
            page_content="doc_C_high rerank content",
            metadata={"rerank_score": 0.9, "page": 3},
        ),
        Document(
            page_content="doc_D_top rerank content",
            metadata={"rerank_score": 0.95, "page": 4},
        ),
        Document(
            page_content="doc_E_topmost rerank content",
            metadata={"rerank_score": 0.99, "page": 5},
        ),
    ]
    markers = ("doc_A_low", "doc_B_mid", "doc_C_high", "doc_D_top", "doc_E_topmost")

    def fake_count(text: str) -> int:
        n_docs = sum(m in text for m in markers)
        return _BUDGET + 5000 if n_docs >= 4 else _BUDGET - 5000

    mock_llm = MagicMock()
    result, sent_messages, _state = await _run_generate(mock_llm, docs, fake_count)

    assert result["performance"]["relevant_docs_count"] == 3
    # R1a-03: 최종 상태는 노드 반환 dict로 반영된다 (in-place state 변이 제거)
    assert [d.metadata["rerank_score"] for d in result["relevant_docs"]] == [
        0.99,
        0.95,
        0.9,
    ]
    human_content = sent_messages[0][0].content
    positions = [
        human_content.index("doc_E_topmost"),
        human_content.index("doc_D_top"),
        human_content.index("doc_C_high"),
    ]
    assert positions == sorted(positions)
    assert "doc_A_low" not in human_content
    assert "doc_B_mid" not in human_content


@pytest.mark.asyncio
async def test_preload_model_swallows_get_llm_exception(caplog):
    """프리로드 중 get_llm 예외가 태스크 밖으로 누출되지 않고 로그만 남기는지 검증합니다."""
    with patch(
        "core.model_loader.ModelManager.get_llm",
        new=AsyncMock(side_effect=RuntimeError("ollama down")),
    ):
        await pb._preload_model()

    assert "프리로드 실패" in caplog.text


@pytest.mark.asyncio
async def test_preload_scheduled_only_once_per_process(monkeypatch):
    """_register_and_finalize가 두 번 호출되어도 프리로드는 한 번만 스케줄링되는지 검증합니다."""
    # CI(IS_CI_TEST=true)에서는 _schedule_model_preload의 테스트 환경 가드가 조기 반환하므로
    # 스케줄링 로직 자체를 검증하려면 가드를 비활성화해야 합니다.
    monkeypatch.delenv("IS_CI_TEST", raising=False)
    monkeypatch.delenv("IS_UNIT_TEST", raising=False)
    resource_manager = MagicMock()
    resource_manager.register_retrievers = AsyncMock(return_value=None)

    with (
        patch(
            "core.pipeline_builder.get_resource_manager", return_value=resource_manager
        ),
        patch(
            "core.pipeline_builder.build_graph",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch("core.pipeline_builder._preload_model", new=AsyncMock()) as mock_preload,
    ):
        builder = PipelineBuilder(session_id="test-ctx-guard")
        await builder._register_and_finalize("hash1", MagicMock(), MagicMock())
        await builder._register_and_finalize("hash2", MagicMock(), MagicMock())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    mock_preload.assert_awaited_once()


def test_config_loads_updated_num_ctx_and_num_predict(monkeypatch):
    """config.yml의 num_ctx=8192 / ollama_num_predict=2048이 로드되는지 검증합니다."""
    monkeypatch.delenv("OLLAMA_NUM_CTX", raising=False)
    monkeypatch.delenv("OLLAMA_NUM_PREDICT", raising=False)
    reloaded = importlib.reload(config_module)
    assert reloaded.OLLAMA_NUM_CTX == 8192
    assert reloaded.OLLAMA_NUM_PREDICT == 2048
