"""
Todo 9 검증 (R1a-03): CTX 가드 트림·재정렬 결과가 노드 반환 dict("relevant_docs")로
전달되어 체크포인트 최종 상태와 하이드레이션/인용 소스가 정합해야 합니다.

기존 결함: generate가 `state["relevant_docs"] = ranked`로 입력 dict를 in-place 변이만 하고
노드 반환 dict에 포함하지 않아, LangGraph 리듀서(overwrite)로 최종 상태에 반영되지 않았다
(실증 probe: x=999 → x=0 소실).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from core.graph_builder import generate
from core.model_loader import ModelManager

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


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


async def _run_generate(mock_llm: MagicMock, docs, fake_count) -> tuple:
    sent_messages: list = []

    async def mock_astream(messages, config=None):  # noqa: ARG001
        sent_messages.append(messages)
        yield AIMessageChunk(
            content="최종 답변", response_metadata={"prompt_eval_count": 5}
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
async def test_generate_returns_trimmed_docs_in_state_channel():
    """예산 초과로 3문서 → 2문서 트림 시 반환 dict에 최종(트림·내림차순) docs가 포함됩니다."""

    def fake_count(text: str) -> int:
        n_docs = sum(m in text for m in _DOC_MARKERS)
        return _BUDGET + 5000 if n_docs >= 3 else _BUDGET - 5000

    mock_llm = MagicMock()
    result, _sent, state = await _run_generate(mock_llm, _make_docs(), fake_count)

    # R1a-03: 노드 반환 dict가 최종 상태를 전달한다 (in-place 변이 미반영 문제 해소)
    assert "relevant_docs" in result
    assert [d.metadata["rerank_score"] for d in result["relevant_docs"]] == [0.9, 0.5]
    assert result["performance"]["relevant_docs_count"] == 2


@pytest.mark.asyncio
async def test_generate_no_trim_does_not_overwrite_state_channel():
    """예산 이내이면 트림이 없으므로 relevant_docs를 반환하지 않아 기존 상태를 보존합니다."""

    def fake_count(text: str) -> int:  # noqa: ARG001
        return _BUDGET - 1000

    mock_llm = MagicMock()
    result, _sent, _state = await _run_generate(mock_llm, _make_docs(), fake_count)

    assert "relevant_docs" not in result
    assert result["performance"]["relevant_docs_count"] == 3
