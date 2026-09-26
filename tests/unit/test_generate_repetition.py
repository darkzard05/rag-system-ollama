"""Raw-side answer-repetition regression tests (mocked LLM, hermetic).

Verdict source: reports/measure_repetition_20260926_152916.md — the LLM
echoes a repeated claim (~4x) because duplicated chunks reach the prompt
untouched (``format_context`` appends every doc; ``_apply_ctx_guard`` trims
by token budget only, which a small prompt never trips).

These tests fail pre-fix and pass post-fix:

- identical normalized-text chunks are dropped before prompt build
  (``format_context`` + ``_apply_ctx_guard`` assembly level);
- the system prompt actually sent to the (mocked) LLM contains the claim
  once even when retrieval hands over 4x duplicated docs;
- the synthesis prompt carries a stable anti-echo marker.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from common.config import PROMPT_TEMPLATES_CONFIG
from core.graph.graph_builder import _apply_ctx_guard, format_context, generate
from core.model_loader import ModelManager

_CLAIM = "The reserve ratio is 12 percent."
_NUM_CTX = 8192
_NUM_PREDICT = 2048


def _make_echo_docs(n: int = 5) -> list[Document]:
    """Same claim once per doc on distinct pages (mirrors the Task-1 page set)."""
    pages = [3, 10, 11, 14, 19]
    docs = []
    for i in range(n):
        # Trailing-whitespace variant: normalization must still catch it.
        pad = " " if i % 2 else ""
        docs.append(
            Document(
                page_content=f"{_CLAIM}{pad}",
                metadata={
                    "rerank_score": 0.9 - i * 0.01,
                    "page": pages[i % len(pages)],
                    "current_section": "일반 본문",
                },
            )
        )
    return docs


def test_format_context_drops_identical_chunks() -> None:
    """Duplicated chunks must not survive into the prompt string."""
    ctx = format_context(_make_echo_docs(5))

    assert ctx.count(_CLAIM) == 1


def test_apply_ctx_guard_drops_identical_chunks_pre_prompt() -> None:
    """Identical chunks are dropped before the prompt is built (budget intact)."""
    with (
        patch("core.graph._generate.OLLAMA_NUM_CTX", _NUM_CTX),
        patch("core.graph._generate.OLLAMA_NUM_PREDICT", _NUM_PREDICT),
        patch("core.graph._generate.count_tokens_rough", return_value=100),
    ):
        trimmed, context, _removed = _apply_ctx_guard(_make_echo_docs(5), "query")

    assert len(trimmed) == 1
    assert context.count(_CLAIM) == 1


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_generate_sends_deduped_context_to_llm() -> None:
    """Mocked-LLM 4x-echo scenario: the prompt sent must hold the claim once."""
    sent_messages: list = []
    payload = json.dumps(
        {
            "final_answer": "답변 본문입니다.",
            "citations": [],
            "confidence": 0.9,
        }
    )

    async def mock_astream(messages: list, config: object = None) -> object:
        sent_messages.append(messages)
        yield AIMessageChunk(
            content=payload, response_metadata={"prompt_eval_count": 5}
        )

    mock_json_llm = MagicMock()
    mock_json_llm.astream = mock_astream
    mock_llm = MagicMock()
    mock_llm.bind.return_value = mock_json_llm
    mock_llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    state = {"input": "질문입니다", "relevant_docs": _make_echo_docs(4)}
    config = {"configurable": {"llm": mock_llm}}

    with (
        patch("core.graph._generate.OLLAMA_NUM_CTX", _NUM_CTX),
        patch("core.graph._generate.OLLAMA_NUM_PREDICT", _NUM_PREDICT),
        patch("core.graph._generate.count_tokens_rough", return_value=100),
        patch("core.graph._glue.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        result = await generate(state, config, writer=MagicMock())

    assert len(sent_messages) == 1
    sys_prompt = sent_messages[0][0].content
    assert sys_prompt.count(_CLAIM) == 1
    # Deduped doc list is propagated so downstream sees the same single chunk.
    assert result["performance"]["relevant_docs_count"] == 1


def test_synthesis_prompt_has_anti_echo_instruction() -> None:
    """The v2.0 synthesis prompt must carry the stable anti-echo marker."""
    template = PROMPT_TEMPLATES_CONFIG.get("structured_output", "")

    assert "[No Repetition]" in template
