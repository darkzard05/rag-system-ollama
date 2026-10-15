"""Repro: 동일 docs에 대한 쿼리당 컨텍스트 3중 재렌더 결함.

결함 경로:
  generate._apply_ctx_guard(docs) → 렌더+해시 1회, context는 상태에 저장되지 않음
  verify_answer → format_context(docs) 재호출 (2차 렌더+재해시)
  RAGSystem.aquery → format_context(docs) 재호출 (3차 렌더+재해시)

수정 계약:
  generate가 context를 GraphState에 저장하면 verify/aquery는 재사용하고
  format_context를 재호출하지 않는다.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessageChunk

from core.graph._generate import _apply_ctx_guard, _docs_fingerprint, format_context
from core.model_loader import ModelManager


def _make_docs(n: int = 10) -> list[Document]:
    return [
        Document(
            page_content=f"Triple-render repro doc {i} with distinct body {i * 7}.",
            metadata={
                "rerank_score": float(i),
                "page": i + 1,
                "current_section": "테스트",
            },
        )
        for i in range(n)
    ]


class _NullWriter:
    def __call__(self, data: object) -> None:
        pass


@pytest.mark.asyncio
async def test_verify_reuses_generate_context(monkeypatch) -> None:
    """verify_answer가 상태의 context를 재사용하면 format_context를 호출하지 않는다."""
    import core.graph._grade_verify as gver

    docs = _make_docs(10)
    trimmed, generate_context, _removed = _apply_ctx_guard(docs, "질문입니다")
    assert generate_context  # 전제: 가드가 비어 있지 않은 컨텍스트를 만든다
    # [PoC-2] 지문 계약: context는 트림된 docs에서 렌더되므로 상태의
    # relevant_docs도 트림 결과와 일치해야 재사용된다.
    state_docs = trimmed if _removed else docs

    state = {
        "input": "질문입니다",
        "response": "답변입니다",
        "relevant_docs": state_docs,
        "context": generate_context,
        "context_fingerprint": _docs_fingerprint(state_docs),
        "regeneration_count": 0,
    }
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=SimpleNamespace(content='{"faithful": true, "issues": []}')
    )
    config = {"configurable": {"llm": llm}}

    monkeypatch.setattr(gver, "VERIFICATION_ENABLED", True)
    monkeypatch.setattr(gver, "_should_verify", lambda *a, **k: True)

    format_calls = 0
    real_format = gver.format_context

    def counting_format(docs_arg: list[Document]) -> str:
        nonlocal format_calls
        format_calls += 1
        return real_format(docs_arg)

    monkeypatch.setattr(gver, "format_context", counting_format)

    result = await gver.verify_answer(state, config, writer=_NullWriter())

    assert result["verification_route"] == "end"
    assert format_calls == 0, (
        f"triple-render defect: verify가 format_context를 {format_calls}회 재호출함"
    )


def test_resolve_cached_context_prefers_state(monkeypatch) -> None:
    """aquery/verify 공용 헬퍼: 상태 context가 있으면 재사용, 없으면 폴백한다."""
    import core.graph._generate as ggen

    docs = _make_docs(5)
    trimmed, generate_context, removed = _apply_ctx_guard(docs, "질문입니다")
    state_docs = trimmed if removed else docs

    format_calls = 0
    real_format = ggen.format_context

    def counting_format(docs_arg: list[Document]) -> str:
        nonlocal format_calls
        format_calls += 1
        return real_format(docs_arg)

    monkeypatch.setattr(ggen, "format_context", counting_format)

    # 상태에 context+지문이 일치하면 재사용하고 format_context를 호출하지 않는다.
    resolved = ggen.resolve_cached_context(
        {
            "context": generate_context,
            "context_fingerprint": ggen._docs_fingerprint(state_docs),
        },
        state_docs,
    )
    assert resolved == generate_context
    assert format_calls == 0

    # 상태에 context가 없으면 None을 반환해 호출자가 format_context로 폴백한다.
    assert ggen.resolve_cached_context({}, docs) is None
    assert ggen.resolve_cached_context({"context": ""}, docs) is None
    # [PoC-2] 지문 불일치(stale)도 None으로 거부한다.
    stale = ggen.resolve_cached_context(
        {"context": generate_context, "context_fingerprint": "stale"},
        state_docs,
    )
    assert stale is None


def _patch_inference_session() -> MagicMock:
    mock_session = MagicMock()
    mock_session.return_value.__aenter__ = AsyncMock()
    mock_session.return_value.__aexit__ = AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_generate_overwrites_stale_context_on_regenerate() -> None:
    """재생성 시 generate가 가드를 다시 계산해 상태의 오래된 context를 덮어쓴다."""
    from core.graph.graph_builder import generate

    new_docs = _make_docs(3)
    state = {
        "input": "질문입니다",
        "relevant_docs": new_docs,
        "is_cached": False,
        "context": "STALE-CONTEXT",
    }
    mock_llm = MagicMock()
    mock_llm._convert_chunk_to_thought_and_content = lambda chunk: (chunk.content, "")

    async def mock_astream(messages, config=None):  # noqa: ARG001
        yield AIMessageChunk(
            content='{"final_answer": "답변", "reasoning": "추론"}',
            response_metadata={"prompt_eval_count": 5},
        )

    mock_llm.bind.return_value.astream = mock_astream
    config = {"configurable": {"llm": mock_llm}}

    with (
        patch("core.graph._generate.OLLAMA_NUM_CTX", 8192),
        patch("core.graph._generate.OLLAMA_NUM_PREDICT", 2048),
        patch("core.graph._generate.count_tokens_rough", side_effect=lambda t: 1),
        patch("core.graph._glue.adispatch_custom_event", new=AsyncMock()),
        patch.object(ModelManager, "inference_session", _patch_inference_session()),
    ):
        result = await generate(state, config, writer=MagicMock())

    expected = format_context(new_docs)
    assert result["context"] == expected
    assert result["context"] != "STALE-CONTEXT"
