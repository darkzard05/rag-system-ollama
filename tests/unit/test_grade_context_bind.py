"""Grade and generate must request the same Ollama context window.

ChatOllama does not merge instance fields when ``options`` is supplied. A
grade bind without ``num_ctx`` asks Ollama for its default context, so the
following generate forces a model reload before the first answer token.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import core.graph._grade_verify as gv
from common.config import OLLAMA_NUM_CTX


class _DummySession:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> bool:
        return False


class _FakeLLM:
    def __init__(self) -> None:
        self.bind_kwargs: dict[str, Any] = {}

    def bind(self, **kwargs: Any) -> _FakeLLM:
        self.bind_kwargs = kwargs
        return self


@pytest.mark.asyncio
async def test_grade_bind_requests_generate_context_window() -> None:
    llm = _FakeLLM()
    docs = [Document(page_content="body", metadata={"rerank_score": 0.1})]
    state = {
        "input": "low confidence grade ctx probe",
        "relevant_docs": docs,
        "retry_count": 0,
        "short_query": False,
    }
    config = {"configurable": {"llm": llm, "messages": []}}

    async def _fake_invoke(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            content='{"action":"generate","is_relevant":true,"reason":"ok"}'
        )

    async def _no_event(*args: object, **kwargs: object) -> None:
        return None

    with (
        patch.object(
            gv.ModelManager, "inference_session", return_value=_DummySession()
        ),
        patch.object(gv, "adispatch_custom_event", _no_event),
        patch.object(gv, "_safe_invoke", _fake_invoke),
        patch.object(gv, "_start_speculative_generate", lambda *a, **k: None),
        patch.object(gv, "_enter_stage", lambda *a, **k: MagicMock()),
        patch.object(gv.SessionManager, "add_status_log", lambda *a, **k: None),
        patch.object(gv.SessionManager, "set", lambda *a, **k: None),
        patch.object(gv.SessionManager, "get", lambda *a, **k: None),
    ):
        result = await gv.grade_documents(state, config, writer=MagicMock())

    assert result["route"] == "generate"
    assert llm.bind_kwargs.get("options", {}).get("num_ctx") == OLLAMA_NUM_CTX
