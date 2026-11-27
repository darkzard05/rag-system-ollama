"""Citations must survive the LangGraph state boundary on cache hits."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import END, START, StateGraph

from api.schemas import GraphState
from core.graph._generate import generate

_CITATIONS = [{"doc_id": "abc123", "text_span": "CM3 is a causal LM", "page": 3}]


def _config() -> dict:
    return {
        "configurable": {
            "session_id": "graph-cite",
            "thread_id": "graph-cite",
            "llm": MagicMock(),
        }
    }


def test_graph_state_has_citations_channel() -> None:
    assert "citations" in GraphState.__annotations__


def test_langgraph_keeps_declared_citations_channel() -> None:
    def fake_node(_state: GraphState) -> dict:
        return {"response": "answer", "citations": _CITATIONS}

    workflow = StateGraph(GraphState)
    workflow.add_node("fake", fake_node)
    workflow.add_edge(START, "fake")
    workflow.add_edge("fake", END)

    result = workflow.compile().invoke(
        {
            "input": "q",
            "intent": "general",
            "route": "generate",
            "search_queries": [],
            "relevant_docs": [],
            "response": None,
            "thought": None,
            "context": None,
            "context_fingerprint": None,
            "performance": None,
            "search_weights": None,
            "query_embedding": None,
            "is_cached": True,
            "cached_response": "answer",
            "cached_citations": _CITATIONS,
            "short_query": False,
            "retry_count": 0,
            "verification_route": None,
            "verification_issues": None,
            "regeneration_count": 0,
        }
    )

    assert result["citations"] == _CITATIONS


@pytest.mark.asyncio
async def test_cache_hit_generate_returns_citations() -> None:
    state = {
        "input": "q",
        "intent": "general",
        "route": "generate",
        "search_queries": [],
        "relevant_docs": [],
        "response": None,
        "thought": None,
        "is_cached": True,
        "cached_response": "cached answer",
        "cached_citations": _CITATIONS,
        "retry_count": 0,
    }
    with patch("core.graph._graph_core.adispatch_custom_event"):
        result = await generate(state, _config(), writer=MagicMock())

    assert result["response"] == "cached answer"
    assert result["citations"] == _CITATIONS
