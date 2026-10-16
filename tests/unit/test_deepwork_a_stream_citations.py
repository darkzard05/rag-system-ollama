"""Phase A — streaming citation verification (fail-closed at the source).

Covers:
- _filter_stream_cite_items drops unknown doc_id, maps int index, passes
  through when docs is None (legacy contract for direct _generate_pass calls).
- _generate_pass with docs= filters citations_add items + final citations event.
- merge_citations_add valid_ids drops ghosts; None keeps legacy behavior.
- Parse-failure path always dispatches authoritative citations event (reset).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import core.graph._generate as gen
from ui.components.streaming_core import merge_citations_add


def _doc(sid: str) -> Document:
    return Document(page_content=f"content {sid}", metadata={"doc_id": sid})


def _cit(doc_id: object) -> dict:
    return {
        "doc_id": doc_id,
        "text_span": "span",
        "section": "s",
        "page": 1,
        "score": 0.9,
    }


class _FakeChunk:
    def __init__(self, content: str) -> None:
        self.content = content
        self.response_metadata: dict[str, Any] = {}


def _make_llm(tokens: list[str]) -> MagicMock:
    chunks = [_FakeChunk(t) for t in tokens]

    async def _astream(_messages: Any, config: Any = None) -> Any:
        for c in chunks:
            yield c

    llm = MagicMock()
    llm.bind.return_value.astream = _astream
    del llm._convert_chunk_to_thought_and_content
    return llm


async def _aenter(*args: Any, **kwargs: Any) -> Any:
    return None


async def _aexit(*args: Any, **kwargs: Any) -> Any:
    return False


def _split(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


async def _patched_pass(**kwargs: Any) -> Any:
    """Run _generate_pass with infra mocks + event capture; returns (result, events)."""
    events: list[tuple[str, dict[str, Any]]] = []

    async def _capture(name: str, data: dict[str, Any], **kw: Any) -> None:
        events.append((name, data))

    with (
        patch.object(gen, "get_resource_manager") as rm,
        patch.object(gen.ModelManager, "inference_session") as sess,
        patch.object(gen, "_dispatch_event", _capture),
    ):
        sess.return_value.__aenter__ = _aenter
        sess.return_value.__aexit__ = _aexit
        coord = rm.return_value
        coord.use_llm.return_value.__aenter__ = _aenter
        coord.use_llm.return_value.__aexit__ = _aexit
        result = await gen._generate_pass(**kwargs)
    return result, events


def test_filter_drops_unknown_and_maps_index() -> None:
    docs = [_doc("d1"), _doc("d2")]
    out = gen._filter_stream_cite_items(
        [_cit("d1"), _cit(999), _cit("ghost"), _cit(1)], docs
    )
    assert [c["doc_id"] for c in out] == ["d1", "d2"]


def test_filter_passthrough_without_docs() -> None:
    items = [_cit(999), _cit("ghost")]
    assert gen._filter_stream_cite_items(items, None) == items
    assert gen._filter_stream_cite_items(items, []) == items


@pytest.mark.asyncio
async def test_generate_pass_filters_stream_and_final_with_docs() -> None:
    docs = [_doc("d1"), _doc("d2")]
    payload = (
        '{"final_answer":"ok",'
        '"citations":['
        '{"doc_id":"d1","text_span":"s","section":"s","page":1,"score":0.9},'
        '{"doc_id":999,"text_span":"x","section":"s","page":1,"score":0.9},'
        '{"doc_id":1,"text_span":"y","section":"s","page":2,"score":0.8}'
        "]}"
    )
    llm = _make_llm(_split(payload, 7))
    result, events = await _patched_pass(
        llm=llm,
        sys_msg=MagicMock(),
        human_msg=MagicMock(),
        prompt_version="v1",
        config={},
        writer=object(),
        attempt=1,
        docs=docs,
    )
    assert result["parse_failed"] is False
    adds = [d for n, d in events if n == "citations_add"]
    assert adds, "valid citations must still stream"
    streamed = [o["doc_id"] for d in adds for o in d["items"]]
    assert 999 not in streamed
    assert set(streamed) == {"d1", "d2"}
    finals = [d for n, d in events if n == "citations"]
    assert finals, "final citations event must be dispatched"
    assert {c["doc_id"] for c in finals[-1]["citations"]} == {"d1", "d2"}


@pytest.mark.asyncio
async def test_generate_pass_passthrough_without_docs_pins_legacy() -> None:
    payload = (
        '{"final_answer":"ok",'
        '"citations":[{"doc_id":"d1","text_span":"s","section":"s","page":1,"score":0.9}]}'
    )
    llm = _make_llm(_split(payload, 5))
    _, events = await _patched_pass(
        llm=llm,
        sys_msg=MagicMock(),
        human_msg=MagicMock(),
        prompt_version="v1",
        config={},
        writer=object(),
        attempt=1,
    )
    adds = [d for n, d in events if n == "citations_add"]
    assert adds
    assert [o["doc_id"] for d in adds for o in d["items"]] == ["d1"]


@pytest.mark.asyncio
async def test_parse_failure_dispatches_authoritative_reset() -> None:
    docs = [_doc("d1")]
    bad = '{"final_answer":"Hel lo" BROKEN}}}}not json'
    llm = _make_llm(_split(bad, 5))
    result, events = await _patched_pass(
        llm=llm,
        sys_msg=MagicMock(),
        human_msg=MagicMock(),
        prompt_version="v1",
        config={},
        writer=object(),
        attempt=1,
        docs=docs,
    )
    assert result["parse_failed"] is True
    finals = [d for n, d in events if n == "citations"]
    assert finals, "parse failure must still dispatch authoritative citations"
    assert finals[-1]["citations"] == []
    assert "BROKEN" not in result["parsed_answer"].final_answer


def test_merge_valid_ids_drops_ghost() -> None:
    merged = merge_citations_add(
        [],
        items=[_cit("d1"), _cit("ghost")],
        base_index=0,
        arr_len=2,
        valid_ids={"d1"},
    )
    assert [c["doc_id"] for c in merged] == ["d1"]


def test_merge_without_valid_ids_pins_legacy() -> None:
    merged = merge_citations_add([], items=[_cit("ghost")], base_index=0, arr_len=1)
    assert [c["doc_id"] for c in merged] == ["ghost"]
