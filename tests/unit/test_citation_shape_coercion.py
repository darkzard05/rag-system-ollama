"""Small-model citation shape drift: ``{"doc:<id>": "<anchor>"}``.

qwen3:4b in json_object mode sometimes promotes the ``[doc:<id>]`` context
token to an object key instead of emitting ``{"doc_id": "...", "anchor": ...}``.
That used to raise ten pydantic ``doc_id Field required`` errors, discard the
already-parsed ``final_answer`` and leave the answer with zero citations.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

import core.graph._generate as gen
from api.schemas import AnswerStructure
from common.utils import coerce_citation_mapping, doc_stable_id
from core.graph._generate import _normalize_citations, _schema_fallback_answer
from core.graph._graph_core import _recover_citations


def _doc(sid: str) -> Document:
    return Document(page_content=f"content {sid}", metadata={"doc_id": sid})


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


def _split(text: str, n: int) -> list[str]:
    return [text[i : i + n] for i in range(0, len(text), n)]


async def _aenter(*args: Any, **kwargs: Any) -> Any:
    return None


async def _aexit(*args: Any, **kwargs: Any) -> Any:
    return False


async def _patched_pass(**kwargs: Any) -> tuple[Any, list[tuple[str, dict]]]:
    events: list[tuple[str, dict]] = []

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


def test_coerce_mapping_form_uses_doc_token_as_id() -> None:
    out = coerce_citation_mapping({"doc:abc123": "causal masking"})
    assert out["doc_id"] == "abc123"
    assert out["anchor"] == "causal masking"


def test_coerce_strips_typographic_quote_residue() -> None:
    # The observed 4B output carried a curly closing quote inside the token.
    assert coerce_citation_mapping({"doc:abc123”": "kw"})["doc_id"] == "abc123"
    assert coerce_citation_mapping({"doc_id": "doc:abc123”"})["doc_id"] == "abc123"


def test_coerce_strips_doc_prefix_from_doc_id_value() -> None:
    assert coerce_citation_mapping({"doc_id": "[doc:abc123]"})["doc_id"] == "abc123"
    assert coerce_citation_mapping({"doc_id": "doc:abc123"})["doc_id"] == "abc123"


def test_coerce_preserves_normal_and_fail_closed_shapes() -> None:
    normal = {"doc_id": "abc123", "anchor": "kw"}
    assert coerce_citation_mapping(normal) == normal
    int_id = {"doc_id": 3, "anchor": "kw"}
    assert coerce_citation_mapping(int_id)["doc_id"] == 3
    # null / missing id stays as-is so the membership boundary still drops it.
    null_id = {"doc_id": None, "anchor": "kw"}
    assert coerce_citation_mapping(null_id) is null_id
    assert coerce_citation_mapping("not-a-dict") == "not-a-dict"


def test_answer_structure_accepts_mapping_form() -> None:
    parsed = AnswerStructure(
        **{
            "final_answer": "answer",
            "citations": [{"doc:abc123": "kw"}],
            "confidence": 0.9,
        }
    )
    assert parsed.citations[0].doc_id == "abc123"
    assert parsed.citations[0].anchor == "kw"


def test_normalize_keeps_mapping_form(caplog: pytest.LogCaptureFixture) -> None:
    docs = [_doc("abc123"), _doc("def456")]
    assert doc_stable_id(docs[0]) == "abc123"
    with caplog.at_level(logging.WARNING):
        out = _normalize_citations([{"doc:abc123": "kw"}], docs)
    assert [c["doc_id"] for c in out] == ["abc123"]
    assert [c["anchor"] for c in out] == ["kw"]
    assert not [r for r in caplog.records if "드롭" in r.getMessage()]


def test_normalize_logs_one_summary_warning(caplog: pytest.LogCaptureFixture) -> None:
    docs = [_doc("abc123")]
    with caplog.at_level(logging.WARNING):
        out = _normalize_citations(
            [{"doc_id": "ghost1"}, {"doc_id": "ghost2"}, {"doc_id": None}], docs
        )
    assert out == []
    drops = [r for r in caplog.records if "드롭" in r.getMessage()]
    assert len(drops) == 1
    assert "3건" in drops[0].getMessage()


def test_recover_citations_mapping_form() -> None:
    blob = (
        '{"final_answer": "ans", "citations": ['
        '{"doc:abc123": "causal masking"}, '
        '{"doc:def456": "decoder"}]}'
    )
    recovered = _recover_citations(blob)
    assert recovered == [
        {"doc_id": "abc123", "anchor": "causal masking"},
        {"doc_id": "def456", "anchor": "decoder"},
    ]


def test_schema_fallback_preserves_final_answer() -> None:
    fallback = _schema_fallback_answer(
        {
            "final_answer": "kept answer",
            "citations": "not-a-list",
            "confidence": 7,
        }
    )
    assert fallback.final_answer == "kept answer"
    assert fallback.citations == []
    assert fallback.confidence == 1.0


@pytest.mark.asyncio
async def test_generate_pass_streams_mapping_form_citations() -> None:
    docs = [_doc("abc123"), _doc("def456")]
    payload = (
        '{"final_answer":"CM3 uses causal masking in the decoder.",'
        '"citations":['
        '{"doc:abc123": "causal masking"},'
        '{"doc_id": "doc:def456", "anchor": "decoder"}'
        "],"
        '"confidence":0.9}'
    )
    assert json.loads(payload)  # payload is valid JSON (only the shape is off)

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
    assert result["parsed_answer"].final_answer.startswith("CM3 uses causal")
    streamed = [
        o["doc_id"] for n, d in events if n == "citations_add" for o in d["items"]
    ]
    assert set(streamed) == {"abc123", "def456"}
    finals = [d for n, d in events if n == "citations"]
    assert {c["doc_id"] for c in finals[-1]["citations"]} == {"abc123", "def456"}


@pytest.mark.asyncio
async def test_generate_pass_keeps_answer_on_schema_failure() -> None:
    docs = [_doc("abc123")]
    payload = '{"final_answer":"kept answer","citations":"oops","confidence":0.9}'
    llm = _make_llm(_split(payload, 6))
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
    assert result["parsed_answer"].final_answer == "kept answer"
    finals = [d for n, d in events if n == "citations"]
    assert finals[-1]["citations"] == []
