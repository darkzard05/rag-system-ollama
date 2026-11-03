"""The prompt's citation doc_id contract must match format_context tokens.

format_context emits ``[doc:<stable_id>]`` tokens (hash strings), so the
structured-output schema and few-shot examples must ask the model to copy
that token verbatim. An integer schema makes small local models emit
``"doc_id": null``, which the fail-closed normalizer drops.
"""

import re

from langchain_core.documents import Document

from common.config import PROMPT_TEMPLATES_CONFIG
from common.utils import doc_stable_id
from core.graph._generate import _normalize_citations, format_context


def _prompt() -> str:
    prompt = PROMPT_TEMPLATES_CONFIG.get("structured_output", "")
    assert prompt, "structured_output prompt is empty"
    return prompt


def test_schema_declares_doc_id_as_context_token_string():
    prompt = _prompt()
    assert '"doc_id": {"type": "string"' in prompt
    assert '"doc_id": {"type": "integer"' not in prompt
    assert "그대로 복사" in prompt


def test_static_prompt_blocks_precede_dynamic_context():
    """정적 블록은 [Context] 앞에 있어야 Ollama prefix 캐시가 재사용한다.

    회귀 배경: 스키마/예시가 [Context] 뒤에 있으면 매 질의마다 동적 컨텍스트
    이후 토큰을 다시 평가해 prompt eval이 ~6.5s까지 늘어났다.
    """
    prompt = _prompt()
    last_context = prompt.rindex("[Context]")
    last_question = prompt.rindex("[Question]")

    assert prompt.rindex("</JSON_Schema>") < last_context
    assert prompt.rindex("</Few_Shot_Examples>") < last_context
    assert last_context < last_question


def test_few_shot_ids_match_stable_id_token_shape():
    prompt = _prompt()
    example_ids = re.findall(r"\[doc:([0-9a-f]{8,})\]", prompt)
    assert example_ids, "few-shot examples must use hash-shaped doc tokens"
    assert set(example_ids) == set(re.findall(r'"doc_id":\s*"([0-9a-f]{8,})"', prompt))


def test_normalize_preserves_stable_id_copied_from_context():
    docs = [
        Document(page_content="alpha body", metadata={}),
        Document(page_content="beta body", metadata={}),
    ]
    context = format_context(docs)
    token = re.search(r"\[doc:([^\]]+)\]", context)
    assert token is not None
    stable_id = token.group(1)
    assert stable_id == doc_stable_id(docs[0])

    out = _normalize_citations([{"doc_id": stable_id, "anchor": "alpha"}], docs)
    assert len(out) == 1
    assert out[0]["doc_id"] == stable_id


def test_normalize_still_drops_null_doc_id():
    docs = [Document(page_content="alpha body", metadata={})]
    assert _normalize_citations([{"doc_id": None, "anchor": "alpha"}], docs) == []
