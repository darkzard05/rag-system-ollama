"""
Todo 1 검증: AstroStream 원시 JSON(raw_json) 라이브 스트리밍 경로.

- (a) 구조화 모드(PROMPT_TEMPLATES_CONFIG에 "structured_output" 존재)에서 LLM이
     내보내는 원시 JSON 토큰들이 `raw_json=True` 플래그와 함께 실시간(라이브) 전송되는지
- (b) 파싱 전 원시 토큰 emit이 파싱 후 one-shot reasoning emit보다 먼저 발생하는지(타이밍)
- (c) JSON 파싱 실패 시 `parse_failed=True` 가 반환되고 예외가 새어나오지 않는지

소스 수정 없이 `core.graph_builder.generate` 의 모듈 레벨 `adispatch_custom_event` 를
monkeypatch 하여 이벤트를 캡처한다 (writer 게이트는 truthy writer 로 통과).
"""

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from core.graph_builder import generate
from core.model_loader import ModelManager


class _NullAsyncCtx:
    """ModelManager.inference_session() 모킹용 널 비동기 컨텍스트 매니저."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


# captured._append_async 를 쓰기 위해 스파이를 리스트 서브클래스로 만든다.
class _Captured(list):
    async def _append_async(self, name, data, config=None):  # noqa: ANN001
        self.append({"name": name, "data": data, "ts": time.perf_counter()})


@pytest.mark.asyncio
async def test_raw_json_live_5_tokens():
    """구조화 모드에서 5개 원시 JSON 토큰이 라이브로 raw_json=True 와 함께 전송된다."""
    captured = _Captured()
    prompt_config = {"structured_output": "CTX:{context} Q:{query}"}

    tokens = [
        '{"final_answer":"Hel',
        'lo world","reasoning',
        '":"r","confidence',
        '":0.9}',
    ]
    # 4개 토큰으로 분할하여도 검증 의도(원시 JSON 라이브 전송)는 동일하다.
    # (요구: 5개) → 5개로 재분할
    tokens = [
        '{"final_answer":"Hel',
        "lo wor",
        'ld","reasoning',
        '":"r","confidence',
        '":0.9}',
    ]
    assert json.loads("".join(tokens)) == {
        "final_answer": "Hello world",
        "reasoning": "r",
        "confidence": 0.9,
    }

    llm = MagicMock()
    # generate()는 llm.bind(response_format=...).astream(...) 를 호출하므로
    # bind() 가 모킹된 astream 을 가진 자기 자신을 리턴하도록 세팅한다.
    llm.bind.return_value = llm

    async def mock_astream(messages, config=None):  # noqa: ANN001
        for tok in tokens:
            yield SimpleNamespace(content=tok, response_metadata={})

    llm.astream = mock_astream
    # 구조화 원시 경로: _convert_chunk_to_thought_and_content 가 없어야 한다.
    if hasattr(llm, "_convert_chunk_to_thought_and_content"):
        del llm._convert_chunk_to_thought_and_content

    docs = [Document(page_content="context doc")]

    with (
        patch("core.graph_builder.PROMPT_TEMPLATES_CONFIG", prompt_config),
        patch("core.graph_builder.adispatch_custom_event", new=captured._append_async),
        patch.object(ModelManager, "inference_session", _NullAsyncCtx),
    ):
        result = await generate(
            {"input": "q", "relevant_docs": docs},
            {"configurable": {"llm": llm}},
            writer=MagicMock(),
        )

    responses = [c for c in captured if c["name"] == "response_chunk"]
    assert len(responses) == 6, (
        f"expected 6 response_chunk (5 raw + 1 thought), got {len(responses)}"
    )

    raw_events = responses[:5]
    thought_event = responses[5]

    # 5개 원시 emit 이 각각 raw_json=True, content==토큰
    for i, ev in enumerate(raw_events):
        assert ev["data"]["raw_json"] is True, f"raw emit {i} missing raw_json flag"
        assert ev["data"]["content"] == tokens[i], f"raw emit {i} content mismatch"
        assert ev["data"]["thought"] == "", f"raw emit {i} must not carry thought"

    # one-shot reasoning emit: content="" / thought == parsed reasoning "r"
    assert thought_event["data"]["content"] == ""
    assert thought_event["data"]["thought"] == "r"

    # 타이밍: 첫 원시 emit 이 completion(thought) emit 보다 먼저
    first_raw_ts = raw_events[0]["ts"]
    completion_ts = thought_event["ts"]
    assert first_raw_ts < completion_ts, "raw emits must precede completion emit"

    assert result["parse_failed"] is False
    assert result["response"] == "Hello world"


@pytest.mark.asyncio
async def test_parse_failure_preserves_flag():
    """구조화 모드에서 손상된 JSON 토큰 스트림은 parse_failed=True 로 폴백된다."""
    captured = _Captured()
    prompt_config = {"structured_output": "CTX:{context} Q:{query}"}

    bad_tokens = [
        '{"final_answer":"Hel',
        'lo world" BROKEN_EXTRA',
        "}}}}not json",
    ]
    joined = "".join(bad_tokens)
    is_valid = True
    try:
        json.loads(joined)
    except ValueError:
        is_valid = False
    assert is_valid is False, "bad token stream must NOT be valid JSON"

    llm = MagicMock()
    llm.bind.return_value = llm

    async def mock_astream(messages, config=None):  # noqa: ANN001
        for tok in bad_tokens:
            yield SimpleNamespace(content=tok, response_metadata={})

    llm.astream = mock_astream
    if hasattr(llm, "_convert_chunk_to_thought_and_content"):
        del llm._convert_chunk_to_thought_and_content

    docs = [Document(page_content="context doc")]

    result = None
    with (
        patch("core.graph_builder.PROMPT_TEMPLATES_CONFIG", prompt_config),
        patch("core.graph_builder.adispatch_custom_event", new=captured._append_async),
        patch.object(ModelManager, "inference_session", _NullAsyncCtx),
    ):
        # 예외가 새어나오지 않아야 한다 (내부에서 폴백 처리).
        result = await generate(
            {"input": "q", "relevant_docs": docs},
            {"configurable": {"llm": llm}},
            writer=MagicMock(),
        )

    assert result is not None
    assert result["parse_failed"] is True
    # 폴백 final_answer 는 원시(full) 응답 그대로
    assert "BROKEN_EXTRA" in result["response"]
