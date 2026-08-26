"""
Todo 4 검증: 사고(thought) 라이브 스트리밍 경로.

- (a) 비구조화 모드(PROMPT_TEMPLATES_CONFIG={})에서 LLM이 thought/content 를 분리
     산출할 때, thought 가 content 와 인터리브되어 라이브 전송되는지
     (벌크 duplicate reasoning emit 가 없어야 함)
- (b) 구조화 모드에서 스트리밍 중 thought 채널은 비어있고, 완료 시점에 단 1회의
     one-shot reasoning emit(content="", thought==parsed.reasoning) 만 발생하는지
     (원시 JSON 토큰이 thought 를 오염시키지 않아야 함)

소스 수정 없이 `core.graph_builder.generate` 의 모듈 레벨 `adispatch_custom_event` 를
monkeypatch 하여 이벤트를 캡처한다.
"""

import json
import time
from types import SimpleNamespace
from typing import Any
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


class _Captured(list):
    """이벤트 캡처용 리스트 서브클래스 (async append 스파이)."""

    async def _append_async(self, name, data, config=None):  # noqa: ANN001
        self.append({"name": name, "data": data, "ts": time.perf_counter()})


@pytest.mark.asyncio
async def test_non_structured_thought_interleaved():
    """비구조화 모드: thought 가 content 와 인터리브되어 라이브 전송되고 벌크 duplicate reasoning emit 가 없다."""
    captured = _Captured()
    # 빈 설정 -> use_structured_output=False
    prompt_config: dict[str, Any] = {}

    llm = MagicMock()
    llm.bind.return_value = llm
    # thought/content 가 섞여(stream) 방출되도록 청크를 인터리브 배치한다.
    # content 청크들은 합쳐져 유효한 AnswerStructure JSON 이 되도록 한다
    # (non-structured 모드에서도 generate 는 full_response 를 JSON 파싱 시도).
    ordered = [
        SimpleNamespace(content="T:idea1", response_metadata={}),
        SimpleNamespace(content='{"reasoning":"r","final_a', response_metadata={}),
        SimpleNamespace(content="T:idea2", response_metadata={}),
        SimpleNamespace(content='nswer":"Hello"}', response_metadata={}),
    ]

    async def mock_astream(messages, config=None):  # noqa: ANN001
        for c in ordered:
            yield c

    llm.astream = mock_astream

    def split(chunk):
        content = chunk.content
        # thought 청크("T:" 접두) vs answer 청크 구분
        if content.startswith("T:"):
            return "", content[2:]
        return content, ""

    llm._convert_chunk_to_thought_and_content = split

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
    assert len(responses) == 4, f"expected 4 response_chunk, got {len(responses)}"

    thought_events = [c for c in responses if c["data"]["thought"]]
    content_events = [c for c in responses if c["data"]["content"]]

    # 라이브 thought 방출이 최소 1회 존재
    assert len(thought_events) >= 1, "no live thought emissions during streaming"
    # content 방출도 존재
    assert len(content_events) >= 1

    # 비구조화 모드: raw_json 플래그가 False 여야 함 (raw_json 이슈 없음)
    for c in responses:
        assert c["data"].get("raw_json") is False

    # thought 와 content 가 인터리브(섞여) 전송되는지:
    # thought 가 content 보다 앞과 뒤 양쪽에 나타나야 함 (단순 한 덩어리 벌크가 아님)
    seq = ["T" if c["data"]["thought"] else "C" for c in responses]
    assert "T" in seq
    assert "C" in seq
    assert seq.index("T") < seq.index("C"), "first emit should be thought (live)"
    # 마지막 thought 위치가 첫 content 위치보다 뒤에 있어야 인터리브됨
    last_t = max(i for i, v in enumerate(seq) if v == "T")
    assert last_t > seq.index("C"), (
        "thought must appear after a content emit (interleaved)"
    )

    # 비구조화 모드에서는 완료 시점 one-shot reasoning emit(벌크 duplicate) 이 없다:
    # 마지막 이벤트는 content 를 가진 일반 청크여야 함.
    assert responses[-1]["data"]["content"] != "", (
        "non-structured mode must not end with a bulk duplicate reasoning emit"
    )

    # content 가 유효한 JSON 이므로 파싱 성공
    assert result["parse_failed"] is False


@pytest.mark.asyncio
async def test_structured_thought_one_shot():
    """구조화 모드: 스트리밍 중 thought 비어있고 완료 시 단 1회 one-shot reasoning emit."""
    captured = _Captured()
    prompt_config = {"structured_output": "CTX:{context} Q:{query}"}

    # 원시 JSON 에 reasoning 필드 포함
    tokens = [
        '{"final_answer":"Hel',
        'lo world","reasoning',
        '":"step by step","confidence',
        '":0.9}',
    ]
    assert json.loads("".join(tokens))["reasoning"] == "step by step"

    llm = MagicMock()
    llm.bind.return_value = llm

    async def mock_astream(messages, config=None):  # noqa: ANN001
        for tok in tokens:
            yield SimpleNamespace(content=tok, response_metadata={})

    llm.astream = mock_astream
    # 구조화 원시 경로: split 메서드가 없어야 함
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
    assert len(responses) == 5, (
        f"expected 5 response_chunk (4 raw + 1 thought), got {len(responses)}"
    )

    raw_events = responses[:4]
    thought_event = responses[4]

    # 스트리밍 중 thought 채널은 비어있어야 함 (원시 JSON 이 thought 를 오염 X)
    for i, ev in enumerate(raw_events):
        assert ev["data"]["thought"] == "", f"raw emit {i} must not carry thought"
        assert ev["data"]["raw_json"] is True
        assert ev["data"]["content"] == tokens[i]

    # 완료 시점 정확히 1회의 one-shot thought emit
    assert thought_event["data"]["content"] == ""
    assert thought_event["data"]["thought"] == "step by step"

    assert result["parse_failed"] is False
    assert result["response"] == "Hello world"
