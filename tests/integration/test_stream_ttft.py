"""
F2 게이트 검증: 구조화(structured / raw_json) 모드 실시간성(TTFT) 통합 테스트.

- 모의 LLM(`astream`)이 토큰 사이 `await asyncio.sleep(...)` 으로 시간을 벌며
  여러 토큰을 실시간으로 yield 하는지
- `generate()` 가 완료되기 **이전**에 첫 `response_chunk`(빈 content 가 아닌 원시 JSON
  토큰)가 수신되는지 — 즉 TTFT(time-to-first-token) 가 생성 완료 시점보다 빠른지

소스 수정 없이 `core.graph_builder.adispatch_custom_event` 를 monkeypatch 하여
`(timestamp, event_name, data)` 를 캡처한다 (writer 게이트는 truthy writer 로 통과).
"""

import asyncio
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
async def test_structured_stream_ttft_first_chunk_before_completion():
    """구조화 모드 스트리밍: 첫 response_chunk 수신 시점이 생성 완료 이전임을 입증."""
    captured = _Captured()
    prompt_config = {"structured_output": "CTX:{context} Q:{query}"}

    # 원시 JSON 토큰들을 시간 간격을 두고 실시간 yield 하는 모의 LLM.
    tokens = [
        '{"final_answer":"Hel',
        'lo world","reasoning',
        '":"r","confidence',
        '":0.9}',
    ]

    llm = MagicMock()

    async def mock_astream(messages, config=None):  # noqa: ANN001
        for tok in tokens:
            # 각 토큰 사이에 실제 시간을 소비 → first_chunk_time < completion_time 이
            # 진짜 타이밍 단언이 되도록 한다.
            await asyncio.sleep(0.05)
            yield SimpleNamespace(content=tok, response_metadata={})

    llm.astream = mock_astream
    if hasattr(llm, "_convert_chunk_to_thought_and_content"):
        del llm._convert_chunk_to_thought_and_content

    docs = [Document(page_content="context doc")]

    with (
        patch("core.graph_builder.PROMPT_TEMPLATES_CONFIG", prompt_config),
        patch("core.graph_builder.adispatch_custom_event", new=captured._append_async),
        patch.object(ModelManager, "inference_session", _NullAsyncCtx),
    ):
        completion_time = time.perf_counter()
        await generate(
            {"input": "q", "relevant_docs": docs},
            {"configurable": {"llm": llm}},
            writer=MagicMock(),
        )
        completion_time = time.perf_counter()  # overwrite after return

    # 첫 response_chunk 중 빈 content 가 아닌(원시 JSON 토큰) 이벤트의 타임스탬프.
    response_chunks = [c for c in captured if c["name"] == "response_chunk"]
    assert response_chunks, "no response_chunk events captured"

    first_chunk = next(c for c in response_chunks if c["data"].get("content"))
    first_chunk_time = first_chunk["ts"]

    # 실시간성(TTFT) 입증: 첫 토큰이 모델 생성 완료 이전에 스트리밍되었다.
    assert first_chunk_time < completion_time, (
        f"first_chunk_time={first_chunk_time} must be < "
        f"completion_time={completion_time} (TTFT violation)"
    )
