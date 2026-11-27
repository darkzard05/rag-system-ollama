"""
F2 게이트 검증: 구조화(structured / raw_json) 모드 실시간성(TTFT) 통합 테스트.

- 모의 LLM(`astream`)이 토큰 사이 `await asyncio.sleep(...)` 으로 시간을 벌며
  여러 토큰을 실시간으로 yield 하는지
- `generate()` 가 완료되기 **이전**에 첫 `response_chunk`(빈 content 가 아닌 원시 JSON
  토큰)가 수신되는지 — 즉 TTFT(time-to-first-token) 가 생성 완료 시점보다 빠른지

소스 수정 없이 `core.graph.graph_builder.adispatch_custom_event` 를 monkeypatch 하여
`(timestamp, event_name, data)` 를 캡처한다 (writer 게이트는 truthy writer 로 통과).
"""

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from api.streaming_handler import StreamingResponseHandler, get_streaming_handler
from core.graph.graph_builder import generate
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
    # generate() wraps the injected llm with json_llm = llm.bind(...); the bound
    # object must carry the mocked astream so tokens are actually yielded.
    llm.bind.return_value = llm
    if hasattr(llm, "_convert_chunk_to_thought_and_content"):
        del llm._convert_chunk_to_thought_and_content

    docs = [Document(page_content="context doc")]

    with (
        patch("core.graph._generate.PROMPT_TEMPLATES_CONFIG", prompt_config),
        patch("core.graph._glue.adispatch_custom_event", new=captured._append_async),
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


class _FakeChunk:
    """messages 모드 청크 객체 — handler 가 ``hasattr(chunk_obj, 'content')`` 로 소비."""

    def __init__(self, content: str = "") -> None:
        self.content = content
        self.content_blocks: list[dict[str, object]] = []
        self.additional_kwargs: dict[str, object] = {}


async def _scripted_tokens(
    n: int, initial_delay_s: float = 0.05, pulled: list[int] | None = None
):
    """첫 토큰 전 지연(모델 콜드 스타트 시뮬레이션) + n 개 콘텐츠 토큰.

    ``pulled`` 가 주어지면 소비자가 첫 청크를 받은 시점에 소스에서 몇 개의
    토큰이 방출됐는지 세기 위해 인덱스를 기록한다 (벽시계 시간과 무관한 증거).
    """
    await asyncio.sleep(initial_delay_s)
    for i in range(n):
        if pulled is not None:
            pulled.append(i)
        yield ("messages", (_FakeChunk(content=f"tok{i} "), {}))


async def _measure_first_chunk(handler) -> tuple[str, float, int]:
    """첫 콘텐츠 청크 내용, TTFT, 첫 청크 방출 시까지 소비된 소스 토큰 수."""
    pulled: list[int] = []
    first_content: str | None = None
    pulled_at_first_chunk = 0
    async for chunk in handler.stream_graph_events(_scripted_tokens(8, pulled=pulled)):
        if chunk.content and first_content is None:
            first_content = chunk.content
            pulled_at_first_chunk = len(pulled)
    assert first_content is not None, "첫 콘텐츠 청크를 찾지 못했습니다."
    return (
        first_content,
        handler.metrics.first_token_latency or 0.0,
        pulled_at_first_chunk,
    )


@pytest.mark.asyncio
async def test_ttft_unaffected_by_buffer_size():
    """기본 설정(버퍼 크기 4)의 첫 토큰 방출이 size=1 기준과 동일함을 입증한다.

    ``TokenStreamBuffer.add_token`` 의 첫 토큰 바이패스는 버퍼 크기와 무관하게
    첫 콘텐츠 토큰을 즉시 flush 하므로, 크기 4(기본 config)와 크기 1(기존
    의미) 모두 **소스 토큰 1개만 소비한 시점**에 첫 콘텐츠 청크가 방출되어야 한다.

    두 TTFT 값의 직접 비교는 CI 러너의 콜드 스타트 지터(첫 측정에만 0.1s 이상의
    초기화 오버헤드가 실림) 때문에 플레이키하다. 그래서 핵심 증거는 벽시계 시간이
    아닌 소스 토큰 소비 수로 두고, TTFT 는 스크립트 지연(0.05s)보다 훨씬 큰
    절대 상한만으로 실시성을 sanity-check 한다.
    """
    # 기본 설정 — get_streaming_handler() 는 call-time 으로 UI_CONTENT_BUFFER_SIZE
    # (config.yml ui.streaming.content_buffer_size, 기본 4) 를 읽는다.
    default_handler = get_streaming_handler()
    assert default_handler.buffer.buffer_size == 4

    (
        default_first,
        default_ttft,
        default_pulled,
    ) = await _measure_first_chunk(default_handler)

    # 기존 size=1 의미 기준선
    baseline_handler = StreamingResponseHandler(content_buffer_size=1)
    baseline_first, baseline_ttft, baseline_pulled = await _measure_first_chunk(
        baseline_handler
    )

    # 첫 토큰 바이패스 보존: 두 경우 모두 첫 콘텐츠 청크가 정확히 첫 토큰.
    assert default_first == "tok0 "
    assert baseline_first == "tok0 "

    # 결정적 증거: 버퍼 크기와 무관하게 소스 토큰 1개만 읽고 즉시 방출했다.
    # 첫 토큰 바이패스가 없으면 size=4 는 4개를 모은 뒤에야 첫 청크를 내보낸다.
    assert default_pulled == 1, (
        f"default buffer held the first token: pulled={default_pulled}"
    )
    assert baseline_pulled == 1, (
        f"baseline buffer held the first token: pulled={baseline_pulled}"
    )

    # 실시성 sanity: 0.05s 스크립트 지연 + CI 러너 지터를 포함해도 상한 초과 금지.
    ttft_limit = 0.5
    assert default_ttft < ttft_limit, (
        f"default TTFT too slow: {default_ttft:.4f}s (limit={ttft_limit}s)"
    )
    assert baseline_ttft < ttft_limit, (
        f"baseline TTFT too slow: {baseline_ttft:.4f}s (limit={ttft_limit}s)"
    )
