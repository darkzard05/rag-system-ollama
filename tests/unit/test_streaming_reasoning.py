from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessageChunk

from core.graph_builder import generate


class MockChunk(AIMessageChunk):
    """테스트용 AIMessageChunk 모의 객체"""

    def __init__(self, content="", thought="", is_last=False):
        # langchain_ollama의 최신 구조와 유사하게 모킹
        content_blocks = []
        if thought:
            content_blocks.append({"type": "reasoning", "reasoning": thought})
        if content:
            content_blocks.append({"type": "text", "text": content})

        super().__init__(content=content, content_blocks=content_blocks)
        if is_last:
            self.response_metadata = {"prompt_eval_count": 10, "model": "test-model"}
        else:
            self.response_metadata = {}


@pytest.mark.asyncio
@patch("core.graph_builder.adispatch_custom_event")
async def test_generate_streams_thought_and_content(mock_dispatch):
    """generate 노드가 사고 과정과 답변 본문을 실시간으로 StreamWriter로 전달하는지 테스트"""

    # 1. 의존성 모킹
    mock_llm = MagicMock()
    # generate()는 llm.bind(response_format=...).astream(...) 을 호출하므로
    # bind() 가 모킹된 astream 을 가진 자기 자신을 리턴하도록 세팅한다.
    mock_llm.bind.return_value = mock_llm

    # astream 제너레이터 모킹 (사고 2번, 답변 2번 생성)
    async def mock_astream(*args, **kwargs):
        chunks = [
            MockChunk(thought="I am thinking "),
            MockChunk(thought="about the answer."),
            MockChunk(content="The answer is "),
            MockChunk(content="42.", is_last=True),
        ]
        for chunk in chunks:
            yield chunk

    mock_llm.astream = mock_astream

    # _convert_chunk_to_thought_and_content 로직 모킹 (실제 클래스 로직과 동일하게 구현)
    def mock_convert(chunk):
        thought = ""
        content = ""
        for block in chunk.content_blocks:
            if block["type"] == "reasoning":
                thought += block["reasoning"]
            elif block["type"] == "text":
                content += block["text"]
        return content, thought

    mock_llm._convert_chunk_to_thought_and_content = mock_convert

    # 2. 입력 데이터 준비
    state = {
        "input": "What is the answer?",
        "relevant_docs": [
            MagicMock(page_content="Reference doc", metadata={"page": 1})
        ],
        "is_cached": False,
    }

    config = {"configurable": {"llm": mock_llm}}
    writer = MagicMock()

    # 3. 테스트 실행
    # ModelManager.inference_session은 async context manager이므로 모킹 필요
    # patch.object로 교체해 테스트 종료 후 원복 (클래스 속성 오염 방지)
    from core.model_loader import ModelManager

    with patch.object(ModelManager, "inference_session") as mock_session:
        mock_session.return_value.__aenter__ = AsyncMock()
        mock_session.return_value.__aexit__ = AsyncMock()

        result = await generate(state, config, writer=writer)

    # 4. 검증
    # adispatch_custom_event가 thought와 content를 포함하여 호출되었는지 확인
    # "response_chunk" 이벤트가 2번 이상 thought가 있고 2번 이상 content가 있어야 함
    thought_calls = [
        call
        for call in mock_dispatch.call_args_list
        if len(call.args) > 1 and call.args[1].get("thought")
    ]
    content_calls = [
        call
        for call in mock_dispatch.call_args_list
        if len(call.args) > 1 and call.args[1].get("content")
    ]

    assert len(thought_calls) >= 2
    assert len(content_calls) >= 2

    # 누적된 결과 확인
    assert result["thought"] == "I am thinking about the answer."
    assert result["response"] == "The answer is 42."
    assert result["performance"]["input_token_count"] == 10
