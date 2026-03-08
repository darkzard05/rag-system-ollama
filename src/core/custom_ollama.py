import logging
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class DeepThinkingChatOllama(ChatOllama):
    """
    Ollama의 공식 'content_blocks' (reasoning 등) 지원을 활용하는 ChatOllama 래퍼.
    최신 langchain-ollama 버전은 DeepSeek-R1 등의 모델에서 사고 과정을 자동으로 분리합니다.
    """

    def __init__(self, **kwargs: Any):
        # [표준] stream_content_blocks 옵션을 활성화하여 사고 과정을 분리해서 받도록 설정
        # 이 옵션이 켜지면 Ollama API의 reasoning 필드가 content_blocks로 들어옵니다.
        if "stream_content_blocks" not in kwargs:
            kwargs["stream_content_blocks"] = True
        super().__init__(**kwargs)

    def _convert_chunk_to_thought_and_content(
        self, chunk: AIMessageChunk
    ) -> tuple[str, str]:
        """
        AIMessageChunk에서 사고 과정(thought)과 실제 답변(content)을 표준 방식으로 추출합니다.
        1순위: content_blocks (최신 표준)
        2순위: content (리스트 형태의 복합 콘텐츠)
        3순위: additional_kwargs (레거시/커스텀 필드)
        """
        content = ""
        thought = ""

        # A. [최신 표준] content_blocks 확인
        # langchain-ollama 0.2.0+ 버전에서 stream_content_blocks=True일 때 활성화됨
        if hasattr(chunk, "content_blocks") and chunk.content_blocks:
            for block in chunk.content_blocks:
                if not isinstance(block, dict):
                    continue

                b_type = block.get("type")
                if b_type == "reasoning":
                    thought += block.get("reasoning", "")
                elif b_type == "thought":  # 일부 모델 변종 대응
                    thought += block.get("thought", "")
                elif b_type == "text":
                    content += block.get("text", "")

        # B. [복합 콘텐츠] chunk.content가 리스트인 경우 (Anthropic 스타일 등)
        if not content and not thought and isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict):
                    i_type = item.get("type")
                    if i_type == "text":
                        content += item.get("text", "")
                    elif i_type in ["reasoning", "thought", "thinking"]:
                        thought += item.get(i_type, item.get("text", ""))
                elif isinstance(item, str):
                    content += item

        # C. [기본/레거시] 일반 문자열 콘텐츠 및 additional_kwargs
        if not content and isinstance(chunk.content, str):
            content = chunk.content

        if not thought:
            # 레거시 DeepSeek-R1 등에서 사용하던 필드들 체크
            thought = (
                chunk.additional_kwargs.get("reasoning")
                or chunk.additional_kwargs.get("thinking")
                or chunk.additional_kwargs.get("thought")
                or ""
            )

        return str(content), str(thought)
