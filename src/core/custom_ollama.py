import asyncio
import logging
from typing import Any

import ollama
from langchain_core.messages import AIMessageChunk
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

# reasoning 지원 모델 접두사 허용 목록. Ollama의 think 파라미터를 지원하지 않는
# 일반 모델에 reasoning=True를 넘기면 요청이 거부될 수 있어, allowlist 외에는
# reasoning을 ChatOllama에 전달하지 않는다.
_THINKING_PREFIXES = ("qwen3", "deepseek", "r1")


class DeepThinkingChatOllama(ChatOllama):
    """
    Ollama의 공식 'content_blocks' (reasoning 등) 지원을 활용하는 ChatOllama 래퍼.
    최신 langchain-ollama 버전은 DeepSeek-R1 등의 모델에서 사고 과정을 자동으로 분리합니다.
    """

    def __init__(self, **kwargs: Any):
        # [F2b] reasoning은 kwargs에서 분리해 처리한다. langchain-ollama 0.3.10에서
        # reasoning은 Ollama 요청의 'think' 파라미터로 매핑된다 (chat_models.py:680).
        # 기본값은 True이되 allowlist 외 모델에는 전달하지 않아 think 미지원 모델을 보호한다.
        reasoning = kwargs.pop("reasoning", None)
        if reasoning is None:
            reasoning = True
        model_name = str(kwargs.get("model", "")).lower()
        if reasoning is True and any(
            prefix in model_name for prefix in _THINKING_PREFIXES
        ):
            kwargs["reasoning"] = reasoning
        # [R1b-01] timeout을 별도 kwargs로 받아 ollama httpx 클라이언트에 주입한다.
        # ChatOllama의 최상위 'timeout' 필드는 존재하지 않아 무시되므로, Client/AsyncClient
        # 생성자에 전달되는 client_kwargs로 라우팅해야 실제 HTTP 타임아웃이 적용된다.
        timeout = kwargs.pop("timeout", None)
        if timeout is not None:
            client_kwargs = dict(kwargs.get("client_kwargs") or {})
            client_kwargs["timeout"] = timeout
            kwargs["client_kwargs"] = client_kwargs
        super().__init__(**kwargs)
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._loop_safe_async_client: ollama.AsyncClient | None = None

    # =========================================================================
    # [FIX] Event loop is closed 오류 해결:
    # 스트리밍 턴마다 새 스레드/루프가 생성되므로, 루프 변경 감지 시 클라이언트를 자동 갱신
    # =========================================================================
    @property
    def _async_client(self) -> ollama.AsyncClient:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if (
            self._loop_safe_async_client is None
            or self._active_loop is not current_loop
            or (current_loop and current_loop.is_closed())
        ):
            client_kwargs = dict(getattr(self, "client_kwargs", None) or {})
            self._loop_safe_async_client = ollama.AsyncClient(
                host=self.base_url,
                **client_kwargs,
            )
            self._active_loop = current_loop
        return self._loop_safe_async_client

    @_async_client.setter
    def _async_client(self, value: ollama.AsyncClient) -> None:
        """langchain-ollama declares _async_client as a writeable PrivateAttr, so a
        getter-only override is an unsafe narrowing and mypy rejects it with
        [override]. Declaring the setter restores the writeable contract.

        This setter is defensive: pydantic routes the parent __init__ assignment into
        __pydantic_private__ rather than through this descriptor, so reads keep coming
        from the getter above and the eagerly built parent client stays unused. If it
        ever is called, drop the cached client so the next read rebuilds it against the
        running event loop instead of pinning a dead one.
        """
        self._loop_safe_async_client = None
        self._active_loop = None

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
                if b_type in ["reasoning", "thought", "thinking"]:
                    thought += str(
                        block.get("reasoning")
                        or block.get("thought")
                        or block.get("thinking")
                        or ""
                    )
                elif b_type == "text":
                    content += str(block.get("text") or "")

        # B. [복합 콘텐츠] chunk.content가 리스트인 경우 (Anthropic 스타일 등)
        if not content and not thought and isinstance(chunk.content, list):
            for item in chunk.content:
                if isinstance(item, dict):
                    i_type = item.get("type")
                    if i_type == "text":
                        content += str(item.get("text") or "")
                    elif i_type in ["reasoning", "thought", "thinking"]:
                        # i_type에 해당하는 값과 'text' 필드 중 있는 것을 추출
                        val = item.get(i_type) or item.get("text")
                        thought += str(val or "")
                elif isinstance(item, str):
                    content += item

        # C. [기본/레거시] 일반 문자열 콘텐츠 및 additional_kwargs
        if not content and isinstance(chunk.content, str):
            content = chunk.content

        if not thought:
            # reasoning_content가 최신 langchain-ollama(0.3.x)의 키, 나머지는 레거시 폴백
            thought = (
                chunk.additional_kwargs.get("reasoning_content")
                or chunk.additional_kwargs.get("reasoning")
                or chunk.additional_kwargs.get("thinking")
                or chunk.additional_kwargs.get("thought")
                or ""
            )

        return str(content), str(thought)
