import logging
from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, ChatResult

logger = logging.getLogger(__name__)

# [최적화] 전역에서 ChatOllama를 상속받기 위해 최소한의 임포트만 유지하거나,
# 런타임에 임포트된 클래스를 사용하도록 구조를 변경합니다.
# 여기서는 클래스 정의를 위해 상단에 두고, 무거운 연산이 시작되는 시점을 통제합니다.
from langchain_ollama import ChatOllama  # noqa: E402


class DeepThinkingChatOllama(ChatOllama):
    """
    Ollama의 비표준 'thinking' 필드를 지원하는 커스텀 ChatOllama 클래스.
    Ollama API 응답의 message.thinking 데이터를 캡처하여 additional_kwargs에 저장합니다.
    """

    def _prepare_ollama_request(
        self,
        messages: list["BaseMessage"],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Ollama 요청 파라미터를 준비합니다."""
        formatted_messages = self._convert_messages_to_ollama_messages(messages)

        options = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "top_p": self.top_p,
            "num_ctx": self.num_ctx,
            "stop": stop,
        }

        if "options" in kwargs:
            options.update(kwargs.pop("options"))

        request_kwargs = {
            "model": self.model,
            "messages": formatted_messages,
            "options": {k: v for k, v in options.items() if v is not None},
            "keep_alive": getattr(self, "keep_alive", None),
        }

        # [최적화] Ollama v0.6.0+ 공식 'think' 파라미터 우선 처리
        if "think" in kwargs:
            request_kwargs["think"] = kwargs.pop("think")
        elif any(
            r in self.model.lower() for r in ["deepseek-r1", "thought", "reasoning"]
        ):
            request_kwargs["think"] = True

        # 나머지 추가 인자 전달
        for k, v in kwargs.items():
            if k not in request_kwargs:
                request_kwargs[k] = v

        return request_kwargs

    def _stream(
        self,
        messages: list["BaseMessage"],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Iterator["ChatGenerationChunk"]:
        """동기 스트리밍을 오버라이딩하여 thinking 필드를 추출합니다."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        from core.model_loader import ModelManager

        try:
            request_kwargs = self._prepare_ollama_request(messages, stop, **kwargs)
            request_kwargs["stream"] = True

            host_url = str(self.base_url) if self.base_url else "http://localhost:11434"
            sync_client = ModelManager.get_client(host=host_url)
            response_gen = sync_client.chat(**request_kwargs)

            for part in response_gen:
                if not part or not hasattr(part, "message"):
                    continue

                message_part = part.message
                content = getattr(message_part, "content", "")
                thinking = getattr(message_part, "thinking", "")

                if not content and not thinking:
                    continue

                chunk = AIMessageChunk(
                    content=content,
                    additional_kwargs={"thinking": thinking} if thinking else {},
                    id=getattr(part, "model", None),
                )

                yield ChatGenerationChunk(message=chunk)

        except Exception as e:
            logger.error(f"[CustomOllama] 동기 스트리밍 중 오류: {e}")
            raise

    def _generate(
        self,
        messages: list["BaseMessage"],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> "ChatResult":
        """비스트리밍 생성을 오버라이딩하여 thinking 필드를 추출합니다."""
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        from core.model_loader import ModelManager

        try:
            request_kwargs = self._prepare_ollama_request(messages, stop, **kwargs)
            request_kwargs["stream"] = False

            sync_client = ModelManager.get_client(host=self.base_url)
            response = sync_client.chat(**request_kwargs)

            message_part = getattr(response, "message", None)
            content = getattr(message_part, "content", "")
            thinking = getattr(message_part, "thinking", "")

            message = AIMessage(
                content=content,
                additional_kwargs={"thinking": thinking} if thinking else {},
                id=getattr(response, "model", None),
            )

            return ChatResult(generations=[ChatGeneration(message=message)])

        except Exception as e:
            logger.error(f"[CustomOllama] 생성 중 오류: {e}")
            raise

    async def _astream(
        self,
        messages: list["BaseMessage"],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator["ChatGenerationChunk"]:
        """비동기 스트리밍을 오버라이딩하여 thinking 필드를 직접 추출합니다."""
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        from core.model_loader import ModelManager

        try:
            request_kwargs = self._prepare_ollama_request(messages, stop, **kwargs)
            request_kwargs["stream"] = True

            async_client = ModelManager.get_async_client(host=self.base_url)

            # Ollama API 호출 (비동기 제너레이터 획득)
            response_gen = await async_client.chat(**request_kwargs)

            async for part in response_gen:
                if not part or not hasattr(part, "message"):
                    continue

                message_part = part.message
                content = getattr(message_part, "content", "")
                thinking = getattr(message_part, "thinking", "")

                if not content and not thinking:
                    continue

                chunk = AIMessageChunk(
                    content=content,
                    additional_kwargs={"thinking": thinking} if thinking else {},
                    id=getattr(part, "model", None),
                )

                yield ChatGenerationChunk(message=chunk)

        except Exception as e:
            logger.error(f"[CustomOllama] 비동기 스트리밍 중 오류: {e}")
            raise
