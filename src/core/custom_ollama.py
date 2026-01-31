import logging
from collections.abc import AsyncIterator
from typing import Any

import ollama
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)


class DeepThinkingChatOllama(ChatOllama):
    """
    Ollama의 비표준 'thinking' 필드를 지원하는 커스텀 ChatOllama 클래스.
    Ollama API 응답의 message.thinking 데이터를 캡처하여 additional_kwargs에 저장합니다.
    """

    @property
    def async_client(self) -> ollama.AsyncClient:
        """비동기 클라이언트를 지연 초기화하고 재사용합니다."""
        # Pydantic 모델의 필드 보호를 우회하기 위해 object.__setattr__ 사용
        if not hasattr(self, "_async_client") or self._async_client is None:
            object.__setattr__(
                self, "_async_client", ollama.AsyncClient(host=self.base_url)
            )
        return self._async_client

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        비동기 스트리밍을 오버라이딩하여 thinking 필드를 직접 추출합니다.
        """
        try:
            # 메시지 형식 변환 (LangChain -> Ollama)
            formatted_messages = []
            for m in messages:
                role = "user"
                if m.type == "system":
                    role = "system"
                elif m.type == "ai":
                    role = "assistant"
                formatted_messages.append({"role": role, "content": m.content})

            # Ollama 요청 옵션 구성
            options = {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
                "top_p": self.top_p,
                "num_ctx": self.num_ctx,
            }
            if stop:
                options["stop"] = stop

            # 재사용되는 클라이언트를 통해 스트리밍 실행
            async for part in await self.async_client.chat(
                model=self.model,
                messages=formatted_messages,
                stream=True,
                options=options,
                **kwargs,
            ):
                message_part = part.get("message", {})
                content = message_part.get("content", "")
                # 핵심: message 객체 내부의 thinking 필드 추출
                thinking = message_part.get("thinking", "")

                chunk = AIMessageChunk(
                    content=content,
                    additional_kwargs={"thinking": thinking} if thinking else {},
                    id=part.get("model"),  # 대략적인 ID 부여
                )

                yield ChatGenerationChunk(message=chunk)

        except Exception as e:
            logger.error(f"[CustomOllama] 스트리밍 오류: {e}")
            raise
        # 참고: 일부 버전에서는 client에 별도의 close()가 없을 수 있으므로
        # 에러 발생 시 로그만 남기고 종료합니다.
        # 만약 라이브러리가 명시적 close를 지원한다면 여기서 호출 가능합니다.
