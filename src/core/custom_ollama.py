import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk
import ollama

logger = logging.getLogger(__name__)

class DeepThinkingChatOllama(ChatOllama):
    """
    Ollama의 비표준 'thinking' 필드를 지원하는 커스텀 ChatOllama 클래스.
    Ollama API 응답의 message.thinking 데이터를 캡처하여 additional_kwargs에 저장합니다.
    """

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        비동기 스트리밍을 오버라이딩하여 thinking 필드를 직접 추출합니다.
        """
        try:
            client = ollama.AsyncClient(host=self.base_url)
            
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

            # 스트리밍 실행
            async for part in await client.chat(
                model=self.model,
                messages=formatted_messages,
                stream=True,
                options=options,
                **kwargs
            ):
                message_part = part.get("message", {})
                content = message_part.get("content", "")
                # 핵심: message 객체 내부의 thinking 필드 추출
                thinking = message_part.get("thinking", "")
                
                chunk = AIMessageChunk(
                    content=content,
                    additional_kwargs={"thinking": thinking} if thinking else {},
                    id=part.get("model") # 대략적인 ID 부여
                )
                
                yield ChatGenerationChunk(message=chunk)
                
        except Exception as e:
            logger.error(f"[CustomOllama] 스트리밍 오류: {e}")
            # 에러 발생 시 기본 구현으로 폴백 시도하거나 예외 발생
            raise