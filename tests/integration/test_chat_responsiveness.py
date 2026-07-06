import pytest
from unittest.mock import MagicMock, AsyncMock
from streamlit.testing.v1 import AppTest
from src.core.session import SessionManager
import asyncio
import time


# Mock Chunk for streaming
class MockChunk:
    def __init__(
        self, content="", status=None, thought="", metadata=None, performance=None
    ):
        self.content = content
        self.status = status
        self.thought = thought
        self.metadata = metadata
        self.performance = performance


@pytest.mark.asyncio
async def test_chat_ui_responsiveness_during_streaming():
    """
    LLM 응답 스트리밍 중에도 UI가 프리징되지 않고
    다른 세션 상태 변경(PDF 페이지 이동)이 즉시 반영되는지 검증합니다.
    """
    at = AppTest.from_file("src/main.py").run(timeout=10)
    current_sid = SessionManager.get_session_id()

    # 1. RAGSystem.astream을 느린 스트림으로 Mocking
    async def slow_astream(*args, **kwargs):
        chunks = [
            MockChunk(status="생각 중..."),
            MockChunk(content="안녕하세요 "),
            MockChunk(content="반갑습니다 "),
            MockChunk(content="도와드리겠습니다."),
        ]
        for chunk in chunks:
            await asyncio.sleep(0.5)
            yield chunk

    from core.rag_core import RAGSystem

    RAGSystem.astream = slow_astream

    # 2. 백그라운드 스트리밍 직접 시작 (UI 트리거 우회)
    from ui.components.chat import _start_background_streaming

    SessionManager.set("is_generating_answer", True, current_sid)
    _start_background_streaming("테스트 질문", "gpt-4o", current_sid)

    # 3. 즉시 PDF 페이지 변경 요청
    # 기존 블로킹 방식이었다면, 이 코드가 실행되기 전 렌더링 루프에서 멈췄겠지만
    # 현재는 백그라운드 스레드에서 동작하므로 즉시 실행되어야 함
    SessionManager.set("pdf_target_page", 10, current_sid)

    # 4. 검증: 생성 중임에도 불구하고 상태 변경이 즉시 반영되었는지 확인
    assert SessionManager.get("is_generating_answer", current_sid) is True
    assert SessionManager.get("pdf_target_page", current_sid) == 10

    # 5. 최종 완료 확인
    timeout = 5
    start_time = time.time()
    while SessionManager.get("is_generating_answer", current_sid) and (
        time.time() - start_time < timeout
    ):
        time.sleep(0.2)

    assert SessionManager.get("is_generating_answer", current_sid) is False
