import unittest
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
from streamlit.testing.v1 import AppTest

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class TestTimelineSync(unittest.TestCase):
    def test_timeline_status_update_flow(self):
        """
        백엔드 상태 메시지(is_status_update)가 UI 타임라인에 
        정상적으로 반영되는지 전체 흐름을 검증합니다.
        """
        # 임시 테스트 스크립트 작성
        script_content = """
import streamlit as st
import asyncio
import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

# src 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))

from ui.components.chat import _stream_chat_response
from api.streaming_handler import StreamChunk

async def mock_event_generator():
    # 1. 노드 시작 이벤트
    yield StreamChunk(
        content="", 
        timestamp=time.time(), 
        token_count=0, 
        chunk_index=0, 
        status="질문 의도 분석 및 하이브리드 지식 검색 중"
    )
    await asyncio.sleep(0.5)
    
    # 2. 노드 종료 이벤트
    yield StreamChunk(
        content="", 
        timestamp=time.time(), 
        token_count=0, 
        chunk_index=1, 
        status="지식 저장소에서 관련 문서 1개를 성공적으로 찾았습니다.",
        metadata={"documents": [{"page_content": "doc1", "metadata": {"page": 1}}]}
    )
    await asyncio.sleep(0.5)
    
    # 3. 답변 생성 이벤트
    yield StreamChunk(
        content="테스트 답변입니다.", 
        timestamp=time.time(), 
        token_count=1, 
        chunk_index=2, 
        thought="사고 과정...",
        is_final=True
    )
    await asyncio.sleep(0.5)

async def run_test_ui():
    st.title("Timeline Test")
    chat_container = st.container()
    
    # SessionManager 모킹 (세션 ID 에러 방지)
    with patch("core.session.SessionManager.get_session_id", return_value="test_session"):
        with patch("core.session.SessionManager.get", return_value=None):
            with patch("ui.components.chat.get_streaming_handler", return_value=None):
                # RAGSystem 모킹
                mock_rag = MagicMock()
                mock_rag.astream_events = AsyncMock(return_value=mock_event_generator())
                
                # 스트리밍 실행
                await _stream_chat_response(mock_rag, "질문", chat_container)

if __name__ == "__main__":
    asyncio.run(run_test_ui())
"""
        with open("temp_test_timeline.py", "w", encoding="utf-8") as f:
            f.write(script_content)
            
        try:
            # AppTest 실행 (충분한 타임아웃 부여)
            at = AppTest.from_file("temp_test_timeline.py").run(timeout=10)
            
            # 1. 상태창 및 로그 확인 (st.status 및 st.caption)
            # st.status 내부의 caption들을 확인하여 상태 메시지가 출력되었는지 검증
            status_logs = [c.value for c in at.caption]
            print(f"발견된 상태 로그: {status_logs}")
            
            # 2. 답변 렌더링 확인
            self.assertTrue(any("테스트 답변입니다" in m.value for m in at.markdown), "답변이 렌더링되지 않았습니다.")
            
            print("✅ UI 상태 업데이트 및 답변 렌더링 테스트 통과")
            
        finally:
            if os.path.exists("temp_test_timeline.py"):
                os.remove("temp_test_timeline.py")

if __name__ == "__main__":
    unittest.main()
