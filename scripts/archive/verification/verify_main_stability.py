"""
src/main.py의 안정성 및 무결성 수정 사항을 검증하는 스크립트
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import importlib

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

class TestMainStability(unittest.TestCase):
    def setUp(self):
        # Streamlit 및 관련 모듈 모킹
        self.st_mock = MagicMock()
        self.atexit_mock = MagicMock()
        self.threading_mock = MagicMock()
        self.scriptrunner_mock = MagicMock()
        
        # SessionState 모킹
        self.session_state = MagicMock()
        
        # 모듈 패치
        self.patches = [
            patch('streamlit.set_page_config', self.st_mock.set_page_config),
            patch('streamlit.cache_resource', lambda x: x),
            patch('streamlit.session_state', self.session_state),
            patch('atexit.register', self.atexit_mock.register),
            patch('threading.Thread', self.threading_mock.Thread),
            patch('streamlit.runtime.scriptrunner.add_script_run_ctx', self.scriptrunner_mock.add_script_run_ctx),
            patch('common.logging_config.setup_logging', return_value=MagicMock()),
            patch('nest_asyncio.apply', MagicMock())
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        if 'main' in sys.modules:
            del sys.modules['main']

    def test_atexit_cached_registration(self):
        """Task 2: atexit 등록이 함수 내에 있는지 확인"""
        import main
        # 모듈 임포트 시 1회 호출됨
        self.atexit_mock.register.assert_called_once()
        
        # 리셋 후 수동 호출 시 1회 더 호출되는지 확인 (구조적 확인)
        self.atexit_mock.register.reset_mock()
        main._register_cleanup_handlers()
        self.atexit_mock.register.assert_called_once()

    def test_thread_context_injection(self):
        """Task 3: 백그라운드 스레드에 컨텍스트가 주입되는지 확인"""
        import main
        
        # Mock SessionManager
        with patch('core.session.SessionManager') as sm:
            # new_file_uploaded가 True인 상황 시뮬레이션
            sm.get.side_effect = lambda k, **kwargs: True if k == "new_file_uploaded" else None
            sm.get_session_id.return_value = "test_session"
            
            # _handle_pending_tasks 실행
            main._handle_pending_tasks()
            
            # add_script_run_ctx가 호출되었는지 확인
            self.scriptrunner_mock.add_script_run_ctx.assert_called()
            
            # is_building_rag가 설정되었는지 확인
            sm.set.assert_any_call("is_building_rag", True)

if __name__ == "__main__":
    unittest.main()
