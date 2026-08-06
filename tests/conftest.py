"""
Shared pytest fixtures for RAG System tests.
"""

import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

# per-process isolation: 모듈 레벨 auth_manager/기본 AuthenticationManager 가
# 실제 .model_cache 파일을 건드리지 않도록 임시 디렉터리로 격리합니다.
_AUTH_TEST_DIR = tempfile.mkdtemp(prefix="rag_ollama_auth_")
os.environ["AUTH_STATE_FILE"] = os.path.join(_AUTH_TEST_DIR, "auth_state.json")
os.environ["AUTH_SECRET_FILE"] = os.path.join(_AUTH_TEST_DIR, ".jwt_secret")

# Add src to sys.path
BASE_DIR = Path(__file__).parent.parent.absolute()
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture
def mock_llm():
    """Returns a mock LLM for testing."""
    from unittest.mock import MagicMock

    return MagicMock()


@pytest.fixture
def session_context():
    return f"test_session_{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session", autouse=True)
def _disable_preload_in_tests():
    """pytest-asyncio의 함수 스코프 루프 교체로 프리로드 태스크가 락을 잡고 좌초되는 데드락을 차단합니다."""
    import core.pipeline_builder as pb

    pb._preload_scheduled = True
    return
