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

from core.session import SessionManager  # noqa: E402

# 횡단 의존성 방지: `pythonpath=["src"]` 환경에서 `import core` 와 `import
# src.core` 가 서로 다른 모듈 객체로 로드되어, 같은 클래스(`SessionManager` 등)가
# 테스트/소스별로 복제되고 전역 상태가 공유되지 않는 문제가 발생합니다.
# `src.core.*` 가 이미 로드되어 있다면 `core.*` 별칭이 동일 객체를 가리키도록
# sys.modules 를 병합합니다 (반대 방향도 보정).
import pkgutil

for _pkg in ("core", "ui", "common", "api", "infra", "security", "services"):
    _src_mod = sys.modules.get(f"src.{_pkg}")
    _mod = sys.modules.get(_pkg)
    if _src_mod is not None and _mod is not None and _src_mod is not _mod:
        # 두 별칭이 모두 로드된 경우 하나로 통일 (src.* 를 정통으로 취함)
        sys.modules[_pkg] = _src_mod
        # 하위 모듈도 재귀적으로 병합
        _prefix = f"src.{_pkg}."
        for _name, _modobj in list(sys.modules.items()):
            if _name.startswith(_prefix):
                _alias = _name[len("src.") :]
                if _alias in sys.modules and sys.modules[_alias] is not _modobj:
                    sys.modules[_alias] = _modobj


@pytest.fixture(autouse=True)
def _reset_session_manager_per_test():
    """테스트 간 전역 상태 누수를 차단합니다.

    SessionManager는 프로세스 전역 딕셔너리를 주 저장소로 사용하며, 일부 테스트는
    main.py import를 통해 UI-sync 어댑터(StreamlitSessionSync)를 전역에 부착합니다.
    어댑터가 남아 있으면 코어 로직(add_message 등)이 실제 UI session_state에
    의존하게 되어 테스트가 깨지므로, 각 테스트 전후로 폴백 저장소와 어댑터를
    모두 초기화합니다. 두 import 별칭(core / src.core)을 함께 정리합니다.
    """
    SessionManager.reset()
    SessionManager.set_ui_sync(None)
    yield
    SessionManager.reset()
    SessionManager.set_ui_sync(None)


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
