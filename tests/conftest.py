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

# 횡단 의존성 방지: `sys.path` 에 `src` 가 들어가면 `import core` 와
# `import src.core` 가 서로 다른 모듈 객체로 로드되어, 같은 클래스
# (`SessionManager` 등)나 함수(`host_pressure_exceeded` 등)가 테스트/소스별로
# 복제되고 전역 상태·monkeypatch 타깃이 불일치하는 문제가 발생합니다.
#
# 소스 모듈(`src.main` 등)은 언프리픽스 `core.*` 를 사용하므로, canonical 을
# `src.core.*` 로 고정하고 `core.*` 별칭이 **항상 동일 객체**를 가리키도록
# 강제합니다. conftest 가 먼저 로드되므로 이후 어떤 테스트가 `import src.core.x`
# 를 해도 `core.x` 는 같은 객체입니다. 하위 모듈까지 전부 매핑합니다.
for _name, _modobj in list(sys.modules.items()):
    if _name.startswith("src.") and not _name.startswith("src.src."):
        _alias = _name[len("src.") :]
        # src.pkg.x -> pkg.x 별칭을 같은 객체로 (이미 있으면 덮어쓰지 않음)
        if _alias not in sys.modules:
            sys.modules[_alias] = _modobj

# 아직 로드되지 않은 하위 모듈도 future import 시 동일 객체를 쓰도록,
# 이미 로드된 core.* 가 src.* 와 다르면 src.* 를 정통으로 통일.
for _pkg in ("core", "ui", "common", "api", "infra", "security", "services"):
    _src_name = f"src.{_pkg}"
    _src_mod = sys.modules.get(_src_name)
    _mod = sys.modules.get(_pkg)
    if _src_mod is not None and _mod is not None and _src_mod is not _mod:
        sys.modules[_pkg] = _src_mod
        sys.modules[_src_name] = _src_mod


def _current_session_manager():
    """Return the SessionManager class that is *actually* bound to ``core.session``
    right now.

    Some tests (e.g. ``test_core_imports_without_streamlit``) pop ``core.session``
    from ``sys.modules`` to force a fresh re-import, which creates a *new*
    ``SessionManager`` class object. A module-level cached reference would then
    point at the stale (pre-pop) class while the rest of the code uses the new
    one, so state no longer shared. Re-importing here always yields the live
    class, whatever alias the caller uses.
    """
    from core.session import SessionManager

    return SessionManager


@pytest.fixture(autouse=True)
def _reset_session_manager_per_test():
    """테스트 간 전역 상태 누수를 차단합니다.

    SessionManager는 프로세스 전역 딕셔너리를 주 저장소로 사용하며, 일부 테스트는
    main.py import를 통해 UI-sync 어댑터(StreamlitSessionSync)를 전역에 부착합니다.
    어댑터가 남아 있으면 코어 로직이 실제 UI session_state에 의존하게 되어
    테스트가 깨지므로, 각 테스트 전후로 폴백 저장소와 어댑터를 모두 초기화합니다.

    ``_current_session_manager()`` 를 통해 **현재 유효한** SessionManager 클래스를
    매번 다시 가져오므로, 다른 테스트가 ``sys.modules`` 를 더럽혀 새 클래스 객체를
    만들었더라도 그 객체의 전역 상태를 정리할 수 있습니다.
    """
    SessionManager = _current_session_manager()
    SessionManager.reset()
    SessionManager.set_ui_sync(None)
    yield
    SessionManager = _current_session_manager()
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
    """프리로드 스케줄링이 pytest-asyncio의 함수 스코프 루프 교체와 충돌해
    모델 Lock을 잡은 태스크가 좌초되는 데드락을 원천 차단합니다.

    소스의 실제 가드(pipeline_builder._schedule_model_preload)는
    ``IS_UNIT_TEST``/``IS_CI_TEST`` 환경변수이므로, 세션 전역으로 그 환경변수를
    설정해 스케줄 태스크 생성을 막고 세션 종료 시 원값을 복원합니다.
    """
    import os

    prev = os.environ.get("IS_UNIT_TEST")
    os.environ["IS_UNIT_TEST"] = "true"
    yield
    if prev is None:
        os.environ.pop("IS_UNIT_TEST", None)
    else:
        os.environ["IS_UNIT_TEST"] = prev
