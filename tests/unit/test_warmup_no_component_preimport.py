"""부트 웜업 스레드가 Streamlit 커스텀 컴포넌트를 선점 임포트하지 않는지 검증.

회귀 배경: ``_warm_heavy_imports`` 가 백그라운드 스레드에서
``streamlit_pdf_viewer`` 를 임포트하면, 해당 스레드에는 ScriptRunContext 가
없어 ``declare_component`` 가 등록을 건너뛴다. 모듈 객체만 ``sys.modules`` 에
캐시되므로 이후 script 스레드의 lazy import 는 등록을 재시도하지 않고,
컴포넌트 경로가 404 가 되어 PDF 뷰어 iframe 이 비어버린다.
"""

from __future__ import annotations

import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import src.main as main_module

_COMPONENT_MODULE = "streamlit_pdf_viewer"


def _pop_component_modules() -> dict[str, ModuleType]:
    saved: dict[str, ModuleType] = {}
    for name in list(sys.modules):
        if name == _COMPONENT_MODULE or name.startswith(f"{_COMPONENT_MODULE}."):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_component_modules(saved: dict[str, ModuleType]) -> None:
    for name in list(sys.modules):
        if name == _COMPONENT_MODULE or name.startswith(f"{_COMPONENT_MODULE}."):
            del sys.modules[name]
    for name, module in saved.items():
        sys.modules[name] = module


@pytest.fixture
def isolated_component_modules():
    saved = _pop_component_modules()
    try:
        yield
    finally:
        _restore_component_modules(saved)


def test_heavy_import_warmup_does_not_import_pdf_viewer(
    isolated_component_modules: None,
) -> None:
    """웜업 스레드는 pdf viewer 모듈을 sys.modules 에 캐시하면 안 된다."""
    assert _COMPONENT_MODULE not in sys.modules

    worker = threading.Thread(
        target=main_module._warm_heavy_imports,
        name="heavy-import-warmup",
        daemon=True,
    )
    worker.start()
    worker.join(timeout=30)

    assert not worker.is_alive(), "웜업 스레드가 30초 내 종료되지 않았다"
    assert _COMPONENT_MODULE not in sys.modules, (
        "백그라운드 웜업이 pdf viewer 를 선점 임포트하면 ScriptRunContext 부재로 "
        "컴포넌트가 미등록된 채 모듈만 캐시되어 뷰어가 404 가 된다"
    )


def test_declare_component_without_script_context_is_not_registered(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """컴포넌트 등록은 ScriptRunContext 가 있을 때만 일어난다(회귀 전제)."""
    from streamlit.components.v1 import component_registry

    registry = SimpleNamespace(component_registry=MagicMock())
    monkeypatch.setattr(component_registry, "get_instance", lambda: registry)
    monkeypatch.setattr(component_registry, "get_script_run_ctx", lambda: None)

    component_registry.declare_component("regression_probe", path=str(tmp_path))

    registry.component_registry.register_component.assert_not_called()


def test_declare_component_with_script_context_is_registered(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """script 스레드에서의 임포트는 정상 등록된다(양성 경로)."""
    from streamlit.components.v1 import component_registry

    registry = SimpleNamespace(component_registry=MagicMock())
    monkeypatch.setattr(component_registry, "get_instance", lambda: registry)
    monkeypatch.setattr(component_registry, "get_script_run_ctx", lambda: object())

    component_registry.declare_component("regression_probe", path=str(tmp_path))

    registry.component_registry.register_component.assert_called_once()
