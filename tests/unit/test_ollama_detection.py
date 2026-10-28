import pytest
from core._loaders import _is_ollama_server_up
import core._loaders as loaders


def test_is_ollama_server_up_fail_closed_on_cold_start(monkeypatch):
    monkeypatch.setattr(loaders, "_models_cache", None)
    monkeypatch.setattr(loaders, "OLLAMA_BASE_URL", "http://127.0.0.1:59999")

    is_up = _is_ollama_server_up()
    assert is_up is False, (
        "Ollama가 닫혀있으나 최초 실행 시 True로 오판함 (Fail-open 결함)"
    )
