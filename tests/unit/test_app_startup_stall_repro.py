"""RED→GREEN regression tests for the P0 cold-start stall.

1. Model-list lookup must never block the Streamlit render thread on a cache miss.
2. The maintenance worker must not unload VRAM during the boot grace window
   (racing the startup warmup).
3. Importing ``core.document_processor`` must not eagerly import OCR engines.
"""

import sys
import time

from unittest.mock import patch


def test_fetch_available_models_non_blocking_on_cache_miss():
    """캐시 부재 시 메인 스레드는 동기 network I/O를 수행하지 않아야 함."""
    from core import _loaders

    _loaders._models_cache = None
    _loaders._models_fetching_event.clear()
    if _loaders._models_cache_lock.locked():
        _loaders._models_cache_lock.release()

    with patch("ollama.Client") as mock_client:

        def slow_list(*args, **kwargs):
            time.sleep(2.0)
            return {"models": []}

        mock_client.return_value.list.side_effect = slow_list

        t0 = time.perf_counter()
        models = _loaders._fetch_available_models_cached(non_blocking_fallback=True)
        elapsed = time.perf_counter() - t0

        assert elapsed < 0.1, f"Main thread blocked for {elapsed:.2f}s on model fetch"
        assert len(models) >= 1


def test_maintenance_worker_does_not_clear_vram_on_immediate_startup():
    """백그라운드 워커는 부팅 유예 중 clear_vram을 실행하면 안 됨."""
    from main import _check_windows_integrity

    from core.session import SessionManager

    SessionManager.reset()
    assert SessionManager.get_stats().get("active_sessions", 0) == 0

    with patch("core.model_loader.ModelManager.clear_vram") as mock_clear:
        _check_windows_integrity(uptime_seconds=5.0)
        mock_clear.assert_not_called()


def test_ocr_modules_are_not_eagerly_imported():
    """document_processor 로드 시 무거운 OCR 엔진이 즉시 로드되면 안 됨."""
    ocr_modules = [
        "pymupdf4llm.ocr.rapidocr_api",
        "pymupdf4llm.ocr.paddleocr_api",
        "rapidocr_onnxruntime",
    ]
    for mod in ocr_modules:
        sys.modules.pop(mod, None)

    import core.document_processor  # noqa: F401

    for mod in ocr_modules:
        assert mod not in sys.modules, f"Heavy OCR module eagerly imported: {mod}"
