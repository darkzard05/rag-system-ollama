"""viewer PDF 캐시 키 계약 검증 — 빈 결과는 캐시하지 않는다.

지연 저장 도입 이후 파일이 아직 없을 때의 b"" 가 300초 캐시에 고정되면
세션 전체가 PDF를 못 읽는다. 따라서 실제 파일 읽기/페이지 추출을 수행하는
내부 캐시 함수만 path+mtime 기준으로 캐시하고, 외부 래퍼는 stat 으로 존재를
확인한 뒤에만 캐시 함수를 호출한다.
"""

import functools
import inspect
import os
from collections.abc import Callable
from unittest import mock

import pymupdf as fitz

import ui.components.viewer as viewer_module
from common.utils import safe_cache_data


def _make_pdf(tmp_path, name: str = "sample.pdf", pages: int = 2) -> str:
    path = str(tmp_path / name)
    doc = fitz.open()
    for _ in range(pages):
        doc.new_page()
    doc.save(path)
    doc.close()
    return path


def _wrap_for_cache(raw: Callable) -> Callable:
    """프로덕션 장식(safe_cache_data under Streamlit runtime)을 재현한다.

    유닛 테스트 프로세스에는 Streamlit 런타임이 없어 모듈 레벨 장식이 원본
    함수로 폴백된다. 실제 st.cache_data 적중/실패는 Streamlit 도메인이라
    재검증하지 않고, canonical한 인자 키 캐시(functools.lru_cache)로 대체해
    WRAPPER 계약(경로+mtime 키 집계)을 잠근다.
    """

    def fake_cache_data(**kwargs):
        def deco(f):
            return functools.lru_cache(maxsize=None)(f)

        return deco

    with (
        mock.patch("streamlit.runtime.exists", return_value=True),
        mock.patch("streamlit.cache_data", fake_cache_data),
    ):
        return safe_cache_data(ttl=300, show_spinner=False)(raw)


def test_pdf_bytes_cached_per_path_and_mtime(tmp_path) -> None:
    """동일 pdf_path+mtime 은 디스크 1회 읽기, 다른 경로는 별도 캐시."""
    first_path = _make_pdf(tmp_path, "a.pdf")
    second_path = _make_pdf(tmp_path, "b.pdf")
    cached = _wrap_for_cache(viewer_module._read_pdf_bytes_cached)

    reads: list[str] = []
    real_open = open

    def counting_open(file, mode="r", *args, **kwargs):
        if mode == "rb":
            reads.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    first_mtime = os.stat(first_path).st_mtime_ns
    with mock.patch("builtins.open", counting_open):
        first = cached(first_path, first_mtime)
        again = cached(first_path, first_mtime)
    assert first == again
    assert reads == [first_path], "동일 경로+mtime 반복 호출은 디스크를 1회만 읽어야 함"

    reads.clear()
    with mock.patch("builtins.open", counting_open):
        cached(second_path, os.stat(second_path).st_mtime_ns)
    assert reads == [second_path], "다른 경로는 별도 캐시 엔트리(재읽기)여야 함"

    # 외부 래퍼는 파일이 없으면 캐시 함수를 호출하지 않고 b"" 를 반환한다.
    with mock.patch.object(viewer_module, "_read_pdf_bytes_cached") as mock_cached:
        assert viewer_module._get_pdf_bytes(str(tmp_path / "missing.pdf")) == b""
        mock_cached.assert_not_called()


def test_pdf_cache_key_is_path_and_mtime_not_page() -> None:
    """내부 캐시 키는 pdf_path+mtime 이며 페이지/뷰어 키에 의존하지 않는다."""
    func = viewer_module._read_pdf_bytes_cached
    raw = getattr(func, "__wrapped__", func)
    sig = inspect.signature(raw)
    assert list(sig.parameters) == ["pdf_path", "mtime_ns"]
    forbidden = {"current_page", "viewer_key", "page"}
    assert forbidden.isdisjoint(sig.parameters)

    decorator = inspect.getsource(raw).splitlines()[0]
    assert decorator.startswith("@safe_cache_data")
    assert "ttl=300" in decorator

    # 외부 래퍼는 장식되지 않아야 한다 (미존재 b"" 캐시 오염 원천 차단).
    assert not hasattr(viewer_module._get_pdf_bytes, "__wrapped__")
