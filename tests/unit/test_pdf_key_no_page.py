"""B13 future-tracking: _get_pdf_bytes 캐시 키 계약 검증.

PDF 바이트는 pdf_path 단위로만 캐시되어야 한다 (NB4). current_page /
viewer_key 같은 페이지·뷰어 키에 의존해서는 안 되며, 향후 컴포넌트
업그레이드가 per-page 캐싱을 도입할 때 이 테스트가 회귀를 잡는다.
"""

import functools
import inspect
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


def _wrap_for_cache(raw: Callable[[str], bytes]) -> Callable[[str], bytes]:
    """프로덕션 장식(safe_cache_data under Streamlit runtime)을 재현한다.

    유닛 테스트 프로세스에는 Streamlit 런타임이 없어 모듈 레벨 장식이 원본
    함수로 폴백된다(raw — safe_cache_data는 st.runtime.exists()가 False면
    원본을 그대로 반환). 또한 실제 st.cache_data는 런타임 밖에서 호출 시
    호출마다 fresh MemoryCacheStorageManager를 생성해 재전송을 막지 못한다.
    따라서 런타임 존재와 cache API를 함께 모킹하고 — canonical한 인자 키
    캐시(functools.lru_cache)로 대체해 — B13의 WRAPPER 계약(경로 키 집계)을
    잠근다. 실제 st.cache_data 적중/실패는 Streamlit 도메인이라 재검증하지
    않는다.
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


def test_pdf_bytes_cached_per_path_not_page(tmp_path) -> None:
    """NB4 계약: 동일 pdf_path 는 디스크 1회 읽기, 다른 경로는 별도 캐시."""
    first_path = _make_pdf(tmp_path, "a.pdf")
    second_path = _make_pdf(tmp_path, "b.pdf")
    cached = _wrap_for_cache(viewer_module._get_pdf_bytes)

    reads: list[str] = []
    real_open = open

    def counting_open(file, mode="r", *args, **kwargs):
        if mode == "rb":
            reads.append(str(file))
        return real_open(file, mode, *args, **kwargs)

    with mock.patch("builtins.open", counting_open):
        first = cached(first_path)
        again = cached(first_path)
    assert first == again
    assert reads == [first_path], "동일 경로 반복 호출은 디스크를 1회만 읽어야 함"

    reads.clear()
    with mock.patch("builtins.open", counting_open):
        cached(second_path)
    assert reads == [second_path], "다른 경로는 별도 캐시 엔트리(재읽기)여야 함"

    # 오류 소화 의미론 유지: 없는 파일 → b"" (빈 값도 경로 키로 캐시됨)
    assert cached(str(tmp_path / "missing.pdf")) == b""


def test_pdf_cache_key_not_page_dependent() -> None:
    """future-tracking: 키는 pdf_path 단일 인자 — 페이지/뷰어 키 금지."""
    func = viewer_module._get_pdf_bytes
    raw = getattr(func, "__wrapped__", func)
    sig = inspect.signature(raw)
    assert list(sig.parameters) == ["pdf_path"]
    forbidden = {"current_page", "viewer_key", "page"}
    assert forbidden.isdisjoint(sig.parameters)

    decorator = inspect.getsource(raw).splitlines()[0]
    assert decorator.startswith("@safe_cache_data")
    assert "ttl=300" in decorator
