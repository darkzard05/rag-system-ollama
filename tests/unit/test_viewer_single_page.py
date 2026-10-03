"""단일 페이지 PDF 추출 및 주석 범위 지정 검증."""

import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pymupdf
import pytest

import ui.components.viewer as viewer_mod
from core.session import SessionManager
from ui.components.viewer import (
    _get_pdf_page_bytes,
    _page_scoped_annotations,
    render_pdf_viewer,
)

_TOKEN = "CITATION-TOKEN-ZXQJ-9173"


def _make_two_page_pdf(tmp_path: Path, token: str = _TOKEN) -> str:
    """2쪽 테스트 PDF를 만들고 경로를 반환한다."""
    path = str(tmp_path / "sample.pdf")
    doc = pymupdf.open()
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    doc[0].insert_text((72, 100), "PAGE ONE FILLER TEXT ALPHA")
    doc[1].insert_text((72, 200), f"PAGE TWO MARKER {token} TAIL")
    doc.save(path)
    doc.close()
    return path


def test_extract_preserves_rect_roundtrip(tmp_path: Path) -> None:
    """추출 전후 search_for Rect가 동일함을 증명한다."""
    path = _make_two_page_pdf(tmp_path)
    with pymupdf.open(path) as src:
        original = src[1].search_for(_TOKEN)
        assert len(original) == 1
        original_rect = original[0]
    page_bytes = _get_pdf_page_bytes(path, "hash-a", 2)
    assert page_bytes
    with pymupdf.open(stream=page_bytes, filetype="pdf") as single:
        assert len(single) == 1
        found = single[0].search_for(_TOKEN)
        assert len(found) == 1
        assert tuple(found[0]) == tuple(original_rect)


def _make_pdf_with_images(tmp_path: Path, pages: int, images_per_page: int = 6) -> str:
    """페이지마다 여러 이미지를 넣어 공유 리소스가 있는 PDF를 만든다."""
    path = str(tmp_path / "images.pdf")
    doc = pymupdf.open()
    for i in range(pages):
        doc.new_page(width=595, height=842)
        for j in range(images_per_page):
            # 페이지마다 고유한 픽셀 패턴: 서로 다른 이미지 객체가 되게 한다
            pix = pymupdf.Pixmap(
                pymupdf.csRGB,
                pymupdf.IRect(0, 0, 40 + i, 40 + j),
                False,
            )
            pix.set_rect(
                pix.irect, ((i * 37 + j * 11) % 256, (j * 53) % 256, (i * 97) % 256)
            )
            doc[i].insert_image(
                pymupdf.Rect(40 + j * 30, 60, 90 + j * 30, 110), pixmap=pix
            )
    doc.save(path)
    doc.close()
    return path


def test_extract_actually_shrinks_payload(tmp_path: Path) -> None:
    """선택 페이지보다 훨씬 커야 하고(공유 리소스), 실제로 크게 줄어야 한다.

    select()만 쓰면 페이지 트리는 줄어도 다른 페이지의 이미지 객체가 그대로
    직렬화되어 0.7% 감소에 그친다. garbage=4가 없으면 이 어서션이 실패한다.
    """
    path = _make_pdf_with_images(tmp_path, pages=10)
    with open(path, "rb") as f:
        full_size = len(f.read())

    page_bytes = _get_pdf_page_bytes(path, "hash-shrink", 5)
    assert page_bytes

    # 공유 리소스가 있는 구조에서 선택본이 원본과 "비슷한 크기"면 GC가 안 돌아간 것
    assert len(page_bytes) < full_size * 0.5, (
        f"extraction did not shrink payload: {len(page_bytes):,} vs {full_size:,}"
    )


def test_extract_preserves_images_on_selected_page(tmp_path: Path) -> None:
    """가비지 컬렉션 후에도 선택 페이지의 이미지 개수/크기가 보존된다."""
    path = _make_pdf_with_images(tmp_path, pages=6, images_per_page=4)
    with pymupdf.open(path) as src:
        original_images = src[3].get_images(full=True)
        assert original_images

    page_bytes = _get_pdf_page_bytes(path, "hash-img", 4)
    with pymupdf.open(stream=page_bytes, filetype="pdf") as single:
        extracted_images = single[0].get_images(full=True)

    assert len(extracted_images) == len(original_images)
    # xref 번호는 GC로 재부여되므로 픽셀 차원과 바이트 크기로 비교한다
    assert sorted(i[2:4] for i in extracted_images) == sorted(
        i[2:4] for i in original_images
    )


def test_get_pdf_page_bytes_valid_page(tmp_path: Path) -> None:
    """유효 페이지는 1쪽짜리 문서를 반환한다."""
    path = _make_two_page_pdf(tmp_path)
    page_bytes = _get_pdf_page_bytes(path, "hash-a", 1)
    assert page_bytes
    with pymupdf.open(stream=page_bytes, filetype="pdf") as single:
        assert len(single) == 1


def test_get_pdf_page_bytes_out_of_range(tmp_path: Path) -> None:
    """범위 밖 페이지는 빈 바이트를 반환한다."""
    path = _make_two_page_pdf(tmp_path)
    assert _get_pdf_page_bytes(path, "hash-a", 3) == b""
    assert _get_pdf_page_bytes(path, "hash-a", 99) == b""


def test_get_pdf_page_bytes_missing_path(tmp_path: Path) -> None:
    """없는 경로는 빈 바이트를 반환한다."""
    missing = str(tmp_path / "nope.pdf")
    assert _get_pdf_page_bytes(missing, "hash-a", 1) == b""


def test_get_pdf_page_bytes_page_zero(tmp_path: Path) -> None:
    """page=0은 빈 바이트를 반환한다."""
    path = _make_two_page_pdf(tmp_path)
    assert _get_pdf_page_bytes(path, "hash-a", 0) == b""
    assert _get_pdf_page_bytes("", "hash-a", 1) == b""


def test_page_scoped_keeps_matching_and_remaps() -> None:
    """현재 페이지만 남기고 page를 1로 바꾼다."""
    annotations = [
        {"page": 1, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
        {"page": 2, "x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0},
    ]
    scoped = _page_scoped_annotations(annotations, 2)
    assert len(scoped) == 1
    assert scoped[0]["page"] == 1
    assert scoped[0]["x"] == 5.0


def test_page_scoped_preserves_coords_and_no_mutate() -> None:
    """좌표/색상을 verbatim 유지하고 입력을 변경하지 않는다."""
    item = {
        "page": 2,
        "x": 72.5,
        "y": 200.25,
        "width": 120.0,
        "height": 14.5,
        "color": "red",
        "extra": "keep-me",
    }
    snapshot = dict(item)
    scoped = _page_scoped_annotations([item], 2)
    assert len(scoped) == 1
    moved = scoped[0]
    assert moved["page"] == 1
    assert moved["x"] == 72.5
    assert moved["y"] == 200.25
    assert moved["width"] == 120.0
    assert moved["height"] == 14.5
    assert moved["color"] == "red"
    assert moved["extra"] == "keep-me"
    assert item == snapshot
    assert scoped[0] is not item


def test_page_scoped_skips_invalid() -> None:
    """비 dict와 page 강제변환 실패 항목을 건너뛴다."""
    annotations = [
        "not-a-dict",
        None,
        42,
        {"x": 1.0},
        {"page": None, "x": 1.0},
        {"page": "nan!", "x": 1.0},
        {"page": "2", "x": 9.0, "y": 9.0, "width": 1.0, "height": 1.0},
    ]
    scoped = _page_scoped_annotations(annotations, 2)
    assert len(scoped) == 1
    assert scoped[0]["page"] == 1
    assert scoped[0]["x"] == 9.0


def test_page_scoped_empty() -> None:
    """빈 입력은 빈 리스트를 반환한다."""
    assert _page_scoped_annotations([], 1) == []


def _ensure_pdf_viewer_module(
    monkeypatch: pytest.MonkeyPatch, stub: Callable[..., None]
) -> Any:
    """streamlit_pdf_viewer 모듈의 pdf_viewer를 기록 스텁으로 교체한다."""
    try:
        import streamlit_pdf_viewer
    except ImportError:
        module = types.ModuleType("streamlit_pdf_viewer")
        module.__dict__["pdf_viewer"] = stub
        sys.modules["streamlit_pdf_viewer"] = module
        return module
    monkeypatch.setattr(streamlit_pdf_viewer, "pdf_viewer", stub)
    return streamlit_pdf_viewer


def test_render_success_single_page(monkeypatch, tmp_path: Path) -> None:
    """추출 성공 시 pages_to_render=[1]과 재매핑 주석을 전달한다."""
    path = _make_two_page_pdf(tmp_path)
    original = [
        {"page": 2, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0},
        {"page": 1, "x": 5.0, "y": 6.0, "width": 7.0, "height": 8.0},
    ]
    SessionManager.set("pdf_annotations", [dict(item) for item in original])
    calls: dict = {}

    def stub(**kwargs: Any) -> None:
        calls.update(kwargs)
        return None

    _ensure_pdf_viewer_module(monkeypatch, stub)
    monkeypatch.setattr(viewer_mod, "_get_pdf_page_bytes", lambda p, h, pg: b"P1")

    def _fail_full(path_arg: str) -> bytes:
        raise AssertionError("성공 경로는 전체 바이트를 읽으면 안 됨")

    monkeypatch.setattr(viewer_mod, "_get_pdf_bytes", _fail_full)
    render_pdf_viewer(path, 2, "hash123")
    assert calls["input"] == b"P1"
    assert calls["pages_to_render"] == [1]
    assert calls["annotations"] == [
        {"page": 1, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}
    ]
    assert calls["render_text"] is True
    assert calls["annotation_outline_size"] == 2
    assert calls["scroll_behavior"] == "instant"


def test_render_fallback_full_bytes(monkeypatch, tmp_path: Path) -> None:
    """추출 실패 시 원본 주석과 함께 전체 바이트로 폴백한다."""
    path = _make_two_page_pdf(tmp_path)
    original = [{"page": 2, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}]
    SessionManager.set("pdf_annotations", [dict(item) for item in original])
    calls: dict = {}

    def stub(**kwargs: Any) -> None:
        calls.update(kwargs)
        return None

    _ensure_pdf_viewer_module(monkeypatch, stub)
    monkeypatch.setattr(viewer_mod, "_get_pdf_page_bytes", lambda p, h, pg: b"")
    monkeypatch.setattr(viewer_mod, "_get_pdf_bytes", lambda p: b"FULL")
    render_pdf_viewer(path, 2, "hash123")
    assert calls["input"] == b"FULL"
    assert calls["pages_to_render"] == [2]
    assert calls["annotations"] == original
