"""TASK 5 Wave2 — index-time annotation memo.

``extract_annotations_from_docs`` 의 on-demand fitz 경로에 memo-first 조회를
검증합니다: 1회차는 fitz를 열어 확정 라인박스를 저장하고, 동일 키 2회차는
fitz 오픈 없이 O(1) 적중해야 합니다. top-k=5 finalize는 50ms 미만입니다.
"""

import asyncio
import contextlib
import importlib
import time
from unittest.mock import patch

import pytest

from cache.coord_cache import coord_cache, memo_norm_hash
from common.utils import extract_annotations_from_docs

CONTENT_A = (
    "The retrieval pipeline extracts precise highlight coordinates for citations."
)
CONTENT_B = "Semantic chunking respects markdown headers and preserves section context."


@pytest.fixture
def isolated_memo_db(tmp_path, monkeypatch):
    """메모 SQLite를 tmp 경로로 격리합니다 (실 캐시 불변).

    ``coord_cache``는 프로세스 전역 싱글톤으로 이전 테스트의 aiosqlite
    연결(``_conn``)과 ``_schema_ready`` 플래그를 들고 있을 수 있습니다.
    경로만 monkeypatch하면 비동기 저장은 stale 연결(구 DB 파일)에 쓰고
    동기 조회는 새 tmp 파일을 열어 순서 의존 실패
    (``None == [{'page': 1}]``)가 발생하므로, 패치 전후로 연결을 닫고
    상태를 리셋해 격리합니다.
    """
    db_dir = tmp_path / "coord_cache"
    db_dir.mkdir()
    mod = importlib.import_module("cache.coord_cache")
    with contextlib.suppress(Exception):
        asyncio.run(coord_cache.close())
    # close()가 owner 루프 없이 조기 반환한 경우를 대비한 방어적 리셋.
    coord_cache._conn = None
    coord_cache._schema_ready = False
    coord_cache._conn_healthy = False
    monkeypatch.setattr(mod, "COORD_CACHE_DIR", db_dir)
    monkeypatch.setattr(mod, "COORD_CACHE_DB", db_dir / "coords.db")
    yield db_dir
    with contextlib.suppress(Exception):
        asyncio.run(coord_cache.close())


class _FakeQuad:
    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1


class _FakePage:
    def __init__(self, quads):
        self._quads = quads

    def get_textpage(self):
        return object()

    def search_for(self, part, textpage=None):
        assert len(part) >= 12
        return list(self._quads)


class _FakePdf:
    def __init__(self, page):
        self._page = page

    def __getitem__(self, idx):
        assert idx == 0
        return self._page


class _FitzSpy:
    """``open_pdf_document`` 대체 스파이 (오픈 횟수 계수)."""

    def __init__(self):
        self.opens = 0

    @contextlib.contextmanager
    def open(self, path):
        self.opens += 1
        page = _FakePage(
            [
                _FakeQuad(10.0, 20.0, 60.0, 30.0),
                _FakeQuad(65.0, 20.0, 110.0, 30.0),
            ]
        )
        yield _FakePdf(page)


def _doc(tmp_path, content, file_hash="memo-test-hash", page=1):
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    return {
        "page_content": content,
        "metadata": {
            "file_path": str(pdf),
            "file_hash": file_hash,
            "page": page,
        },
    }


def test_memo_norm_hash_contract():
    """Given 정규화 전후 텍스트 / When 해시 계산 / Then 대소문자·공백에
    불변하고 서로 다른 텍스트는 다른 키이며 Task3 키 규격(16hex)이다."""
    assert memo_norm_hash(CONTENT_A) == memo_norm_hash("  " + CONTENT_A.upper() + "\n")
    assert memo_norm_hash(CONTENT_A) != memo_norm_hash(CONTENT_B)
    assert memo_norm_hash("") == "0" * 16
    assert len(memo_norm_hash(CONTENT_A)) == 16


def test_memo_sync_roundtrip(isolated_memo_db):
    """Given 메모 저장 / When 동일 키 조회·다른 페이지 조회 / Then 적중은
    동일 페이로드를, 미스는 None을 반환한다."""
    annots = [{"page": 1, "x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}]
    assert coord_cache.save_annotation_memo_sync("h1", 2, "n1", annots) is True
    assert coord_cache.get_annotation_memo_sync("h1", 2, "n1") == annots
    assert coord_cache.get_annotation_memo_sync("h1", 3, "n1") is None
    assert coord_cache.get_annotation_memo_sync("h1", 2, "other") is None
    assert coord_cache.get_annotation_memo_sync("", 2, "n1") is None


def test_memo_async_api_compatible_with_sync(isolated_memo_db):
    """Given async 저장 / When sync 조회(또는 그 역) / Then 동일 키로 공유돼
    Task3 prefetch 경로와 키 호환된다."""

    async def _roundtrip():
        ok = await coord_cache.save_annotation_memo("h9", 1, "n9", [{"page": 1}])
        assert ok is True
        hit = await coord_cache.get_annotation_memo("h9", 1, "n9")
        assert hit == [{"page": 1}]

    asyncio.run(_roundtrip())
    try:
        assert coord_cache.get_annotation_memo_sync("h9", 1, "n9") == [{"page": 1}]
    finally:
        asyncio.run(coord_cache.close())


def test_second_finalize_zero_fitz_opens(isolated_memo_db, tmp_path):
    """Given 동일 키 문서 / When finalize 2회 / Then 2회차는 fitz 오픈 0회로
    1회차와 동일한 어노테이션을 반환한다."""
    spy = _FitzSpy()
    docs = [_doc(tmp_path, CONTENT_A)]
    with patch("common.utils.open_pdf_document", spy.open):
        first = extract_annotations_from_docs(docs)
    assert len(first) > 0
    assert spy.opens == 1

    with patch("common.utils.open_pdf_document", spy.open):
        second = extract_annotations_from_docs(docs)
    assert spy.opens == 1
    assert second == first


def test_memo_hit_topk5_under_50ms(isolated_memo_db, tmp_path):
    """Given top-k=5 문서 1회차 저장 / When fitz 호출 시 실패시키는 mock으로
    2회차 실행 / Then fitz 무오픈·동일 결과·50ms 미만이다."""
    spy = _FitzSpy()
    docs = [_doc(tmp_path, f"{CONTENT_A} variant {i}", page=1) for i in range(5)]
    with patch("common.utils.open_pdf_document", spy.open):
        expected = extract_annotations_from_docs(docs)
    assert len(expected) > 0
    # Task 4: 동일 PDF 5청크는 파일당 1회 오픈으로 처리된다.
    assert spy.opens == 1

    def _must_not_open(path):
        raise AssertionError("memo hit must not open fitz")

    with patch("common.utils.open_pdf_document", _must_not_open):
        start = time.perf_counter()
        actual = extract_annotations_from_docs(docs)
        elapsed = time.perf_counter() - start
    assert actual == expected
    assert elapsed < 0.05, f"memo hit {elapsed * 1000:.1f}ms exceeds 50ms"


def test_memo_miss_without_file_hash_falls_back_to_fitz(tmp_path):
    """Given file_hash 없는 문서 / When 추출 / Then 메모 없이 fitz 경로로
    정상 어노테이션을 반환한다."""
    spy = _FitzSpy()
    pdf = tmp_path / "nofile.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    docs = [
        {
            "page_content": CONTENT_B,
            "metadata": {"file_path": str(pdf), "page": 1},
        }
    ]
    with patch("common.utils.open_pdf_document", spy.open):
        annots = extract_annotations_from_docs(docs)
    assert len(annots) > 0
    assert spy.opens == 1


def test_memo_store_failure_does_not_break_extraction(
    isolated_memo_db, tmp_path, monkeypatch
):
    """Given 메모 저장 실패 / When 추출 / Then 어노테이션 결과는 그대로다."""

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(coord_cache, "save_annotation_memo_sync", _boom, raising=True)
    spy = _FitzSpy()
    docs = [_doc(tmp_path, CONTENT_B)]
    with patch("common.utils.open_pdf_document", spy.open):
        annots = extract_annotations_from_docs(docs)
    assert len(annots) > 0
