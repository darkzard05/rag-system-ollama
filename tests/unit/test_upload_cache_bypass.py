"""Upload cache bypass — keep the upload callback off the disk-hash meat grinder.

``_find_cached_upload`` used to glob every ``data/temp/upload_*.pdf`` and run a
full SHA-256 pass over every size-matching candidate on the UI thread. The
legacy rescan is now gone entirely: only the content-addressed
``upload_{hash}.pdf`` is probed (O(1) stat), and orphans are reclaimed by the
background sweep. These tests pin that contract:

* a content-addressed hit must return before any glob;
* a miss must not glob and must not hash any candidate.
"""

from pathlib import Path
from unittest.mock import patch

from src.main import _find_cached_upload

from core.document_processor import compute_file_hash


def test_find_cached_upload_immediate_hit(tmp_path: Path):
    """표준 파일명이 일치하면 디스크 glob 스캔 없이 O(1)로 즉시 반환되는지 검증."""
    content = b"%PDF-1.4 sample content for hash test"
    fhash = compute_file_hash("", data=content)
    target_file = tmp_path / f"upload_{fhash}.pdf"
    target_file.write_bytes(content)

    with patch("pathlib.Path.glob") as mock_glob:
        res = _find_cached_upload(str(tmp_path), content, fhash)
        assert res == str(target_file)
        # glob 스캔이 호출되지 않아야 함
        mock_glob.assert_not_called()


def test_find_cached_upload_never_rescans_legacy_files(tmp_path: Path):
    """동일 크기 legacy 파일이 많아도 재해시/재스캔하지 않는지 검증.

    이 경로는 업로드 콜백에서 실행되므로 파일 크기에 비례하는 I/O(43MB 기준
    실측 258ms)를 허용하지 않는다.
    """
    content = b"%PDF-1.4 dummy content"
    fhash = "non_existent_hash_123456"

    for i in range(10):
        (tmp_path / f"upload_legacy_{i}.pdf").write_bytes(b"%PDF-1.4 other content")

    hash_calls = []
    real_compute_hash = compute_file_hash

    def spy_compute_hash(path_str: str, **kwargs):
        hash_calls.append(path_str)
        return real_compute_hash(path_str, **kwargs)

    with (
        patch("pathlib.Path.glob") as mock_glob,
        patch(
            "core.document_processor.compute_file_hash", side_effect=spy_compute_hash
        ),
    ):
        res = _find_cached_upload(str(tmp_path), content, fhash)
        assert res is None
        assert hash_calls == [], f"legacy 재해시가 발생함: {hash_calls}"
        mock_glob.assert_not_called()
