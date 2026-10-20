"""Upload cache bypass — keep the upload callback off the disk-hash meat grinder.

``_find_cached_upload`` used to glob every ``data/temp/upload_*.pdf`` and run a
full SHA-256 pass over every size-matching candidate on the UI thread. These
tests pin the two bounds that remove that I/O:

* a content-addressed ``upload_{hash}.pdf`` hit must return before any glob;
* a legacy miss must hash at most three size-matching candidates.
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


def test_find_cached_upload_limits_hash_computations_on_miss(tmp_path: Path):
    """크기가 일치하는 레거시 파일이 많아도 최대 3개까지만 해시를 검사하는지 검증."""
    content = b"%PDF-1.4 dummy content"
    fhash = "non_existent_hash_123456"

    # 동일 크기의 더미 파일 10개 생성
    for i in range(10):
        (tmp_path / f"upload_legacy_{i}.pdf").write_bytes(b"%PDF-1.4 other content")

    hash_calls = []
    real_compute_hash = compute_file_hash

    def spy_compute_hash(path_str: str, **kwargs):
        hash_calls.append(path_str)
        return real_compute_hash(path_str, **kwargs)

    with patch(
        "core.document_processor.compute_file_hash", side_effect=spy_compute_hash
    ):
        res = _find_cached_upload(str(tmp_path), content, fhash)
        assert res is None
        assert len(hash_calls) <= 3, f"해시 검사가 3회를 초과함: {len(hash_calls)}회"
