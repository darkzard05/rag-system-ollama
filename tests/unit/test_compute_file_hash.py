"""compute_file_hash 공용 함수 동작 검증 (R: Group 4 중복 통합)."""

import hashlib

import pytest

from common.utils import compute_file_hash


def test_hash_of_data(tmp_path):
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_file_hash("unused", data=data) == expected


def test_hash_of_file(tmp_path):
    p = tmp_path / "file.bin"
    p.write_bytes(b"file content")
    expected = hashlib.sha256(b"file content").hexdigest()
    assert compute_file_hash(str(p)) == expected


def test_algorithm_param(tmp_path):
    p = tmp_path / "file.txt"
    p.write_bytes(b"data")
    expected = hashlib.md5(b"data").hexdigest()
    assert compute_file_hash(str(p), algorithm="md5") == expected


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        compute_file_hash("definitely/not/here.bin")


def test_data_takes_precedence_over_file(tmp_path):
    """data가 주어지면 파일 존재 여부와 무관하게 data를 해시한다."""
    data = b"from bytes"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_file_hash("non-existent.bin", data=data) == expected
