"""BM25 stemming must strip the particle that actually matches the suffix."""

from __future__ import annotations

from common.text_utils import bm25_tokenizer


def test_single_char_particle_does_not_trim_two_chars() -> None:
    tokens = bm25_tokenizer("인공지능의")

    assert "인공지능" in tokens
    assert "인공지" not in tokens


def test_single_char_subject_particle_keeps_full_stem() -> None:
    tokens = bm25_tokenizer("문서검색시스템이")

    assert "문서검색시스템" in tokens
    assert "문서검색시스" not in tokens


def test_two_char_particle_is_still_removed() -> None:
    tokens = bm25_tokenizer("검색에서")

    assert "검색" in tokens
    assert "검" not in tokens
