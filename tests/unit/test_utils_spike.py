"""Spike test for utils.py extraction (Batch 5.4).

Baseline assertions against the unmodified 717-line ``common/utils.py`` before
splitting into domain modules. Covers the four domains: text (latex/citation
normalization), annotation tooltips, hashing, and document identity.
"""

from __future__ import annotations

from common.utils import (
    doc_stable_id,
    fast_hash,
    normalize_latex_delimiters,
    strip_context_tokens,
)


def test_normalize_latex_delimiters_converts_and_protects_code_blocks() -> None:
    text = (
        r"인라인 \(x^2\) 와 블록 \[ \int_0^1 dx \] 그리고 다음 코드는 보호: "
        r"```python\nprint(r\"\(not latex\)\")\n```"
    )
    out = normalize_latex_delimiters(text)
    assert r"$x^2$" in out
    assert "$$ \\int_0^1 dx $$" in out  # 주변 공백은 보존
    assert r"\(not latex\)" in out  # 코드 블록 내부는 변환 제외
    assert "```python" in out


def test_strip_context_tokens_removes_meta_tokens_keeps_citation_anchors() -> None:
    out = strip_context_tokens("본문 [doc:abc123] [page:3] [score:0.9] 내용 [1]")
    assert "[doc:" not in out
    assert "[page:" not in out
    assert "[score:" not in out
    assert "[1]" in out
    assert " ".join(out.split()) == "본문 내용 [1]"


def test_fast_hash_empty_returns_zero_filled() -> None:
    assert fast_hash("") == "0" * 16
    assert fast_hash("", length=8) == "0" * 8


def test_fast_hash_is_deterministic_and_discriminating() -> None:
    h1 = fast_hash("동일한 입력")
    h2 = fast_hash("동일한 입력")
    h3 = fast_hash("다른 입력")
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
    assert isinstance(h1, str)


def test_doc_stable_id_prefers_doc_id_metadata() -> None:
    doc = {"page_content": "내용", "metadata": {"doc_id": "stable-42"}}
    assert doc_stable_id(doc) == "stable-42"


def test_doc_stable_id_falls_back_to_content_hash() -> None:
    doc = {"page_content": "해시 대상 내용", "metadata": {}}
    out = doc_stable_id(doc)
    assert isinstance(out, str) and len(out) == 16
