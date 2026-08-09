"""평가 하니스 오라클 순수 함수 단위 테스트 (R4-01 수정 검증).

scripts/eval_quality.py의 _doc_relevant가 한국어 엔티티를 영문 PDF 텍스트와
매칭하는지 검증한다. 순수 함수이므로 모델/네트워크 없이 실행 가능하다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
for _path in (str(ROOT), str(ROOT / "scripts")):
    if _path not in sys.path:
        sys.path.append(_path)

from eval_quality import _doc_relevant  # noqa: E402


class _FakeDoc:
    """page_content만 가진 최소 Document 대역 (model-free)."""

    def __init__(self, content: str) -> None:
        self.page_content = content


def test_korean_entity_tokenization_matches_english_doc() -> None:
    """'토큰화' 엔티티로 영문 토큰화 문서를 관련(relevant) 판정해야 한다.

    R4-01: 기존 구현은 한국어 엔티티를 영문 텍스트에 부분문자열 매칭하여
    항상 False를 반환했다. 영문 동의어/어간 맵(tokeniz/token)으로 매칭되어야 한다.
    """
    doc = _FakeDoc(
        "We train CM3 models and tokenize each image with VQVAE-GAN "
        "into 256 image tokens."
    )
    assert _doc_relevant(doc, ("토큰화",)) is True


def test_unrelated_korean_entity_rejects_irrelevant_doc() -> None:
    """'그래프' 엔티티로 무관한 영문 문서를 무관(irrelevant) 판정해야 한다."""
    doc = _FakeDoc(
        "The causal masking objective combines causal and masked language "
        "modeling into a hybrid training objective."
    )
    assert _doc_relevant(doc, ("그래프",)) is False


def test_synonym_map_covers_retrieval_embedding_chunk_sort() -> None:
    """동의어 맵 키(검색/임베딩/청크/정렬)가 영문 어간과 매칭되어야 한다."""
    retriev_doc = _FakeDoc("Hybrid retrieval with dense search over embeddings.")
    assert _doc_relevant(retriev_doc, ("검색", "임베딩")) is True
    chunk_doc = _FakeDoc("Each chunk is indexed separately before ranking.")
    assert _doc_relevant(chunk_doc, ("청크", "정렬")) is True


def test_normalization_ignores_hyphen_and_case() -> None:
    """DALL-E 하이픈과 대소문자를 정규화한 근접 판정이 동작해야 한다."""
    doc = _FakeDoc("dall-e and CM3 models are both autoregressive image models.")
    assert _doc_relevant(doc, ("DALL-E", "cm3")) is True
