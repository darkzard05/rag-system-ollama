"""PDF 래핑 라인 리플로우 + 축약어 예외 처리 테스트 (R2-03).

결함: ``_split_sentences``(semantic_chunker.py:104-144)가 라인 단위로 문장을
분할해 PDF 마크다운의 시각적 줄바꿈(래핑) 파편이 독립 "문장"으로 남았다
(실측: 44줄 중 26줄 = 59% 래핑 파편).

수정: 문장 분할 정규식을 재작성하지 않고, 분할 진입 전에 연속 일반 텍스트
라인을 개행→공백(1:1 길이 보존)으로 병합하는 리플로우 전처리를 도입한다.
빈 줄/마크다운 헤더/리스트 마커/테이블 행은 구조 요소이므로 병합에서 제외한다.
또한 축약어(et al., e.g., i.e., Dr., Mr., vs.) 뒤 마침표는 가짜 문장 경계로
취급하지 않아야 한다.
"""

import pytest

from core.semantic_chunker import EmbeddingBasedSemanticChunker


class _StubEmbedder:
    """``_split_sentences``는 임베딩을 사용하지 않으므로 최소 대역으로 충분."""

    model = "stub"

    def embed_documents(self, texts):
        return [[0.0] * 4 for _ in texts]

    def embed_query(self, text):
        return [0.0] * 4


class _MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self._store.get(key)

    async def set(self, key: str, value: object, **kwargs) -> None:
        self._store[key] = value


def _chunker() -> EmbeddingBasedSemanticChunker:
    return EmbeddingBasedSemanticChunker(
        embedder=_StubEmbedder(),
        cache_manager=_MemoryCache(),
        min_chunk_size=0,
        max_chunk_size=500,
    )


def test_wrapped_paragraph_merges_into_single_sentence():
    """PDF 래핑 파편 3줄이 하나의 문장으로 병합되어야 한다 (라인 단위 파편 금지)."""
    chunker = _chunker()
    text = (
        "This is a long sentence that is wrapped across\n"
        "multiple lines in the PDF extraction without any\n"
        "sentence terminator until the very end."
    )

    sentences = chunker._split_sentences(text)

    # 1. 라인 단위 파편이 아니라 하나의 문장이어야 한다
    assert len(sentences) == 1, (
        f"래핑 파편이 독립 문장으로 남음: {[s['text'] for s in sentences]}"
    )
    # 2. 개행은 공백으로 치환되어 병합된다
    assert sentences[0]["text"] == (
        "This is a long sentence that is wrapped across multiple lines in the "
        "PDF extraction without any sentence terminator until the very end."
    )


def test_reflow_preserves_offsets_across_line_boundaries():
    """리플로우(개행→공백, 1:1 길이 보존)로 문장 오프셋이 원본 좌표와 일치해야 한다."""
    chunker = _chunker()
    text = (
        "First line ends here. Second line\ncontinues and wraps over\ntwo extra lines."
    )

    sentences = chunker._split_sentences(text)

    assert len(sentences) == 2, (
        f"[{len(sentences)}]개 문장으로 분할됨: {[s['text'] for s in sentences]}"
    )
    first, second = sentences
    assert first["text"] == "First line ends here."
    assert first["start"] == 0
    assert first["end"] == len("First line ends here.")
    assert second["text"] == "Second line continues and wraps over two extra lines."
    # 두 번째 문장은 원본 텍스트의 "Second line" 위치에서 시작해야 한다
    assert second["start"] == text.find("Second line")
    assert second["end"] == len(text)


def test_markdown_structure_lines_not_reflowed():
    """마크다운 헤더/리스트/테이블 라인은 리플로우로 병합되지 않아야 한다."""
    chunker = _chunker()
    text = (
        "## Introduction\n"
        "The body text continues\n"
        "onto the very next line of the body paragraph here.\n"
        "\n"
        "## Details\n"
        "- item one\n"
        "- item two\n"
        "| col a | col b |\n"
        "| 1     | 2     |\n"
    )

    sentences = chunker._split_sentences(text)
    texts = [s["text"] for s in sentences]

    # 1. 헤더는 독립 유지 (본문과 병합 금지)
    assert "## Introduction" in texts, texts
    assert "## Details" in texts, texts
    # 2. 래핑 본문 라인 2개는 하나의 문장으로 병합
    assert (
        "The body text continues onto the very next line of the body paragraph here."
    ) in texts, texts
    # 3. 리스트 마커 라인은 각각 독립 유지
    assert "- item one" in texts
    assert "- item two" in texts
    # 4. 테이블 행은 각각 독립 유지 (내부 공백 보존)
    assert "| col a | col b |" in texts
    assert "| 1     | 2     |" in texts


def test_abbreviations_not_treated_as_sentence_boundaries():
    """축약어 마침표(et al., e.g., i.e., Dr., Mr., vs.)는 문장 경계가 아니다."""
    chunker = _chunker()
    text = (
        "As shown in Radford et al. (2021) and e.g. Smith et al. (2020), the model "
        "performs well. It was evaluated by Dr. Lee and Mr. Kim vs. the baseline."
    )

    sentences = chunker._split_sentences(text)

    assert len(sentences) == 2, (
        f"축약어 마침표가 가짜 문장 경계로 분할됨: {[s['text'] for s in sentences]}"
    )
    assert sentences[0]["text"] == (
        "As shown in Radford et al. (2021) and e.g. Smith et al. (2020), "
        "the model performs well."
    )
    assert sentences[1]["text"] == (
        "It was evaluated by Dr. Lee and Mr. Kim vs. the baseline."
    )
