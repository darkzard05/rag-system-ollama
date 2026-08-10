"""정규화 배선 테스트 (R2-02): 병합·중복제거·청크 벡터가 단위노름 벡터를 사용한다.

결함: ``_get_embeddings(..., normalize=True)`` 파라미터가 존재하지만 split_text의
병합(:511)·중복제거(:579)·청크 벡터(:754) 경로에 배선되지 않아, "코사인 유사도"로
주석된 판정이 비정규화 raw dot product로 계산됐다 (실측 norm ≠ 1).

수정: 문장 임베딩(normalize=True)과 청크 벡터를 단위노름으로 정규화해 모든
dot product를 진짜 코사인 유사도로 만든다.
"""

import numpy as np
import pytest

from core.semantic_chunker import EmbeddingBasedSemanticChunker


class _ShortNormEmbedder:
    """단위노름이 아닌(0.5) 결정적 one-hot 벡터를 반환하는 임베더.

    "lazy fox"/"chemistry" 키워드 → 서로 직교하는 벡터. 비정규화 상태에서는
    내적이 0.25(중복 청크)로 ``_prune_duplicates``의 0.98 임계값에 못 미친다.
    """

    def __init__(self, dimension: int = 16) -> None:
        self.dimension = dimension
        self.model_name = "short-norm"

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text):
        v = np.zeros(self.dimension)
        if "lazy fox" in text:
            v[0] = 0.5
        elif "chemistry" in text:
            v[1] = 0.5
        else:
            v[2] = 0.5
        return v.tolist()


class _MemoryCache:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    async def get(self, key: str) -> object | None:
        return self._store.get(key)

    async def set(self, key: str, value: object, **kwargs) -> None:
        self._store[key] = value


def _make_chunker(**kwargs) -> EmbeddingBasedSemanticChunker:
    defaults: dict = dict(
        embedder=_ShortNormEmbedder(),
        cache_manager=_MemoryCache(),
        min_chunk_size=0,
        max_chunk_size=500,
        batch_size=8,
        breakpoint_threshold_type="similarity_threshold",
        similarity_threshold=0.5,  # 직교 문장 간 distance≈1 > 0.5 → 전부 분기
        buffer_size=0,  # 버퍼 평균을 쓰지 않아 결정적 distance 계산
    )
    defaults.update(kwargs)
    return EmbeddingBasedSemanticChunker(**defaults)


_LAZY = "The lazy fox slept under the old oak tree beside the quiet river all night. "
_CHEM = "A chemistry reaction between acids and bases produced a stable salt here. "


@pytest.mark.asyncio
async def test_chunk_vectors_are_unit_norm():
    """split_text가 반환하는 모든 청크 벡터는 단위노름이어야 한다.

    임베더가 norm=0.5 벡터를 주므로, normalizer 배선이 없다면 청크 벡터의
    norm은 0.5가 되어 이 단언이 실패한다.
    """
    chunker = _make_chunker()
    text = (
        _LAZY
        + _CHEM
        + "Different unrelated facts and figures and names are collected here. "
        + "The lazy fox jumped over the fence chasing a mouse into the tall grass. "
        + "More chemistry compounds mixed with water yielded an interesting result. "
        + "Other unrelated trivia and statistics fill this final filler sentence. "
    )

    chunks = await chunker.split_text(text)

    assert len(chunks) >= 2
    for chunk in chunks:
        vec = np.asarray(chunk["vector"], dtype="float32")
        norm = float(np.linalg.norm(vec))
        assert norm == pytest.approx(1.0, abs=1e-5), (
            f"청크 벡터가 단위노름이 아님 (norm={norm}) — 정규화 배선 누락"
        )


@pytest.mark.asyncio
async def test_split_text_passes_normalize_flag(monkeypatch):
    """메인 경로(:664)의 모든 ``_get_embeddings`` 호출에 normalize=True가 전달된다."""
    chunker = _make_chunker()
    calls: list[bool] = []
    original = chunker._get_embeddings

    async def spy(texts, normalize=False):
        calls.append(normalize)
        return await original(texts, normalize)

    monkeypatch.setattr(chunker, "_get_embeddings", spy)

    await chunker.split_text(_LAZY + _CHEM + _LAZY)

    assert calls, "_get_embeddings가 호출되지 않음"
    assert all(calls), f"normalize=True가 모든 호출에 전달되어야 함: {calls}"


@pytest.mark.asyncio
async def test_short_text_path_also_normalizes(monkeypatch):
    """buffer_size 이하(얼리 리턴 경로 :653)에서도 normalize=True가 전달된다."""
    chunker = _make_chunker(buffer_size=10)
    calls: list[bool] = []
    original = chunker._get_embeddings

    async def spy(texts, normalize=False):
        calls.append(normalize)
        return await original(texts, normalize)

    monkeypatch.setattr(chunker, "_get_embeddings", spy)

    result = await chunker.split_text(_LAZY + _CHEM)

    assert result, "split_text가 문장 리스트를 반환해야 함"
    assert calls and all(calls), f"얼리 리턴 경로에서 normalize 미전달: {calls}"


@pytest.mark.asyncio
async def test_duplicate_chunks_pruned_by_normalized_cosine():
    """동일 본문 2회 반복이 정규화 코사인(1.0 > 0.98)으로 중복 제거된다.

    비정규화였으면 내적 = 0.5 * 0.5 = 0.25 < 0.98라 중복이 남아 3개 청크가
    된다. (버퍼 없이 직교 문장이라 각 문장이 독립 청크가 됨)
    """
    chunker = _make_chunker()

    chunks = await chunker.split_text(_LAZY + _CHEM + _LAZY)

    assert len(chunks) == 2, (
        f"동일 문장 2회가 정규화 코사인으로 중복 제거되어야 함: "
        f"{[c['text'][:50] for c in chunks]}"
    )
