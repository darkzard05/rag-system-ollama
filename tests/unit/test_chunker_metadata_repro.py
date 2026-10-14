"""메타데이터 역매핑 재현 테스트 — O(C×D) 전수 스캔 + 매 빌드 스레드풀 생성 (red → green).

결함 위치: ``src/core/semantic_chunker.py``

- ``_extract_metadata_for_chunk``: 오프셋 오름차순으로 이미 정렬된 ``doc_ranges``를
  매 청크마다 전수 스캔한다 (이분 탐색/조기 종료 없음 → O(C×D) 비교).
- ``split_documents``: 호출마다 ``ThreadPoolExecutor()``를 새로 생성·파괴하고,
  ``asyncio.to_thread`` 워커 안에서 또 스레드 풀을 중첩 생성한다.

기대 계약 (수정 후):
- 범위 조회 읽기 횟수가 후보 범위 수에 비례하지 않는다 (bisect + 조기 종료).
- ``split_documents`` 1회 호출당 스레드 풀 신규 생성 0회.
- 반환 메타데이터 스키마(pages/page/is_cross_page/current_section/
  start_index/end_index)는 불변.

이 테스트는 수정 전에는 실패(red)하고 수정 후 통과(green)해야 한다.
"""

import asyncio
import concurrent.futures

from langchain_core.documents import Document

from core.semantic_chunker import EmbeddingBasedSemanticChunker

_N_RANGES = 400
_SCAN_READ_BUDGET = 60


class _MockEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1]


class _CountingRange(dict):
    """``doc_range[key]`` 읽기 횟수를 카운트하는 스파이 dict."""

    def __init__(self, *args, counter: dict, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._counter = counter

    def __getitem__(self, key):  # type: ignore[override]
        self._counter["reads"] += 1
        return super().__getitem__(key)


def _make_sorted_ranges(n: int) -> tuple[list[_CountingRange], dict[str, int]]:
    """오프셋 오름차순·인접(비중첩) doc_ranges — split_documents 생성 규칙과 동일."""
    counter: dict[str, int] = {"reads": 0}
    ranges = [
        _CountingRange(
            start=i * 10,
            end=i * 10 + 9,
            metadata={"page": i + 1},
            counter=counter,
        )
        for i in range(n)
    ]
    return ranges, counter


def test_repro_metadata_range_scan_is_sublinear() -> None:
    """첫 청크에 대해 범위 조회가 전수 스캔(≥2×D 읽기)이면 결함."""
    chunker = EmbeddingBasedSemanticChunker(embedder=_MockEmbedder())
    ranges, counter = _make_sorted_ranges(_N_RANGES)

    chunk = {"start": 0, "end": 5, "text": "x", "current_section": "S"}
    merged = chunker._extract_metadata_for_chunk(chunk, ranges)

    assert merged["page"] == 1, "정확성: 첫 범위가 겹쳐야 한다"
    assert counter["reads"] <= _SCAN_READ_BUDGET, (
        f"범위 조회가 {counter['reads']}회 읽음 (범위 {_N_RANGES}개). "
        f"결함: 정렬된 doc_ranges 전수 스캔 O(D). "
        f"기대: bisect + 조기 종료로 ≤{_SCAN_READ_BUDGET}회"
    )


def test_repro_metadata_result_schema_unchanged() -> None:
    """수정에도 반환 스키마가 유지되는지 보는 안전망 (수정 전후 항상 통과)."""
    chunker = EmbeddingBasedSemanticChunker(embedder=_MockEmbedder())
    ranges: list[dict] = [
        {"start": 0, "end": 9, "metadata": {"page": 1}},
        {"start": 10, "end": 19, "metadata": {"page": 2}},
    ]

    # 청크가 두 페이지에 걸침
    merged = chunker._extract_metadata_for_chunk(
        {"start": 5, "end": 15, "current_section": "SEC"}, ranges
    )
    assert merged["pages"] == [1, 2]
    assert merged["page"] == 1
    assert merged["is_cross_page"] is True
    assert merged["current_section"] == "SEC"
    assert merged["start_index"] == 5
    assert merged["end_index"] == 15

    # 걸치지 않는 청크: 단일 페이지, 교차 플래그 없음
    single = chunker._extract_metadata_for_chunk(
        {"start": 0, "end": 9, "current_section": "SEC"}, ranges
    )
    assert single["pages"] == [1]
    assert single["page"] == 1
    assert single["is_cross_page"] is False


async def test_repro_split_documents_creates_threadpool_per_call(
    monkeypatch,
) -> None:
    """split_documents 1회 호출에서 ThreadPoolExecutor 신규 생성이 발생하면 결함."""
    # asyncio.to_thread의 기본 executor를 패치 "이전"에 사전 생성한다.
    # (새 루프의 기본 풀 생성이 카운터에 잡히는 오탐 방지)
    await asyncio.to_thread(lambda: None)

    created: dict[str, int] = {"pools": 0}
    real_pool = concurrent.futures.ThreadPoolExecutor

    def counting_pool(*args, **kwargs):
        created["pools"] += 1
        return real_pool(*args, **kwargs)

    monkeypatch.setattr(concurrent.futures, "ThreadPoolExecutor", counting_pool)

    chunker = EmbeddingBasedSemanticChunker(embedder=_MockEmbedder())

    async def fake_split_text(text: str) -> list[dict]:
        return [
            {"text": "a", "vector": [0.1], "start": 0, "end": 11},
            {"text": "b", "vector": [0.2], "start": 12, "end": 23},
        ]

    monkeypatch.setattr(chunker, "split_text", fake_split_text)

    docs, vectors = await chunker.split_documents(
        [Document(page_content="hello world")]
    )

    assert len(docs) == 2
    assert len(vectors) == 2
    assert created["pools"] == 0, (
        f"split_documents가 스레드 풀을 {created['pools']}회 신규 생성함. "
        "결함: 호출마다 ThreadPoolExecutor 생성/파괴 + to_thread 중첩. "
        "기대: 0회 (이미 워커 스레드이므로 순수 루프로 처리)"
    )
