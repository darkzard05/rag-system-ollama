"""
IVF 인덱스 nprobe 설정 및 GPU 게이트 검증 테스트 (R3a-02 / R3a-05).

R3a-02: IVF+SQ8 티어(2만 청크 이상)에서 nprobe 미설정(FAISS 기본 1)으로 리콜 붕괴.
R3a-05: faiss-cpu 환경에서 GPU 자동 감지가 무의미하고, HNSW 티어가 GPU 전환
        대상에 포함되며, GPU 활성 시 efSearch 설정이 스킵되는 문제.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from langchain_core.documents import Document
from src.core.retriever_factory import create_vector_store

IVF_TIER_CHUNKS = 20000  # 4단계(IVF) 티어 경계: chunk_count >= 20000
IVF_TIER_NLIST = int(4 * np.sqrt(IVF_TIER_CHUNKS))


@pytest.fixture
def mock_session_manager(monkeypatch):
    """SessionManager.add_status_log를 모킹해 실제 세션 의존성을 제거합니다."""
    import src.core.retriever_factory as rf

    monkeypatch.setattr(rf, "SessionManager", MagicMock())


def _ivf_tier_data() -> tuple[list[Document], np.ndarray]:
    """IVF 티어(2만 청크)를 트리거하는 소형 더미 문서·벡터를 반환합니다."""
    rng = np.random.default_rng(0)
    vectors = rng.random((IVF_TIER_CHUNKS, 128), dtype=np.float32)
    docs = [
        Document(page_content=f"문서 {i}", metadata={"page": i % 10})
        for i in range(IVF_TIER_CHUNKS)
    ]
    return docs, vectors


@pytest.fixture
def small_ivf_index_factory(monkeypatch):
    """index_factory를 소형 IVF(IVF8,SQ8) 실인덱스로 래핑합니다.

    2만 청크 분기(IVF{nlist},SQ8)를 실제로 통과하되 k-means 훈련 비용만 낮춥니다.
    """

    import faiss

    real_index_factory = faiss.index_factory

    def _small_ivf_factory(d: int, index_type: str, metric: int = 0):
        return real_index_factory(d, "IVF8,SQ8", metric)

    monkeypatch.setattr(faiss, "index_factory", _small_ivf_factory)


@pytest.fixture
def gpu_off(monkeypatch):
    """GPU 자동 감지를 확정적으로 차단합니다 (faiss-cpu 환경 가정)."""
    import faiss
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 0)


def _simulate_faiss_gpu(monkeypatch):
    """faiss-gpu 설치 환경을 시뮬레이션합니다 (심볼 + GPU 1개)."""
    import faiss
    import torch

    import core.retriever_factory as rf

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1)
    # faiss-cpu에는 이 심볼들이 없으므로 raising=False로 신규 주입
    monkeypatch.setattr(faiss, "StandardGpuResources", lambda: object(), raising=False)
    monkeypatch.setattr(
        faiss, "GpuMultipleClonerOptions", lambda: MagicMock(), raising=False
    )

    conv_mock = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(faiss, "index_cpu_to_gpu_multiple", conv_mock, raising=False)

    # C-extension 모듈(faiss-cpu)에서는 setattr 주입이 hasattr 체크에 반영되지
    # 않아 _faiss_gpu_supported() 가 False 를 리턴하는 환경 의존성이 있다.
    # GPU 지원 가정을 명시적으로 강제해 테스트를 환경 독립적으로 만든다.
    monkeypatch.setattr(rf, "_faiss_gpu_supported", lambda faiss_mod: True)

    from core.model_loader import ModelManager

    monkeypatch.setattr(
        ModelManager,
        "get_faiss_gpu_resources",
        classmethod(lambda cls: MagicMock()),
    )
    return conv_mock


def test_ivf_nprobe_set_to_min_default(
    monkeypatch, mock_session_manager, gpu_off, small_ivf_index_factory
):
    """R3a-02: IVF 분기에서 index.nprobe == min(16, nlist) 단언 (FAISS 기본 1 금지)."""
    docs, vectors = _ivf_tier_data()
    result = create_vector_store(docs, MagicMock(), vectors=vectors)

    expected = min(16, IVF_TIER_NLIST)
    assert result.index.nprobe == expected


def test_ivf_nprobe_config_driven(
    monkeypatch, mock_session_manager, gpu_off, small_ivf_index_factory
):
    """R3a-02: config.yml index_params.nprobe 값이 코드에 반영됩니다."""
    import common.config

    monkeypatch.setitem(common.config.VECTOR_STORE_CONFIG["index_params"], "nprobe", 8)

    docs, vectors = _ivf_tier_data()
    result = create_vector_store(docs, MagicMock(), vectors=vectors)

    assert result.index.nprobe == min(8, IVF_TIER_NLIST)


def test_hnsw_efsearch_applied_without_faiss_gpu_symbols(
    monkeypatch, mock_session_manager
):
    """R3a-05: faiss-gpu 심볼 부재 시 GPU 감지가 무효화되고 HNSW efSearch가 유지됩니다.

    torch.cuda.is_available() == True 이고 faiss.get_num_gpus() == 1 이어도,
    faiss-gpu(StandardGpuResources 등)가 없으면 GPU 경로를 타면 안 된다.
    """
    import faiss
    import torch

    if hasattr(faiss, "StandardGpuResources"):
        pytest.skip("faiss-gpu가 설치되어 있어 faiss-cpu 전제를 검증할 수 없습니다.")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1)

    n = 3000  # 2단계 티어 (500 ~ 5000): HNSW32,Flat, ef_search=128
    rng = np.random.default_rng(1)
    vectors = rng.random((n, 128), dtype=np.float32)
    docs = [Document(page_content=f"문서 {i}") for i in range(n)]

    result = create_vector_store(docs, MagicMock(), vectors=vectors)

    hnsw_index = faiss.downcast_index(result.index)
    assert hnsw_index.hnsw.efSearch == 128


def test_hnsw_tier_never_attempts_gpu_conversion(monkeypatch, mock_session_manager):
    """R3a-05: HNSW 티어(5000~20000)는 GPU 전환을 시도하지 않아야 합니다.

    FAISS 가이드: HNSW는 "Supported on GPU: no" — 전환 시도는 예외 → CPU 폴백만 유발.
    GPU가 활성 감지돼도 efSearch(256)는 적용되어야 한다.
    """
    import faiss

    conv_mock = _simulate_faiss_gpu(monkeypatch)

    n = 8000  # 3단계 티어 (5000 ~ 20000): HNSW32,SQ8, ef_search=256
    rng = np.random.default_rng(2)
    vectors = rng.random((n, 128), dtype=np.float32)
    docs = [Document(page_content=f"문서 {i}") for i in range(n)]

    result = create_vector_store(docs, MagicMock(), vectors=vectors)

    conv_mock.assert_not_called()
    hnsw_index = faiss.downcast_index(result.index)
    assert hnsw_index.hnsw.efSearch == 256


def test_ivf_tier_gpu_conversion_preserves_nprobe(
    monkeypatch, mock_session_manager, small_ivf_index_factory
):
    """R3a-05: GPU 전환은 2만 이상 IVF 티어에서만 발생하며 nprobe를 보존합니다."""
    conv_mock = _simulate_faiss_gpu(monkeypatch)

    docs, vectors = _ivf_tier_data()
    create_vector_store(docs, MagicMock(), vectors=vectors)

    conv_mock.assert_called_once()
    # nprobe는 GPU 복제 이전 CPU 인덱스에서 설정돼야 복제 시 보존된다.
    cpu_index = conv_mock.call_args[0][1]
    assert cpu_index.nprobe == min(16, IVF_TIER_NLIST)
