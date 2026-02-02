import os
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

# --- 경로 설정 최적화 (절대 경로 기반) ---
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
from langchain_core.documents import Document

from core.rag_core import VectorStoreCache


def test_vector_store_save_verification():
    """캐시 저장 후 검증 로직이 정상 작동하는지 테스트합니다."""
    file_path = "test_dummy.pdf"
    # 실제 파일이 있어야 해시 계산이 가능함
    with open(file_path, "w") as f:
        f.write("dummy content")

    model_name = "test_model"

    # Mock 객체 생성
    mock_vector_store = MagicMock()
    # pickle 가능한 객체로 대체
    mock_bm25 = {"retriever": "mock"}
    doc_splits = [Document(page_content="test", metadata={"source": "test"})]

    cache = VectorStoreCache(file_path, model_name)

    # 1. 정상 저장 시나리오
    def side_effect_save(path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "index.faiss"), "w") as f:
            f.write("data")
        with open(os.path.join(path, "index.pkl"), "w") as f:
            f.write("data")

    mock_vector_store.save_local.side_effect = side_effect_save

    try:
        cache.save(doc_splits, mock_vector_store, mock_bm25)
        assert os.path.exists(cache.faiss_index_path)
        assert os.path.exists(cache.doc_splits_path)
        assert os.path.exists(cache.bm25_retriever_path)
    finally:
        if os.path.exists(cache.cache_dir):
            shutil.rmtree(cache.cache_dir)
        if os.path.exists(file_path):
            os.remove(file_path)


def test_vector_store_save_failure_cleanup():
    """저장 실패 시 캐시가 정리되는지 테스트합니다."""
    file_path = "test_fail.pdf"
    model_name = "test_model"

    mock_vector_store = MagicMock()
    # FAISS 저장이 실패한다고 가정
    mock_vector_store.save_local.side_effect = Exception("Disk Full")

    cache = VectorStoreCache(file_path, model_name)

    with pytest.raises(Exception, match="Disk Full"):
        cache.save([], mock_vector_store, MagicMock())

    # 실패 후 디렉토리가 삭제되었어야 함
    assert not os.path.exists(cache.cache_dir)


if __name__ == "__main__":
    # 수동 실행용
    try:
        test_vector_store_save_verification()
        print("✅ Save verification test passed")
        test_vector_store_save_failure_cleanup()
        print("✅ Save failure cleanup test passed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
