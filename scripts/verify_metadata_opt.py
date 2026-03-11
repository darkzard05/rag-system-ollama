
import sys
import os
import json
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from langchain_core.documents import Document
from cache.coord_cache import coord_cache

def get_obj_size(obj):
    """객체의 대략적인 JSON 직렬화 크기 측정"""
    return len(json.dumps(obj, default=str))

def test_metadata_optimization():
    print("\n=== [1] 메모리 절감 효과 측정 ===")
    
    # 가상의 거대 좌표 데이터 (1000단어)
    large_coords = [(i, i+1, i+2, i+3, f"word_{i}") for i in range(1000)]
    file_hash = "test_file_001"
    page_num = 1
    
    # 최적화 전: 모든 좌표를 포함
    doc_before = Document(
        page_content="테스트 본문 내용입니다.",
        metadata={"word_coords": large_coords, "page": page_num}
    )
    size_before = get_obj_size(doc_before.metadata)
    
    # 최적화 후: 좌표는 캐시에 저장하고 참조만 유지
    coord_cache.save_coords(file_hash, page_num, large_coords)
    doc_after = Document(
        page_content="테스트 본문 내용입니다.",
        metadata={"file_hash": file_hash, "page": page_num, "has_coordinates": True}
    )
    size_after = get_obj_size(doc_after.metadata)
    
    reduction = (1 - (size_after / size_before)) * 100
    print(f"최적화 전 메타데이터 크기: {size_before:,} bytes")
    print(f"최적화 후 메타데이터 크기: {size_after:,} bytes")
    print(f"메모리 절감률: {reduction:.2f}%")
    
    print("\n=== [2] 하이드레이션(복구) 무결성 테스트 ===")
    # 복구 전 확인
    print(f"복구 전 'word_coords' 존재 여부: {'word_coords' in doc_after.metadata}")
    
    # 복구 로직 실행
    f_hash = doc_after.metadata.get("file_hash")
    p_num = doc_after.metadata.get("page")
    if doc_after.metadata.get("has_coordinates"):
        recovered = coord_cache.get_coords(f_hash, p_num)
        if recovered:
            doc_after.metadata["word_coords"] = recovered
            print("성공: 좌표 데이터가 캐시로부터 복구되었습니다.")
    
    # 복구 후 확인
    print(f"복구 후 'word_coords' 존재 여부: {'word_coords' in doc_after.metadata}")
    print(f"복구된 데이터 일치 여부: {doc_after.metadata['word_coords'] == large_coords}")

    # 캐시 정리
    coord_cache.clear_cache(file_hash)

if __name__ == "__main__":
    test_metadata_optimization()
