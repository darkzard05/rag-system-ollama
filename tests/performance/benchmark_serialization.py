import json
import os
import pickle
import time

from langchain_core.documents import Document


def benchmark_serialization():
    print("🚀 [Benchmark] 캐시 직렬화 성능 테스트 (JSON vs Pickle)")

    # 1. 가상 데이터 생성 (1,000개 청크)
    print("📊 테스트 데이터 생성 중 (1,000 chunks)...")
    sample_docs = [
        Document(
            page_content=f"이것은 {i}번째 테스트 문장입니다. 성능 최적화를 위한 가상 데이터입니다."
            * 5,
            metadata={"source": "test.pdf", "page": i // 10, "chunk_index": i},
        )
        for i in range(1000)
    ]

    # Helper: Serialize docs for JSON
    def serialize_docs(docs):
        return [d.dict() for d in docs]

    # --- [JSON Test] ---
    print("\n--- [JSON] 방식 측정 ---")
    start_time = time.time()

    # Save
    json_data = serialize_docs(sample_docs)
    with open("test_cache.json", "w", encoding="utf-8") as f:
        json.dump(json_data, f)
    save_time_json = time.time() - start_time

    # Load
    start_time = time.time()
    with open("test_cache.json", encoding="utf-8") as f:
        loaded_json = json.load(f)
        _ = [Document(**d) for d in loaded_json]
    load_time_json = time.time() - start_time

    print(f"⏱️ 저장 시간: {save_time_json:.4f}초")
    print(f"⏱️ 로드 시간: {load_time_json:.4f}초")

    # --- [Pickle Test] ---
    print("\n--- [Pickle] 방식 측정 ---")
    start_time = time.time()

    # Save
    with open("test_cache.pkl", "wb") as f:
        pickle.dump(sample_docs, f)
    save_time_pkl = time.time() - start_time

    # Load
    start_time = time.time()
    with open("test_cache.pkl", "rb") as f:
        loaded_pkl = pickle.load(f)
    load_time_pkl = time.time() - start_time

    print(f"⏱️ 저장 시간: {save_time_pkl:.4f}초")
    print(f"⏱️ 로드 시간: {load_time_pkl:.4f}초")

    # 2. 결과 비교
    improvement = ((load_time_json - load_time_pkl) / load_time_json) * 100
    print("\n" + "=" * 40)
    print("📈 성능 개선 결과 (로드 속도)")
    print(f"  - JSON: {load_time_json:.4f}초")
    print(f"  - Pickle: {load_time_pkl:.4f}초")
    print(f"  - 개선율: {improvement:.1f}%")
    print("=" * 40)

    # 3. 결함 체크 (무결성 검증)
    print("\n🔍 무결성 검증 중...")
    is_ok = True
    if len(sample_docs) != len(loaded_pkl):
        print("❌ 결함 발견: 문서 개수가 일치하지 않습니다.")
        is_ok = False

    if sample_docs[0].page_content != loaded_pkl[0].page_content:
        print("❌ 결함 발견: 문서 내용이 변형되었습니다.")
        is_ok = False

    if sample_docs[500].metadata != loaded_pkl[500].metadata:
        print("❌ 결함 발견: 메타데이터가 손실되었습니다.")
        is_ok = False

    if is_ok:
        print("✅ 무결성 검증 완료: 데이터 결함 없음.")

    # 파일 정리
    os.remove("test_cache.json")
    os.remove("test_cache.pkl")


if __name__ == "__main__":
    benchmark_serialization()
