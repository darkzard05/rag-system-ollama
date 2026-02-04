import hashlib
import time

def hash_sha256(content, source, page):
    """현재 방식: SHA256 해싱"""
    key = f"{content}_{source}_{page}"
    return hashlib.sha256(key.encode()).hexdigest()

def hash_tuple(content, source, page):
    """최적화 방식: 튜플 해싱"""
    return hash((content, source, page))

def run_benchmark():
    # 500개의 문서 조각 시뮬레이션
    n_docs = 500
    docs = [("This is a sample sentence content " * 10, "report.pdf", i) for i in range(n_docs)]
    
    print(f"--- 중복 제거 해싱 벤치마크 시작 (문서: {n_docs}개) ---")
    
    # SHA256 방식 측정
    start = time.time()
    for _ in range(100):
        for d in docs:
            _ = hash_sha256(*d)
    dur_sha = (time.time() - start) / 100
    print(f"[SHA256] 평균 소요 시간: {dur_sha*1000:.4f} ms")
    
    # Tuple 방식 측정
    start = time.time()
    for _ in range(100):
        for d in docs:
            _ = hash_tuple(*d)
    dur_tuple = (time.time() - start) / 100
    print(f"[Tuple Hash] 평균 소요 시간: {dur_tuple*1000:.4f} ms")
    
    improvement = (dur_sha - dur_tuple) / dur_sha * 100
    print(f"\n성능 향상: {improvement:.2f}%")
    if improvement > 50:
        print("결론: 튜플 해싱이 압도적으로 빠릅니다.")

if __name__ == "__main__":
    run_benchmark()