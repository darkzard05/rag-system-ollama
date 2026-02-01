import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def test_quantization_impact():
    # 1. 가상의 원본 벡터 생성 (384차원, MiniLM 기준)
    np.random.seed(42)
    original_vector = np.random.randn(1, 384).astype(np.float32)
    original_vector /= np.linalg.norm(original_vector)
    
    # 유사한 문서 벡터 생성
    similar_vector = original_vector + np.random.randn(1, 384) * 0.1
    similar_vector /= np.linalg.norm(similar_vector)
    
    true_sim = cosine_similarity(original_vector, similar_vector)[0][0]
    print(f"--- 원본 유사도: {true_sim:.4f} ---")

    # 2. 양자화 시뮬레이션 (Scale & Offset 방식)
    v_min, v_max = similar_vector.min(), similar_vector.max()
    scale = (v_max - v_min) / 255.0
    offset = v_min
    # uint8 범위로 클램핑 및 변환 (오버플로우 방지)
    quantized = np.round((similar_vector - offset) / scale).astype(np.uint8)

    # 3. [버그 방식] 단순 타입 변환 (수정 전 코드의 문제점)
    # uint8 정수값(예: 127)을 그대로 실수 127.0으로 취급
    buggy_vector = quantized.astype(np.float32)
    buggy_vector /= (np.linalg.norm(buggy_vector) + 1e-10)
    buggy_sim = cosine_similarity(original_vector, buggy_vector)[0][0]
    
    # 4. [정상 방식] 복원(Dequantize) 후 사용
    correct_vector = (quantized.astype(np.float32) * scale) + offset
    correct_vector /= (np.linalg.norm(correct_vector) + 1e-10)
    correct_sim = cosine_similarity(original_vector, correct_vector)[0][0]

    print(f"[버그 방식] 유사도: {buggy_sim:.4f} (오차: {abs(true_sim - buggy_sim):.4f})")
    print(f"[정상 방식] 유사도: {correct_sim:.4f} (오차: {abs(true_sim - correct_sim):.4f})")
    
    diff_buggy = abs(true_sim - buggy_sim)
    diff_correct = abs(true_sim - correct_sim)
    
    if diff_buggy > diff_correct * 10:
        print(f"\n결과: 버그 방식의 오차가 정상 방식보다 {diff_buggy/diff_correct:.1f}배 큽니다. 수정이 시급합니다.")

if __name__ == "__main__":
    test_quantization_impact()
