import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from infra.error_recovery import AdaptiveTimeout

logging.basicConfig(level=logging.DEBUG)


def test_adaptive_timeout_lower_bound():
    print("\n=== AdaptiveTimeout 하한선 테스트 시작 ===")

    # 최소 타임아웃을 1.0초로 설정
    adaptive = AdaptiveTimeout(initial_timeout=30.0, min_timeout=1.0)

    # 1. 초고속 실패 시나리오 (0.01초 소요되는 작업 10번 기록)
    print("[*] 0.01초 소요되는 실패 작업 10회 기록 중...")
    for _ in range(10):
        adaptive.record_execution_time(0.01)

    # 2. 타임아웃 계산 결과 확인
    calculated_timeout = adaptive.get_adaptive_timeout()
    print(f"[!] 계산된 타임아웃: {calculated_timeout:.4f}초")

    # 3. 검증
    if calculated_timeout >= 1.0:
        print("✅ 성공: 타임아웃이 하한선(1.0s) 이상으로 유지되었습니다.")
    else:
        print(f"❌ 실패: 타임아웃이 {calculated_timeout:.4f}s까지 추락했습니다!")


if __name__ == "__main__":
    test_adaptive_timeout_lower_bound()
