import asyncio
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from common.circuit_breaker import CircuitBreakerOpen, get_circuit_breaker_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CBTest")


async def test_circuit_breaker_isolation():
    registry = get_circuit_breaker_registry()

    # 실패 시뮬레이션 함수
    async def failing_service():
        raise ConnectionError("Service connection failed")

    # 성공 시뮬레이션 함수
    async def healthy_service():
        return "Success"

    print("\n=== 서킷 브레이커 세션 격리 테스트 시작 ===")

    user_a = "user_A"
    user_b = "user_B"
    service = "pdf_inference"

    # 1. 사용자 A: 연속 실패 시도 (임계값 5회)
    print(f"[*] 사용자 A ({user_a}) 장애 유발 중...")
    breaker_a = registry.get_breaker(service, session_id=user_a, failure_threshold=3)

    for i in range(3):
        try:
            await breaker_a.call_async(failing_service)
        except ConnectionError:
            pass

    print(f"[!] 사용자 A 서킷 상태: {breaker_a.get_state()}")

    # 2. 사용자 A: 차단 확인
    try:
        await breaker_a.call_async(failing_service)
        print("❌ 실패: 사용자 A의 서킷이 열리지 않았습니다.")
    except CircuitBreakerOpen:
        print("✅ 성공: 사용자 A의 요청이 차단되었습니다.")

    # 3. 사용자 B: 정상 호출 (사용자 A와 독립적인지 확인)
    print(f"\n[*] 사용자 B ({user_b}) 정상 요청 시도...")
    breaker_b = registry.get_breaker(service, session_id=user_b)

    try:
        result = await breaker_b.call_async(healthy_service)
        print(f"✅ 성공: 사용자 B는 영향을 받지 않고 '{result}'를 수신했습니다.")
        print(f"[*] 사용자 B 서킷 상태: {breaker_b.get_state()}")
    except CircuitBreakerOpen:
        print("❌ 실패: 사용자 B까지 억울하게 차단되었습니다 (격리 실패).")

    # 4. 전체 메트릭 확인
    print("\n=== 전체 서킷 메트릭 ===")
    # metrics = registry.get_all_metrics()
    # print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(test_circuit_breaker_isolation())
