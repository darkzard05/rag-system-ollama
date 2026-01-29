import asyncio
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from core.thread_safe_session import ThreadSafeSessionManager as SessionManager
from src.core.search_aggregator import SearchResultAggregator, AggregationStrategy


@dataclass
class MockResult:
    doc_id: str
    content: str
    score: float
    node_id: str
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


async def test_session_isolation():
    print("--- 세션 격리 테스트 시작 ---")

    # 세션 A 설정
    SessionManager.set_session_id("session-A")
    SessionManager.init_session()
    SessionManager.set("test_key", "value-A")
    print(f"[Session A] 설정 완료: test_key = {SessionManager.get('test_key')}")

    # 세션 B 설정 (비동기 태스크 시뮬레이션)
    async def run_session_b():
        SessionManager.set_session_id("session-B")
        SessionManager.init_session()
        SessionManager.set("test_key", "value-B")
        print(f"[Session B] 설정 완료: test_key = {SessionManager.get('test_key')}")
        return SessionManager.get("test_key")

    value_b = await asyncio.create_task(run_session_b())

    # 세션 A 값 재확인 (격리 확인)
    value_a = SessionManager.get("test_key")
    print(f"[Session A] 최종 확인: test_key = {value_a}")

    assert value_a == "value-A", f"세션 A 값이 오염됨: {value_a}"
    assert value_b == "value-B", f"세션 B 값이 잘못됨: {value_b}"
    print("--- 세션 격리 테스트 성공 ---")


async def test_aggregator_logic():
    print("\n--- 검색 집계 최적화 테스트 시작 ---")

    aggregator = SearchResultAggregator()

    # 중복된 콘텐츠를 가진 결과 생성
    results = {
        "node1": [
            MockResult("id1", "중복 내용", 0.9, "node1"),
            MockResult("id2", "고유 내용 A", 0.8, "node1"),
        ],
        "node2": [
            MockResult("id3", "중복 내용", 0.85, "node2"),  # id1과 내용 동일
            MockResult("id4", "고유 내용 B", 0.7, "node2"),
        ],
    }

    aggregated, metrics = aggregator.aggregate_results(
        results, strategy=AggregationStrategy.DEDUP_CONTENT
    )

    print(f"입력 결과: 4개, 출력 결과: {len(aggregated)}개")
    print(f"발견된 중복: {metrics.duplicates_found}")

    # 중복 내용 1개 + 고유 내용 2개 = 총 3개여야 함
    assert len(aggregated) == 3, f"결과 개수 불일치: {len(aggregated)}"
    assert metrics.duplicates_found == 1, f"중복 감지 실패: {metrics.duplicates_found}"

    # 중복된 항목 중 점수가 더 높은 것이 유지되었는지 확인
    dup_res = next(r for r in aggregated if r.content == "중복 내용")
    assert dup_res.aggregated_score == 0.9, (
        f"중복 병합 점수 오류: {dup_res.aggregated_score}"
    )

    print("--- 검색 집계 최적화 테스트 성공 ---")


if __name__ == "__main__":
    asyncio.run(test_session_isolation())
    asyncio.run(test_aggregator_logic())
