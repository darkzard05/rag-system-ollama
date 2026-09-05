"""``MemoizingEmbedding`` 쿼리 임베딩 메모이 래퍼 단위 테스트 (계획 §6-2 12종).

single-flight(동일 텍스트 동시 중복 계산 차단), LRU+TTL 캐시, 실패 폭포 전파,
타임아웃 단일 승계, weakref 레지스트리 격리를 검증한다. 동시성 테스트는
``threading.Event`` 게이트 배리어 패턴을 사용해 결정적으로 수행한다.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

import pytest
from langchain_core.embeddings import Embeddings

import core.embedding_memo as memo_mod
from core.embedding_memo import MemoizingEmbedding, clear_memo_instances

_ZERO_STATS = {"hits": 0, "misses": 0, "evictions": 0, "calls": 0, "in_flight": 0}


class _CountingEmbeddings(Embeddings):
    """inner 임베더 페이크 — 호출 횟수 카운터 + 선택적 호출별 게이트.

    ``embed_query``는 1-based 호출 순번을 벡터로 반환한다(계산자/후속 호출 구분용).
    ``gated=True``면 각 호출이 자기 게이트(``gates[idx - 1]``)에서 블록된다.
    ``fail=True``면 게이트 해제 후 ValueError를 raise 한다.
    """

    def __init__(
        self,
        *,
        gated: bool = False,
        fail: bool = False,
        num_gates: int = 0,
        first_call_started: threading.Event | None = None,
    ) -> None:
        self.gated = gated
        self.fail = fail
        self.calls = 0
        self.gates = [threading.Event() for _ in range(num_gates)]
        self.first_call_started = first_call_started
        self._lock = threading.Lock()

    def _gate_for(self, idx: int) -> threading.Event:
        with self._lock:
            while len(self.gates) < idx:
                self.gates.append(threading.Event())
            return self.gates[idx - 1]

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            self.calls += 1
            idx = self.calls
        if self.first_call_started is not None and idx == 1:
            self.first_call_started.set()
        if self.gated:
            self._gate_for(idx).wait()
        if self.fail:
            raise ValueError(f"embed failure {idx}")
        return [float(idx)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0]] * len(texts)


def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    """게이트 해제만으로 정착을 보장할 수 없는 조건을 폴링 대기한다."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("조건 충족 대기 시간 초과")
        time.sleep(0.02)


class TestMemoQueryDedup:
    def test_same_text_twice_inner_called_once(self) -> None:
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner)

        r1 = memo.embed_query("동일 문구")
        r2 = memo.embed_query("동일 문구")

        assert inner.calls == 1
        assert r1 == r2 == [1.0]
        assert memo.stats()["hits"] == 1
        assert memo.stats()["misses"] == 1

    def test_single_flight_two_threads_one_inner_call(self) -> None:
        first_started = threading.Event()
        inner = _CountingEmbeddings(
            gated=True, num_gates=1, first_call_started=first_started
        )
        memo = MemoizingEmbedding(inner)
        results: list[list[float]] = []
        errors: list[BaseException] = []

        def run() -> None:
            try:
                results.append(memo.embed_query("동시 문구"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t_a = threading.Thread(target=run, daemon=True)
        t_a.start()
        # A가 계산자로 inner compute 진입(게이트 블록)함을 확인
        assert first_started.wait(5.0) is True
        assert inner.calls == 1

        t_b = threading.Thread(target=run, daemon=True)
        t_b.start()
        # B가 event.wait에 정착할 시간 부여 후 — 여전히 계산 시작 안 함(단일-flight)
        time.sleep(0.3)
        assert inner.calls == 1
        assert t_b.is_alive()

        inner.gates[0].set()  # A(계산자) 완료 허용 → B는 event로 wake
        t_a.join(timeout=5.0)
        t_b.join(timeout=5.0)
        assert not t_a.is_alive() and not t_b.is_alive()
        assert errors == []
        assert inner.calls == 1
        assert results == [[1.0], [1.0]]

    def test_failure_propagates_to_waiter_without_recompute(self) -> None:
        first_started = threading.Event()
        inner = _CountingEmbeddings(
            gated=True, fail=True, num_gates=1, first_call_started=first_started
        )
        memo = MemoizingEmbedding(inner)
        errors: list[BaseException] = []
        results: list[list[float]] = []

        def run() -> None:
            try:
                results.append(memo.embed_query("실패 문구"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t_computer = threading.Thread(target=run, daemon=True)
        t_computer.start()
        assert first_started.wait(5.0) is True
        t_waiter = threading.Thread(target=run, daemon=True)
        t_waiter.start()
        time.sleep(0.3)  # waiter가 event.wait에 정착

        inner.gates[0].set()  # 계산자 실패 유발 → waiter는 재계산 없이 동일 예외
        t_computer.join(timeout=5.0)
        t_waiter.join(timeout=5.0)
        assert not t_computer.is_alive() and not t_waiter.is_alive()
        assert results == []
        assert len(errors) == 2
        assert all(isinstance(e, ValueError) for e in errors)
        assert errors[0].args == errors[1].args
        assert inner.calls == 1  # waiter 재계산 없음
        assert memo._in_flight == {}  # 실패 엔트리 해제
        assert memo._cache == {}  # 실패는 캐시 미저장
        assert memo.stats()["calls"] == 1

    def test_return_value_is_copy(self) -> None:
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner)

        r1 = memo.embed_query("복사 문구")
        r1.append(99.0)  # 호출자 변이

        r2 = memo.embed_query("복사 문구")
        assert r2 == [1.0]  # 저장값은 변이의 영향을 받지 않음(사본 반환)
        assert r1 == [1.0, 99.0]
        assert inner.calls == 1


class TestMemoCacheSemantics:
    def test_ttl_expiry_recomputes(self) -> None:
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner, ttl=0.05)

        assert memo.embed_query("q") == [1.0]
        assert memo.embed_query("q") == [1.0]  # TTL 내 → hit
        assert inner.calls == 1

        time.sleep(0.15)  # ttl(0.05)을 넉넉히 초과
        assert memo.embed_query("q") == [2.0]  # 만료 → 재계산
        assert inner.calls == 2

    def test_lru_eviction_and_expired_slots_freed(self) -> None:
        # 만료 항목이 캐시 슬롯을 점유하지 않음
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner, maxsize=2, ttl=0.05)
        memo.embed_query("a")  # call 1
        assert list(memo._cache) == ["a"]
        time.sleep(0.15)  # "a" 만료

        memo.embed_query("b")  # call 2 → put 시 만료 스캔이 "a" 제거
        assert list(memo._cache) == ["b"]
        memo.embed_query("c")  # call 3 → [b, c] == maxsize → 퇴출 0
        assert memo.stats()["evictions"] == 0
        assert inner.calls == 3

        # 진짜 LRU: 최근 접근 항목 생존, 가장 오래된 항목 퇴출
        inner2 = _CountingEmbeddings()
        memo2 = MemoizingEmbedding(inner2, maxsize=2, ttl=300.0)
        memo2.embed_query("a")  # call 1
        memo2.embed_query("b")  # call 2
        memo2.embed_query("a")  # hit → "a" MRU
        memo2.embed_query("c")  # call 3 → put 시 가장 오래된 "b" 퇴출
        assert memo2.stats()["evictions"] == 1
        assert list(memo2._cache) == ["a", "c"]
        assert memo2._cache.get("b") is None

        memo2.embed_query("b")  # call 4 — 퇴출된 키 재조회 → miss 재계산
        assert inner2.calls == 4

    def test_ttl_boundary_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = 1000.0

        def fake_monotonic() -> float:
            return now

        monkeypatch.setattr("time.monotonic", fake_monotonic)
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner, ttl=10.0)

        assert memo.embed_query("x") == [1.0]  # t=1000.0 저장 (call 1)
        now = 1009.9999
        assert memo.embed_query("x") == [1.0]  # TTL 내 → hit
        now = 1010.0
        assert memo.embed_query("x") == [1.0]  # 경계(10.0 <= 10.0) → hit, 재계산 없음
        assert inner.calls == 1
        now = 1010.0001
        assert memo.embed_query("x") == [2.0]  # 경계 초과 → 만료 재계산
        assert inner.calls == 2


class TestMemoStatsClearRegistry:
    def test_clear_resets_state_and_recomputes(self) -> None:
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner)
        memo.embed_query("x")  # call 1
        assert inner.calls == 1

        memo.clear()

        assert memo.stats() == _ZERO_STATS
        assert memo._cache == {}
        assert memo._in_flight == {}
        memo.embed_query("x")  # call 2 — clear 후 재계산
        assert inner.calls == 2
        assert memo.stats()["calls"] == 1
        assert memo.stats()["misses"] == 1

    def test_stats_counters_by_scenario(self) -> None:
        inner = _CountingEmbeddings()
        memo = MemoizingEmbedding(inner, maxsize=2, ttl=300.0)
        memo.embed_query("a")  # miss 1, call 1
        memo.embed_query("a")  # hit 1
        memo.embed_query("b")  # miss 2, call 2
        s = memo.stats()
        assert s["hits"] == 1
        assert s["misses"] == 2
        assert s["calls"] == 2
        assert s["evictions"] == 0
        assert s["in_flight"] == 0

        # 계산 중 in_flight == 1 (gated 배리어)
        started = threading.Event()
        inner_g = _CountingEmbeddings(
            gated=True, num_gates=1, first_call_started=started
        )
        memo_g = MemoizingEmbedding(inner_g)
        t = threading.Thread(target=memo_g.embed_query, args=("x",), daemon=True)
        t.start()
        assert started.wait(5.0) is True
        assert memo_g.stats()["in_flight"] == 1
        assert len(memo_g._in_flight) == 1
        inner_g.gates[0].set()
        t.join(timeout=5.0)
        assert not t.is_alive()
        assert memo_g.stats()["in_flight"] == 0
        assert memo_g._in_flight == {}

    def test_registry_and_global_clear(self) -> None:
        inner1 = _CountingEmbeddings()
        inner2 = _CountingEmbeddings()
        m1 = MemoizingEmbedding(inner1)
        m2 = MemoizingEmbedding(inner2)
        m1.embed_query("a")
        m2.embed_query("b")

        assert m1 in memo_mod._memo_instances  # 등록 확인 (생성 시)
        assert m2 in memo_mod._memo_instances

        clear_memo_instances()  # 레지스트리 순회 전체 clear

        assert m1.stats() == _ZERO_STATS
        assert m2.stats() == _ZERO_STATS
        assert m1._cache == {} and m1._in_flight == {}
        assert m2._cache == {} and m2._in_flight == {}
        m1.embed_query("a")  # 초기화 후 재계산
        assert inner1.calls == 2


class TestMemoTakeover:
    def test_takeover_late_completion_isolated_by_identity_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """승계 후 늦은 완료 격리(항등 가드 pop).

        A(게이트 억제 계산자) → B가 A timeout 승계(새 엔트리+계산) → A 늦게 완료해도
        B의 ``_in_flight`` 엔트리가 살아있음 → 신규 도착 E는 세 번째 compute 시작
        않음(inner 호출 카운트 2 유지).
        """
        monkeypatch.setattr(memo_mod, "_WAIT_BOUND", 0.2)
        first_started = threading.Event()
        inner = _CountingEmbeddings(
            gated=True, num_gates=2, first_call_started=first_started
        )
        memo = MemoizingEmbedding(inner)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                memo.embed_query("x")
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        t_a = threading.Thread(target=run, daemon=True)
        t_a.start()
        assert first_started.wait(5.0) is True  # A: call 1, gate[0] 블록
        inflight_a = memo._in_flight["x"]

        t_b = threading.Thread(target=run, daemon=True)
        t_b.start()
        _wait_until(lambda: inner.calls == 2)  # B: 0.2s timeout → 승계 → call 2 블록
        successor = memo._in_flight["x"]
        assert successor is not inflight_a  # B가 새 엔트리 등록(승계 완료)

        inner.gates[0].set()  # A 늦게 완료
        t_a.join(timeout=5.0)
        assert not t_a.is_alive()
        # 항등 가드: A의 후행 pop이 후임자 B의 엔트리를 제거하지 못함
        assert memo._in_flight.get("x") is successor

        # 신규 도착 E — 제3의 compute 없이 캐시(또는 후임자)로 종결
        assert memo.embed_query("x") == [1.0]
        assert inner.calls == 2

        inner.gates[1].set()  # B(후임자) 완료 허용
        t_b.join(timeout=5.0)
        assert not t_b.is_alive()
        assert errors == []
        assert memo._in_flight == {}  # B가 자기 엔트리 pop(단일-flight 유지)


class TestToggle:
    def test_toggle_off_returns_unwrapped_embedder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """토글 off(MEMOIZE_EMBEDDING_QUERY=0) → 풀 우회 직접 호출로 래핑 안 됨."""
        from core.model_loader import load_embedding_model

        monkeypatch.setenv("IS_UNIT_TEST", "true")  # FakeEmbeddings 경로
        monkeypatch.setenv("MEMOIZE_EMBEDDING_QUERY", "0")

        result_off = load_embedding_model()
        assert not isinstance(result_off, MemoizingEmbedding)

        # 토글 on → 래핑 (fixture가 테스트 후 env 복원 + 레지스트리 clear)
        monkeypatch.setenv("MEMOIZE_EMBEDDING_QUERY", "1")
        result_on = load_embedding_model()
        assert isinstance(result_on, MemoizingEmbedding)
