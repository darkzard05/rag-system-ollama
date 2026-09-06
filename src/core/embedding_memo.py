"""쿼리 임베딩 중복 계산 제거용 LRU+TTL+단일-flight 메모이징 래퍼.

동일한 쿼리 원문이 한 번의 RAG 질의 사이클에서 임베더로 여러 번 직렬
임베딩될 수 있는 문제를 제거한다. 단일-flight(동일 텍스트 동시 중복 계산
차단) + LRU 캐시 + TTL 만료 + 실패 폭포 전파 + 타임아웃 단일 승계를 동기
``embed_query`` 시그니처 그대로 제공한다. ``embed_documents``는 순수
passthrough다(메모이제이션 대상 아님).
"""

from __future__ import annotations

import threading
import time
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from langchain_core.embeddings import Embeddings

# waiter가 계산자 완료를 기다리는 최대 시간(초). 초과 시 단일 승계(takeover)로
# 새 계산자가 되어 재계산한다. 0 이하의 값은 즉시 승계로 해석된다.
_WAIT_BOUND = 30.0

# LRU 캐시 기본 상한 및 TTL(초). 상한 초과 시 가장 오래된 항목부터 퇴출된다.
_DEFAULT_MAXSIZE = 1024
_DEFAULT_TTL = 300.0


@dataclass
class _InFlight:
    """진행 중 계산 마커 + 결과/실패 전파 구조체.

    ``vec``/``error``는 단계 2에서 포착한 객체에서 직접 읽어 락 재획득 없이
    waiter에게 배달된다. ``event.set()``은 성공/실패 단계(3/4)에서만 호출된다.
    """

    event: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    vec: np.ndarray | None = None


class MemoizingEmbedding(Embeddings):
    """동기 임베더 데코레이터 — 쿼리 임베딩 중복 계산 제거.

    같은 텍스트에 대한 ``embed_query`` 호출이 TTL 창 안에서는 캐시를 재사용하고,
    동시 요청은 단일 계산자에게 병합(단일-flight)된다. 계산은 반드시 락 밖에서
    수행되고(락은 구조 접근에만 사용), waiter의 ``Event.wait``도 락 밖에서
    수행되어 교착/루프 역전이 없다.
    """

    def __init__(
        self,
        inner: Embeddings,
        *,
        maxsize: int = _DEFAULT_MAXSIZE,
        ttl: float = _DEFAULT_TTL,
    ) -> None:
        self._inner = inner
        self._maxsize = maxsize
        self._ttl = ttl
        self._lock = threading.RLock()
        # key(쿼리 원문) -> (monotonic 등록 시각, float32 벡터)
        self._cache: OrderedDict[str, tuple[float, np.ndarray]] = OrderedDict()
        # key -> 진행 중 계산 마커(단일-flight)
        self._in_flight: dict[str, _InFlight] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "calls": 0,
            "in_flight": 0,
        }
        _memo_instances.add(self)

    @property
    def model(self) -> Any:
        """내부 임베더의 모델 식별자를 위임합니다.

        데코레이터 래퍼는 모델명을 노출하지 않으면 자신이 감싼 임베더의
        ``model``/``model_name`` 이 없으면 ``embedding_memo`` 캐시 키의 기준이
        ``default_model`` 로 폴백되어, 서로 다른 임베더(가짜/실제)가 같은 캐시
        키 공간을 공유하게 된다. (FAISS 차원 불일치 사고의 근본 원인)
        """
        return getattr(self._inner, "model", None)

    @property
    def model_name(self) -> Any:
        """내부 임베더의 모델명을 위임합니다 (``model`` 미보유 임베더 대비)."""
        return getattr(self._inner, "model_name", None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """메모이제이션 없이 내부 임베더로 순수 위임한다."""
        return self._inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """단일-flight 메모이징 쿼리 임베딩.

        단계 1(캐시) / 단계 2(단일-flight 판정)는 락 임계구간에서, 실제 계산과
        waiter 대기는 락 밖에서 수행된다. 타임아웃 시 항등 비교로 단일 승계자가
        되어 재계산하고, 나머지는 후임 체인을 따라 다시 대기한다.
        """
        with self._lock:
            entry = self._cache.get(text)
            if entry is not None:
                timestamp, vec = entry
                if time.monotonic() - timestamp <= self._ttl:
                    # LRU: 접근 시 최근 사용 위치로 이동
                    self._cache.move_to_end(text)
                    self._stats["hits"] += 1
                    return vec.tolist()
                del self._cache[text]
            self._stats["misses"] += 1
            existing = self._in_flight.get(text)
            if existing is not None:
                holder = existing
                is_registrant = False
            else:
                holder = _InFlight()
                self._in_flight[text] = holder
                self._stats["in_flight"] += 1
                is_registrant = True
        if is_registrant:
            # 이 스레드가 계산자 → 락 밖 compute
            return self._compute(text, holder)
        # 이 스레드는 waiter → 락 밖 wait + 타임아웃 시 단일 승계
        return self._await_query(text, holder)

    def _compute(self, text: str, in_flight: _InFlight) -> list[float]:
        """계산자 경로(단계 3/4): 락 미보유 상태에서만 호출된다."""
        with self._lock:
            self._stats["calls"] += 1
        try:
            vec = self._inner.embed_query(text)
        except BaseException as exc:
            # 단계 4: 실패 → error 기록 + 이벤트 set + 항등 가드 pop(캐시 미저장)
            with self._lock:
                in_flight.error = exc
                if self._in_flight.get(text) is in_flight:
                    self._in_flight.pop(text, None)
                    self._stats["in_flight"] -= 1
                in_flight.event.set()
            raise
        # 단계 3: 성공 → vec 기록 + 캐시 저장 + 항등 가드 pop + 이벤트 set
        arr = np.asarray(vec, dtype=np.float32)
        with self._lock:
            in_flight.vec = arr
            self._store_locked(text, arr)
            if self._in_flight.get(text) is in_flight:
                self._in_flight.pop(text, None)
                self._stats["in_flight"] -= 1
            in_flight.event.set()
        return in_flight.vec.tolist()

    def _await_query(self, text: str, captured: _InFlight) -> list[float]:
        """waiter 경로(단계 2): event.wait → set이면 직접 배달, timeout이면 승계."""
        while True:
            if captured.event.wait(timeout=_WAIT_BOUND):
                break
            # timeout — 단일 승계 시도 (항등 비교)
            successor: _InFlight | None = None
            with self._lock:
                current = self._in_flight.get(text)
                if current is captured:
                    # 승계 승자: stale 엔트리 교체 후 이 스레드가 새 계산자
                    self._in_flight.pop(text, None)
                    successor = _InFlight()
                    self._in_flight[text] = successor
                elif current is not None:
                    # 다른 waiter가 이미 승계 → 후임 체인으로 갱신 후 재대기
                    captured = current
            if successor is not None:
                return self._compute(text, successor)
        # 이벤트 set — 포착 객체에서 직접 배달(락 재획득 없음)
        if captured.error is not None:
            raise captured.error
        assert captured.vec is not None
        return captured.vec.tolist()

    def _store_locked(self, text: str, vec: np.ndarray) -> None:
        """락 보유 상태에서만 호출. TTL 만료 스캔 후 put, maxsize 초과 시 퇴출."""
        now = time.monotonic()
        for key in [k for k, (ts, _) in self._cache.items() if now - ts > self._ttl]:
            del self._cache[key]
        self._cache[text] = (now, vec)
        self._cache.move_to_end(text)
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
            self._stats["evictions"] += 1

    def clear(self) -> None:
        """캐시·단일-flight·통계 전부 초기화.

        진행 중인 계산자는 단계 3/4의 항등 가드 pop으로 후임자/신규 엔트리를
        건드리지 않는다.
        """
        with self._lock:
            self._cache.clear()
            self._in_flight.clear()
            self._stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "calls": 0,
                "in_flight": 0,
            }

    def stats(self) -> dict[str, int]:
        """현재 통계 사본을 반환한다."""
        with self._lock:
            return dict(self._stats)


# weakref 레지스트리: 생성된 모든 래퍼를 약참조로 추적한다(테스트 격리용).
_memo_instances: weakref.WeakSet[MemoizingEmbedding] = weakref.WeakSet()


def clear_memo_instances() -> None:
    """등록된 모든 래퍼 인스턴스의 상태를 초기화한다.

    pytest 프로세스에서는 모델 풀 공유로 동일 래퍼가 테스트 간 잔존하므로,
    ``tests/unit/conftest.py``의 autouse 픽스처가 각 테스트 후 이 함수를 호출해
    테스트 간 메모이 오염을 차단한다.
    """
    for instance in list(_memo_instances):
        instance.clear()
