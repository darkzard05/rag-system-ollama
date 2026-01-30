"""
Task 21-2: Event Logger Module
이벤트 로깅 및 추적 시스템
"""

import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class EventType(Enum):
    """이벤트 타입"""

    SYSTEM = "system"
    USER = "user"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    NOTIFICATION = "notification"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AUDIT = "audit"


class EventSeverity(Enum):
    """이벤트 심각도"""

    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"


@dataclass
class Event:
    """이벤트"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.INFO
    severity: EventSeverity = EventSeverity.INFO
    title: str = ""
    message: str = ""
    source: str = ""
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    user_id: str | None = None
    session_id: str | None = None
    correlation_id: str | None = None


@dataclass
class EventQuery:
    """이벤트 쿼리"""

    event_type: EventType | None = None
    severity: EventSeverity | None = None
    source: str | None = None
    user_id: str | None = None
    tags: list[str] = field(default_factory=list)
    start_time: float | None = None
    end_time: float | None = None
    limit: int = 100


class EventStore:
    """이벤트 저장소"""

    def __init__(self, max_events: int = 100000):
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._lock = RLock()

        # 인덱스
        self._by_type: dict[EventType, list[str]] = defaultdict(list)
        self._by_severity: dict[EventSeverity, list[str]] = defaultdict(list)
        self._by_source: dict[str, list[str]] = defaultdict(list)
        self._by_user: dict[str, list[str]] = defaultdict(list)
        self._by_session: dict[str, list[str]] = defaultdict(list)

        # 이벤트 ID to Event 매핑
        self._events_by_id: dict[str, Event] = {}

    def add_event(self, event: Event):
        """이벤트 추가"""
        with self._lock:
            self._events.append(event)
            self._events_by_id[event.event_id] = event

            # 인덱스 업데이트
            self._by_type[event.event_type].append(event.event_id)
            self._by_severity[event.severity].append(event.event_id)
            self._by_source[event.source].append(event.event_id)

            if event.user_id:
                self._by_user[event.user_id].append(event.event_id)

            if event.session_id:
                self._by_session[event.session_id].append(event.event_id)

    def get_event(self, event_id: str) -> Event | None:
        """이벤트 조회"""
        with self._lock:
            return self._events_by_id.get(event_id)

    def query(self, query: EventQuery) -> list[Event]:
        """이벤트 쿼리"""
        with self._lock:
            results = []

            for event in self._events:
                # 필터 적용
                if query.event_type and event.event_type != query.event_type:
                    continue
                if query.severity and event.severity != query.severity:
                    continue
                if query.source and event.source != query.source:
                    continue
                if query.user_id and event.user_id != query.user_id:
                    continue
                if query.tags:
                    if not any(tag in event.tags for tag in query.tags):
                        continue
                if query.start_time and event.timestamp < query.start_time:
                    continue
                if query.end_time and event.timestamp > query.end_time:
                    continue

                results.append(event)

            # 시간 역순 정렬 (최신순)
            results.sort(key=lambda x: x.timestamp, reverse=True)

            return results[: query.limit]

    def get_count_by_type(self) -> dict[str, int]:
        """타입별 개수"""
        with self._lock:
            return {
                event_type.value: len(event_ids)
                for event_type, event_ids in self._by_type.items()
            }

    def get_count_by_severity(self) -> dict[str, int]:
        """심각도별 개수"""
        with self._lock:
            return {
                severity.value: len(event_ids)
                for severity, event_ids in self._by_severity.items()
            }

    def clear(self):
        """이벤트 초기화"""
        with self._lock:
            self._events.clear()
            self._events_by_id.clear()
            self._by_type.clear()
            self._by_severity.clear()
            self._by_source.clear()
            self._by_user.clear()
            self._by_session.clear()


class EventLogger:
    """이벤트 로거"""

    def __init__(self, max_events: int = 100000):
        self._store = EventStore(max_events)
        self._lock = RLock()
        self._handlers: list[Callable] = []

    def log_event(
        self,
        event_type: EventType,
        title: str,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        source: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        """이벤트 로깅"""
        event = Event(
            event_type=event_type,
            severity=severity,
            title=title,
            message=message,
            source=source,
            metadata=metadata or {},
            tags=tags or [],
            user_id=user_id,
            session_id=session_id,
            correlation_id=correlation_id,
        )

        self._store.add_event(event)

        # 핸들러 호출
        with self._lock:
            for handler in self._handlers:
                try:
                    handler(event)
                except Exception:
                    pass

        return event.event_id

    def log_info(self, title: str, message: str, source: str = "", **kwargs) -> str:
        """INFO 레벨 로깅"""
        return self.log_event(
            EventType.INFO,
            title,
            message,
            severity=EventSeverity.INFO,
            source=source,
            **kwargs,
        )

    def log_warning(self, title: str, message: str, source: str = "", **kwargs) -> str:
        """WARNING 레벨 로깅"""
        return self.log_event(
            EventType.WARNING,
            title,
            message,
            severity=EventSeverity.WARNING,
            source=source,
            **kwargs,
        )

    def log_error(self, title: str, message: str, source: str = "", **kwargs) -> str:
        """ERROR 레벨 로깅"""
        return self.log_event(
            EventType.ERROR,
            title,
            message,
            severity=EventSeverity.ERROR,
            source=source,
            **kwargs,
        )

    def log_audit(
        self,
        title: str,
        message: str,
        user_id: str | None = None,
        action: str | None = None,
        **kwargs,
    ) -> str:
        """감사 로깅"""
        metadata = kwargs.pop("metadata", {})
        if action:
            metadata["action"] = action

        return self.log_event(
            EventType.AUDIT,
            title,
            message,
            severity=EventSeverity.INFO,
            source="audit",
            metadata=metadata,
            user_id=user_id,
            **kwargs,
        )

    def log_performance(
        self, operation: str, duration: float, source: str = "", **kwargs
    ) -> str:
        """성능 로깅"""
        metadata = kwargs.pop("metadata", {})
        metadata["operation"] = operation
        metadata["duration"] = duration

        event = self.log_event(
            EventType.PERFORMANCE,
            f"Performance: {operation}",
            f"Operation took {duration:.2f}s",
            severity=EventSeverity.INFO,
            source=source,
            metadata=metadata,
            **kwargs,
        )

        # Event 객체에도 duration 설정
        stored_event = self._store.get_event(event)
        if stored_event:
            stored_event.duration = duration

        return event

    def register_handler(self, handler: Callable):
        """핸들러 등록"""
        with self._lock:
            self._handlers.append(handler)

    def unregister_handler(self, handler: Callable):
        """핸들러 등록 해제"""
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def query_events(self, query: EventQuery) -> list[Event]:
        """이벤트 쿼리"""
        return self._store.query(query)

    def get_recent_events(self, limit: int = 100) -> list[Event]:
        """최근 이벤트 조회"""
        query = EventQuery(limit=limit)
        return self._store.query(query)

    def get_statistics(self) -> dict[str, Any]:
        """통계 조회"""
        with self._lock:
            return {
                "total_events": sum(len(ids) for ids in self._store._by_type.values()),
                "by_type": self._store.get_count_by_type(),
                "by_severity": self._store.get_count_by_severity(),
                "handlers": len(self._handlers),
            }

    def get_events_by_correlation_id(self, correlation_id: str) -> list[Event]:
        """correlation ID로 이벤트 조회"""
        query = EventQuery(limit=10000)
        all_events = self._store.query(query)

        return [event for event in all_events if event.correlation_id == correlation_id]

    def get_user_activity(
        self,
        user_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[Event]:
        """사용자 활동 조회"""
        query = EventQuery(
            user_id=user_id, start_time=start_time, end_time=end_time, limit=limit
        )
        return self._store.query(query)

    def get_session_events(self, session_id: str, limit: int = 1000) -> list[Event]:
        """세션 이벤트 조회"""
        query = EventQuery(limit=limit)
        all_events = self._store.query(query)

        return [event for event in all_events if event.session_id == session_id]


class PerformanceMonitor:
    """성능 모니터"""

    def __init__(self, logger: EventLogger):
        self.logger = logger
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._lock = RLock()

    def record_metric(self, operation: str, duration: float):
        """메트릭 기록"""
        with self._lock:
            self._metrics[operation].append(duration)

            # 로깅
            self.logger.log_performance(operation, duration, source="monitor")

    def get_average(self, operation: str) -> float | None:
        """평균 시간"""
        with self._lock:
            if operation not in self._metrics or not self._metrics[operation]:
                return None

            return sum(self._metrics[operation]) / len(self._metrics[operation])

    def get_stats(self, operation: str) -> dict[str, Any]:
        """통계"""
        with self._lock:
            if operation not in self._metrics or not self._metrics[operation]:
                return {}

            durations = self._metrics[operation]
            durations_sorted = sorted(durations)

            return {
                "operation": operation,
                "count": len(durations),
                "average": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations),
                "p50": durations_sorted[len(durations) // 2],
                "p95": durations_sorted[int(len(durations) * 0.95)],
                "p99": durations_sorted[int(len(durations) * 0.99)],
            }
