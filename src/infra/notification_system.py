"""
Task 21-1: Notification System Module
알림 관리 및 라우팅 시스템
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timedelta
import time
from threading import RLock, Thread
import json
import uuid


class NotificationPriority(Enum):
    """알림 우선순위"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class NotificationStatus(Enum):
    """알림 상태"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    ACKNOWLEDGED = "acknowledged"


class ChannelType(Enum):
    """채널 타입"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"


@dataclass
class NotificationRecipient:
    """알림 수신자"""
    recipient_id: str
    name: str
    email: Optional[str] = None
    slack_id: Optional[str] = None
    webhook_url: Optional[str] = None
    phone: Optional[str] = None
    push_token: Optional[str] = None
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationMessage:
    """알림 메시지"""
    notification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    message: str = ""
    priority: NotificationPriority = NotificationPriority.MEDIUM
    status: NotificationStatus = NotificationStatus.PENDING
    source: str = ""
    tags: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    scheduled_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    channels: List[ChannelType] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class AlertRule:
    """알림 규칙"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    condition: Callable = field(default_factory=lambda: lambda x: False)
    priority: NotificationPriority = NotificationPriority.MEDIUM
    enabled: bool = True
    channels: List[ChannelType] = field(default_factory=list)
    recipients: List[str] = field(default_factory=list)
    cooldown: float = 60.0  # 초
    last_triggered: Optional[float] = None
    trigger_count: int = 0


@dataclass
class NotificationFilter:
    """알림 필터"""
    filter_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    priority_min: Optional[NotificationPriority] = None
    priority_max: Optional[NotificationPriority] = None
    tags: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    channels: List[ChannelType] = field(default_factory=list)
    enabled: bool = True


class NotificationQueue:
    """알림 큐 (우선순위 기반)"""
    
    PRIORITY_ORDER = {
        NotificationPriority.CRITICAL: 1,
        NotificationPriority.HIGH: 2,
        NotificationPriority.MEDIUM: 3,
        NotificationPriority.LOW: 4,
        NotificationPriority.INFO: 5
    }
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._queue: List[NotificationMessage] = []
        self._lock = RLock()
    
    def enqueue(self, notification: NotificationMessage):
        """알림 추가"""
        with self._lock:
            if len(self._queue) >= self.max_size:
                return False
            
            self._queue.append(notification)
            # 우선순위 순으로 정렬
            self._queue.sort(
                key=lambda x: (
                    self.PRIORITY_ORDER.get(x.priority, 999),
                    x.timestamp
                )
            )
            return True
    
    def dequeue(self) -> Optional[NotificationMessage]:
        """알림 추출"""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)
    
    def size(self) -> int:
        """큐 크기"""
        with self._lock:
            return len(self._queue)
    
    def peek(self) -> Optional[NotificationMessage]:
        """첫 번째 알림 확인"""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[0]


class NotificationManager:
    """알림 관리자"""
    
    def __init__(self, num_workers: int = 3):
        self.num_workers = num_workers
        self._queue = NotificationQueue()
        self._recipients: Dict[str, NotificationRecipient] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._filters: Dict[str, NotificationFilter] = {}
        self._history: List[NotificationMessage] = []
        self._max_history = 10000
        self._channels: Dict[ChannelType, Callable] = {}
        self._lock = RLock()
        self._running = False
        self._worker_threads: List[Thread] = []
    
    def register_recipient(self, recipient: NotificationRecipient):
        """수신자 등록"""
        with self._lock:
            self._recipients[recipient.recipient_id] = recipient
    
    def unregister_recipient(self, recipient_id: str) -> bool:
        """수신자 등록 해제"""
        with self._lock:
            if recipient_id in self._recipients:
                del self._recipients[recipient_id]
                return True
            return False
    
    def get_recipient(self, recipient_id: str) -> Optional[NotificationRecipient]:
        """수신자 조회"""
        with self._lock:
            return self._recipients.get(recipient_id)
    
    def create_alert_rule(self, rule: AlertRule) -> str:
        """알림 규칙 생성"""
        with self._lock:
            self._rules[rule.rule_id] = rule
            return rule.rule_id
    
    def delete_alert_rule(self, rule_id: str) -> bool:
        """알림 규칙 삭제"""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
            return False
    
    def evaluate_rules(self, data: Any) -> List[NotificationMessage]:
        """규칙 평가"""
        notifications = []
        
        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                
                # 쿨다운 확인
                if rule.last_triggered:
                    elapsed = time.time() - rule.last_triggered
                    if elapsed < rule.cooldown:
                        continue
                
                # 조건 평가
                try:
                    if rule.condition(data):
                        # 알림 생성
                        notification = NotificationMessage(
                            title=f"Alert: {rule.name}",
                            message=f"Rule '{rule.name}' triggered",
                            priority=rule.priority,
                            source="alert_rule",
                            channels=rule.channels,
                            recipients=rule.recipients,
                            metadata={"rule_id": rule.rule_id, "data": data}
                        )
                        
                        notifications.append(notification)
                        rule.last_triggered = time.time()
                        rule.trigger_count += 1
                
                except Exception:
                    pass
        
        return notifications
    
    def send_notification(self, notification: NotificationMessage) -> bool:
        """알림 전송"""
        with self._lock:
            # 필터 적용
            if not self._apply_filters(notification):
                return False
            
            # 큐에 추가
            if self._queue.enqueue(notification):
                self._add_to_history(notification)
                return True
            return False
    
    def _apply_filters(self, notification: NotificationMessage) -> bool:
        """필터 적용"""
        for filter_obj in self._filters.values():
            if not filter_obj.enabled:
                continue
            
            # 우선순위 확인
            if filter_obj.priority_min:
                priority_values = {
                    NotificationPriority.CRITICAL: 5,
                    NotificationPriority.HIGH: 4,
                    NotificationPriority.MEDIUM: 3,
                    NotificationPriority.LOW: 2,
                    NotificationPriority.INFO: 1
                }
                min_val = priority_values.get(filter_obj.priority_min, 0)
                curr_val = priority_values.get(notification.priority, 0)
                
                if curr_val < min_val:
                    continue
            
            # 태그 확인
            if filter_obj.tags:
                if not any(tag in notification.tags for tag in filter_obj.tags):
                    continue
            
            # 소스 확인
            if filter_obj.sources:
                if notification.source not in filter_obj.sources:
                    continue
            
            # 채널 확인
            if filter_obj.channels:
                if not any(ch in notification.channels for ch in filter_obj.channels):
                    continue
            
            return True
        
        return True
    
    def create_filter(self, filter_obj: NotificationFilter) -> str:
        """필터 생성"""
        with self._lock:
            self._filters[filter_obj.filter_id] = filter_obj
            return filter_obj.filter_id
    
    def delete_filter(self, filter_id: str) -> bool:
        """필터 삭제"""
        with self._lock:
            if filter_id in self._filters:
                del self._filters[filter_id]
                return True
            return False
    
    def _add_to_history(self, notification: NotificationMessage):
        """히스토리에 추가"""
        self._history.append(notification)
        if len(self._history) > self._max_history:
            self._history.pop(0)
    
    def get_history(self, limit: int = 100) -> List[NotificationMessage]:
        """히스토리 조회"""
        with self._lock:
            return self._history[-limit:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 조회"""
        with self._lock:
            total_sent = len(self._history)
            by_priority = {}
            by_status = {}
            by_source = {}
            
            for notif in self._history:
                # 우선순위별
                priority_key = notif.priority.value
                by_priority[priority_key] = by_priority.get(priority_key, 0) + 1
                
                # 상태별
                status_key = notif.status.value
                by_status[status_key] = by_status.get(status_key, 0) + 1
                
                # 소스별
                source_key = notif.source
                by_source[source_key] = by_source.get(source_key, 0) + 1
            
            return {
                'total_sent': total_sent,
                'pending': self._queue.size(),
                'by_priority': by_priority,
                'by_status': by_status,
                'by_source': by_source,
                'recipients': len(self._recipients),
                'rules': len(self._rules),
                'filters': len(self._filters)
            }
    
    def update_notification_status(
        self, 
        notification_id: str, 
        status: NotificationStatus
    ) -> bool:
        """알림 상태 업데이트"""
        with self._lock:
            for notif in self._history:
                if notif.notification_id == notification_id:
                    notif.status = status
                    return True
            return False
    
    def register_channel(
        self, 
        channel_type: ChannelType, 
        handler: Callable
    ):
        """채널 등록"""
        with self._lock:
            self._channels[channel_type] = handler
    
    def send_via_channel(
        self, 
        notification: NotificationMessage, 
        channel: ChannelType
    ) -> bool:
        """특정 채널로 전송"""
        with self._lock:
            if channel not in self._channels:
                return False
            
            try:
                handler = self._channels[channel]
                result = handler(notification)
                
                if result:
                    self.update_notification_status(
                        notification.notification_id,
                        NotificationStatus.SENT
                    )
                else:
                    notification.retry_count += 1
                
                return result
            except Exception:
                notification.retry_count += 1
                return False
    
    def get_pending_notifications(self) -> List[NotificationMessage]:
        """대기 중인 알림 조회"""
        pending = []
        
        for _ in range(self._queue.size()):
            notif = self._queue.dequeue()
            if notif:
                pending.append(notif)
                self._queue.enqueue(notif)  # 다시 추가
        
        return pending
    
    def acknowledge_notification(self, notification_id: str) -> bool:
        """알림 확인"""
        return self.update_notification_status(
            notification_id,
            NotificationStatus.ACKNOWLEDGED
        )
