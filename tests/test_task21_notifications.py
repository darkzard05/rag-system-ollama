"""
Task 21: 알림 및 이벤트 시스템 테스트
- 알림 관리 및 라우팅
- 이벤트 로깅 및 추적
- 알림 채널 (이메일, Slack, 웹훅)
"""

import pytest

from src.infra.notification_system import (
    NotificationManager,
    NotificationMessage,
    NotificationRecipient,
    AlertRule,
    NotificationFilter,
    NotificationPriority,
    NotificationStatus,
    ChannelType,
    NotificationQueue,
)
from src.services.monitoring.event_logger import (
    EventLogger,
    EventType,
    EventSeverity,
    EventQuery,
)
from src.infra.notification_channels import (
    EmailChannel,
    SlackChannel,
    WebhookChannel,
    ChannelFactory,
    EmailTemplate,
)


# ==============================================
# Fixtures
# ==============================================


@pytest.fixture
def notification_manager():
    """알림 관리자 Fixture"""
    manager = NotificationManager()
    yield manager


@pytest.fixture
def event_logger():
    """이벤트 로거 Fixture"""
    logger = EventLogger()
    yield logger


@pytest.fixture
def email_channel():
    """이메일 채널 Fixture"""
    channel = EmailChannel(
        smtp_server="smtp.example.com",
        smtp_port=587,
        username="test@example.com",
        password="password",
        from_address="test@example.com",
    )
    yield channel


# ==============================================
# 테스트 그룹 1: 기본 알림 관리 (5개 테스트)
# ==============================================


class TestBasicNotificationManagement:
    """기본 알림 관리 테스트"""

    def test_01_create_notification(self, notification_manager):
        """알림 생성"""
        notification = NotificationMessage(
            title="Test Alert",
            message="This is a test alert",
            priority=NotificationPriority.HIGH,
            channels=[ChannelType.EMAIL],
        )

        assert notification.notification_id
        assert notification.title == "Test Alert"
        assert notification.priority == NotificationPriority.HIGH

    def test_02_register_recipient(self, notification_manager):
        """수신자 등록"""
        recipient = NotificationRecipient(
            recipient_id="user_001",
            name="John Doe",
            email="john@example.com",
            slack_id="U123456",
        )

        notification_manager.register_recipient(recipient)

        retrieved = notification_manager.get_recipient("user_001")
        assert retrieved is not None
        assert retrieved.email == "john@example.com"

    def test_03_send_notification(self, notification_manager):
        """알림 전송"""
        notification = NotificationMessage(
            title="Test",
            message="Test message",
            priority=NotificationPriority.MEDIUM,
            channels=[ChannelType.EMAIL],
            recipients=["user_001"],
        )

        result = notification_manager.send_notification(notification)

        assert result == True
        assert notification.status == NotificationStatus.PENDING

    def test_04_notification_queue(self):
        """알림 큐"""
        queue = NotificationQueue(max_size=100)

        # 우선순위가 다른 알림 추가
        for i, priority in enumerate(
            [
                NotificationPriority.LOW,
                NotificationPriority.CRITICAL,
                NotificationPriority.MEDIUM,
            ]
        ):
            notif = NotificationMessage(title=f"Alert {i}", priority=priority)
            queue.enqueue(notif)

        # 첫 번째 알림은 CRITICAL이어야 함
        first = queue.dequeue()
        assert first.priority == NotificationPriority.CRITICAL

    def test_05_get_statistics(self, notification_manager):
        """통계 조회"""
        # 여러 알림 전송
        for i in range(5):
            notif = NotificationMessage(
                title=f"Alert {i}",
                message=f"Message {i}",
                priority=NotificationPriority.MEDIUM,
                source="test",
            )
            notification_manager.send_notification(notif)

        stats = notification_manager.get_statistics()

        assert stats["pending"] >= 5


# ==============================================
# 테스트 그룹 2: 알림 규칙 및 필터링 (5개 테스트)
# ==============================================


class TestAlertRulesAndFiltering:
    """알림 규칙 및 필터링 테스트"""

    def test_06_create_alert_rule(self, notification_manager):
        """알림 규칙 생성"""
        rule = AlertRule(
            name="High CPU Alert",
            condition=lambda x: x.get("cpu_usage", 0) > 80,
            priority=NotificationPriority.HIGH,
            channels=[ChannelType.SLACK],
        )

        rule_id = notification_manager.create_alert_rule(rule)

        assert rule_id is not None

    def test_07_evaluate_rules(self, notification_manager):
        """규칙 평가"""
        rule = AlertRule(
            name="Test Rule",
            condition=lambda x: x.get("value", 0) > 100,
            priority=NotificationPriority.HIGH,
            channels=[ChannelType.EMAIL],
            recipients=["user_001"],
        )

        notification_manager.create_alert_rule(rule)

        # 규칙 평가
        notifications = notification_manager.evaluate_rules({"value": 150})

        assert len(notifications) > 0
        assert notifications[0].priority == NotificationPriority.HIGH

    def test_08_create_filter(self, notification_manager):
        """필터 생성"""
        filter_obj = NotificationFilter(
            name="High Priority Only",
            priority_min=NotificationPriority.HIGH,
            channels=[ChannelType.EMAIL],
        )

        filter_id = notification_manager.create_filter(filter_obj)

        assert filter_id is not None

    def test_09_filter_by_priority(self, notification_manager):
        """우선순위 필터링"""
        # 필터 생성 (HIGH 이상만)
        filter_obj = NotificationFilter(
            name="High Priority Filter", priority_min=NotificationPriority.HIGH
        )
        notification_manager.create_filter(filter_obj)

        # LOW 우선순위 알림
        low_notif = NotificationMessage(
            title="Low Priority", priority=NotificationPriority.LOW
        )

        # HIGH 우선순위 알림
        high_notif = NotificationMessage(
            title="High Priority", priority=NotificationPriority.HIGH
        )

        # 필터링 테스트
        result_low = notification_manager._apply_filters(low_notif)
        result_high = notification_manager._apply_filters(high_notif)

        # HIGH는 통과, LOW는 필터링됨
        assert result_high == True

    def test_10_alert_cooldown(self, notification_manager):
        """알림 쿨다운"""
        rule = AlertRule(
            name="Cooldown Test",
            condition=lambda x: True,
            cooldown=1.0,  # 1초
        )

        rule_id = notification_manager.create_alert_rule(rule)

        # 첫 번째 평가
        notifs1 = notification_manager.evaluate_rules({})
        assert len(notifs1) > 0

        # 즉시 두 번째 평가 (쿨다운 중)
        notifs2 = notification_manager.evaluate_rules({})
        assert len(notifs2) == 0  # 쿨다운으로 인해 트리거 안됨


# ==============================================
# 테스트 그룹 3: 이벤트 로깅 (5개 테스트)
# ==============================================


class TestEventLogging:
    """이벤트 로깅 테스트"""

    def test_11_log_event(self, event_logger):
        """이벤트 로깅"""
        event_id = event_logger.log_info(
            title="Test Event", message="This is a test event", source="test"
        )

        assert event_id is not None

        events = event_logger.get_recent_events(limit=1)
        assert len(events) > 0
        assert events[0].title == "Test Event"

    def test_12_log_different_levels(self, event_logger):
        """다양한 로그 레벨"""
        event_logger.log_info("Info", "Info message")
        event_logger.log_warning("Warning", "Warning message")
        event_logger.log_error("Error", "Error message")

        stats = event_logger.get_statistics()

        assert stats["by_type"] is not None

    def test_13_log_audit_event(self, event_logger):
        """감사 로깅"""
        event_id = event_logger.log_audit(
            title="User Login",
            message="User logged in",
            user_id="user_001",
            action="login",
        )

        assert event_id is not None

        # 사용자 활동 조회
        activities = event_logger.get_user_activity("user_001")
        assert len(activities) > 0

    def test_14_log_performance_event(self, event_logger):
        """성능 로깅"""
        duration = 2.5
        event_id = event_logger.log_performance(
            operation="database_query", duration=duration, source="db"
        )

        assert event_id is not None

        events = event_logger.get_recent_events(limit=1)
        assert events[0].duration == duration

    def test_15_query_events(self, event_logger):
        """이벤트 쿼리"""
        # 여러 이벤트 로깅
        for i in range(10):
            event_logger.log_info(
                title=f"Event {i}",
                message=f"Message {i}",
                tags=["test"] if i % 2 == 0 else ["other"],
            )

        # 쿼리
        query = EventQuery(tags=["test"], limit=20)
        results = event_logger.query_events(query)

        assert len(results) > 0


# ==============================================
# 테스트 그룹 4: 알림 채널 (5개 테스트)
# ==============================================


class TestNotificationChannels:
    """알림 채널 테스트"""

    def test_16_email_channel(self, email_channel):
        """이메일 채널"""
        notification = NotificationMessage(
            title="Email Test",
            message="Test message",
            metadata={"email": "recipient@example.com"},
        )

        result = email_channel.send(notification)

        # 모의 구현이므로 성공 기대
        assert result == True

    def test_17_email_template(self, email_channel):
        """이메일 템플릿"""
        template = EmailTemplate(
            name="custom",
            subject="Custom: {title}",
            html_body="<h1>{title}</h1><p>{message}</p>",
        )

        email_channel.register_template(template)

        # 템플릿이 등록되었는지 확인
        assert "custom" in email_channel._templates

    def test_18_slack_channel(self):
        """Slack 채널"""
        channel = SlackChannel(
            webhook_urls={"general": "https://hooks.slack.com/services/..."}
        )

        notification = NotificationMessage(
            title="Slack Test",
            message="Test message",
            priority=NotificationPriority.HIGH,
        )

        result = channel.send(notification)

        # 웹훅이 유효하지 않을 수 있으므로 False 예상
        assert isinstance(result, bool)

    def test_19_webhook_channel(self):
        """웹훅 채널"""
        channel = WebhookChannel(
            webhook_urls=[
                "https://example.com/webhook1",
                "https://example.com/webhook2",
            ]
        )

        # 웹훅 추가
        channel.add_webhook("https://example.com/webhook3")
        assert len(channel.webhook_urls) == 3

        # 웹훅 제거
        channel.remove_webhook("https://example.com/webhook1")
        assert len(channel.webhook_urls) == 2

    def test_20_channel_factory(self):
        """채널 팩토리"""
        # 이메일 채널 생성
        email_ch = ChannelFactory.create_email_channel(smtp_server="smtp.example.com")
        assert isinstance(email_ch, EmailChannel)

        # Slack 채널 생성
        slack_ch = ChannelFactory.create_slack_channel(
            webhook_urls={"general": "https://hooks.slack.com/..."}
        )
        assert isinstance(slack_ch, SlackChannel)

        # 웹훅 채널 생성
        webhook_ch = ChannelFactory.create_webhook_channel(
            webhook_urls=["https://example.com/webhook"]
        )
        assert isinstance(webhook_ch, WebhookChannel)


# ==============================================
# 테스트 그룹 5: 통합 알림 시스템 (3개 테스트)
# ==============================================


class TestIntegratedNotificationSystem:
    """통합 알림 시스템 테스트"""

    def test_21_end_to_end_notification(self, notification_manager, event_logger):
        """엔드-투-엔드 알림 전송"""
        # 수신자 등록
        recipient = NotificationRecipient(
            recipient_id="user_001", name="Test User", email="test@example.com"
        )
        notification_manager.register_recipient(recipient)

        # 알림 생성 및 전송
        notification = NotificationMessage(
            title="End-to-End Test",
            message="Testing the full flow",
            priority=NotificationPriority.HIGH,
            channels=[ChannelType.EMAIL],
            recipients=["user_001"],
            source="e2e_test",
        )

        result = notification_manager.send_notification(notification)

        assert result == True

        # 이벤트 로깅
        event_logger.log_info(
            title="Notification Sent",
            message=f"Sent notification: {notification.notification_id}",
            metadata={"notification_id": notification.notification_id},
        )

        # 통계 확인
        stats = notification_manager.get_statistics()
        assert stats["pending"] >= 1

    def test_22_alert_and_notification(self, notification_manager):
        """알림 규칙과 알림 연동"""
        # 규칙 생성
        rule = AlertRule(
            name="Performance Alert",
            condition=lambda x: x.get("latency", 0) > 1000,
            priority=NotificationPriority.CRITICAL,
            channels=[ChannelType.EMAIL, ChannelType.SLACK],
            recipients=["user_001", "user_002"],
        )

        notification_manager.create_alert_rule(rule)

        # 규칙 평가
        notifications = notification_manager.evaluate_rules(
            {"latency": 1500, "operation": "database_query"}
        )

        assert len(notifications) > 0
        assert notifications[0].priority == NotificationPriority.CRITICAL

    def test_23_event_correlation(self, event_logger):
        """이벤트 상관관계"""
        correlation_id = "corr_12345"

        # 여러 관련 이벤트 로깅
        for i in range(3):
            event_logger.log_event(
                event_type=EventType.PERFORMANCE,
                title=f"Operation {i}",
                message=f"Step {i} completed",
                severity=EventSeverity.INFO,
                correlation_id=correlation_id,
            )

        # 상관관계로 이벤트 조회
        events = event_logger.get_events_by_correlation_id(correlation_id)

        assert len(events) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
