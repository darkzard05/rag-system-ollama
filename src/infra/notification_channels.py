"""
Task 21-3: Notification Channels Module
다양한 알림 채널 구현 (이메일, Slack, 웹훅)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from threading import RLock
import requests


@dataclass
class EmailTemplate:
    """이메일 템플릿"""

    name: str
    subject: str
    html_body: str
    text_body: Optional[str] = None
    variables: List[str] = field(default_factory=list)


@dataclass
class ChannelConfig:
    """채널 설정"""

    enabled: bool = True
    retry_attempts: int = 3
    timeout: float = 10.0
    rate_limit: int = 100  # 초당 메시지 수
    batch_size: int = 1  # 배치 크기


class NotificationChannel(ABC):
    """알림 채널 기본 클래스"""

    def __init__(self, config: ChannelConfig):
        self.config = config
        self._lock = RLock()
        self._message_count = 0
        self._last_reset_time = 0
        self._sent_count = 0
        self._failed_count = 0

    @abstractmethod
    def send(self, notification: Any) -> bool:
        """알림 전송"""
        pass

    def can_send(self) -> bool:
        """전송 가능 여부"""
        return self.config.enabled

    def record_sent(self):
        """전송 기록"""
        with self._lock:
            self._sent_count += 1
            self._message_count += 1

    def record_failed(self):
        """실패 기록"""
        with self._lock:
            self._failed_count += 1
            self._message_count += 1

    def get_stats(self) -> Dict[str, Any]:
        """통계"""
        with self._lock:
            return {
                "sent": self._sent_count,
                "failed": self._failed_count,
                "total": self._message_count,
                "success_rate": (
                    self._sent_count / self._message_count
                    if self._message_count > 0
                    else 0
                ),
            }


class EmailChannel(NotificationChannel):
    """이메일 채널"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_address: str = "",
        config: Optional[ChannelConfig] = None,
    ):
        super().__init__(config or ChannelConfig())
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_address = from_address
        self._templates: Dict[str, EmailTemplate] = {}

        # 기본 템플릿
        self._register_default_templates()

    def _register_default_templates(self):
        """기본 템플릿 등록"""
        self._templates["notification"] = EmailTemplate(
            name="notification",
            subject="[{priority}] {title}",
            html_body="<h2>{title}</h2><p>{message}</p><p>Time: {timestamp}</p>",
            text_body="{title}\n{message}\nTime: {timestamp}",
        )

        self._templates["alert"] = EmailTemplate(
            name="alert",
            subject="[ALERT] {title}",
            html_body='<h2 style="color: red;">{title}</h2><p>{message}</p>',
            text_body="[ALERT] {title}\n{message}",
        )

    def register_template(self, template: EmailTemplate):
        """템플릿 등록"""
        with self._lock:
            self._templates[template.name] = template

    def send(self, notification: Any) -> bool:
        """이메일 전송"""
        if not self.can_send():
            return False

        try:
            # 수신자 조회
            recipients = self._get_email_recipients(notification)
            if not recipients:
                return False

            # 메시지 구성
            template = self._templates.get("notification")
            subject = self._render_template(template.subject, notification)
            html_body = self._render_template(template.html_body, notification)

            # 이메일 발송
            for recipient in recipients:
                self._send_email(recipient, subject, html_body)

            self.record_sent()
            return True

        except Exception:
            self.record_failed()
            return False

    def _get_email_recipients(self, notification: Any) -> List[str]:
        """이메일 수신자 조회"""
        recipients = []

        if hasattr(notification, "recipients"):
            for recipient_id in notification.recipients:
                # 실제 구현에서는 수신자 정보 조회
                if "@" in str(recipient_id):
                    recipients.append(recipient_id)

        if hasattr(notification, "metadata"):
            if "email" in notification.metadata:
                recipients.append(notification.metadata["email"])

        return recipients

    def _render_template(self, template: str, data: Any) -> str:
        """템플릿 렌더링"""
        result = template

        # 기본 필드
        if hasattr(data, "title"):
            result = result.replace("{title}", str(data.title))
        if hasattr(data, "message"):
            result = result.replace("{message}", str(data.message))
        if hasattr(data, "priority"):
            result = result.replace("{priority}", str(data.priority.value))

        # 메타데이터
        if hasattr(data, "metadata"):
            for key, value in data.metadata.items():
                result = result.replace(f"{{{key}}}", str(value))

        # 시간
        result = result.replace(
            "{timestamp}",
            datetime.fromtimestamp(getattr(data, "timestamp", 0)).isoformat(),
        )

        return result

    def _send_email(self, recipient: str, subject: str, html_body: str) -> bool:
        """실제 이메일 발송"""
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.from_address
            msg["To"] = recipient

            # HTML 부분
            html_part = MIMEText(html_body, "html")
            msg.attach(html_part)

            # SMTP 전송 (모의 구현)
            # 실제 환경에서는 여기서 SMTP 서버 연결
            return True

        except Exception:
            return False


class SlackChannel(NotificationChannel):
    """Slack 채널"""

    def __init__(
        self, webhook_urls: Dict[str, str], config: Optional[ChannelConfig] = None
    ):
        super().__init__(config or ChannelConfig())
        self.webhook_urls = webhook_urls

    def add_webhook(self, channel_name: str, webhook_url: str):
        """웹훅 추가"""
        with self._lock:
            self.webhook_urls[channel_name] = webhook_url

    def send(self, notification: Any) -> bool:
        """Slack으로 전송"""
        if not self.can_send():
            return False

        try:
            # 메시지 구성
            payload = self._build_payload(notification)

            # 모든 webhook에 전송
            success_count = 0
            for webhook_url in self.webhook_urls.values():
                try:
                    response = requests.post(
                        webhook_url, json=payload, timeout=self.config.timeout
                    )
                    if response.status_code == 200:
                        success_count += 1
                except Exception:
                    pass

            if success_count > 0:
                self.record_sent()
                return True
            else:
                self.record_failed()
                return False

        except Exception:
            self.record_failed()
            return False

    def _build_payload(self, notification: Any) -> Dict[str, Any]:
        """Slack 페이로드 구성"""
        # 우선순위별 색상
        priority_colors = {
            "critical": "#FF0000",
            "high": "#FF6B6B",
            "medium": "#FFA500",
            "low": "#FFFF00",
            "info": "#4CAF50",
        }

        priority_value = getattr(notification, "priority", None)
        if priority_value:
            priority_str = getattr(priority_value, "value", "info")
        else:
            priority_str = "info"

        color = priority_colors.get(priority_str, "#808080")

        return {
            "attachments": [
                {
                    "color": color,
                    "title": getattr(notification, "title", "Notification"),
                    "text": getattr(notification, "message", ""),
                    "fields": [
                        {"title": "Priority", "value": priority_str, "short": True},
                        {
                            "title": "Source",
                            "value": getattr(notification, "source", "Unknown"),
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": datetime.fromtimestamp(
                                getattr(notification, "timestamp", 0)
                            ).isoformat(),
                            "short": False,
                        },
                    ],
                }
            ]
        }


class WebhookChannel(NotificationChannel):
    """웹훅 채널"""

    def __init__(self, webhook_urls: List[str], config: Optional[ChannelConfig] = None):
        super().__init__(config or ChannelConfig())
        self.webhook_urls = webhook_urls

    def add_webhook(self, url: str):
        """웹훅 URL 추가"""
        if url not in self.webhook_urls:
            self.webhook_urls.append(url)

    def remove_webhook(self, url: str):
        """웹훅 URL 제거"""
        if url in self.webhook_urls:
            self.webhook_urls.remove(url)

    def send(self, notification: Any) -> bool:
        """웹훅으로 전송"""
        if not self.can_send():
            return False

        try:
            # 페이로드 구성
            payload = self._build_payload(notification)

            success_count = 0
            for webhook_url in self.webhook_urls:
                try:
                    response = requests.post(
                        webhook_url,
                        json=payload,
                        timeout=self.config.timeout,
                        headers={"Content-Type": "application/json"},
                    )
                    if response.status_code in [200, 201]:
                        success_count += 1
                except Exception:
                    pass

            if success_count > 0:
                self.record_sent()
                return True
            else:
                self.record_failed()
                return False

        except Exception:
            self.record_failed()
            return False

    def _build_payload(self, notification: Any) -> Dict[str, Any]:
        """웹훅 페이로드 구성"""
        payload = {
            "notification_id": getattr(notification, "notification_id", ""),
            "title": getattr(notification, "title", ""),
            "message": getattr(notification, "message", ""),
            "priority": getattr(
                getattr(notification, "priority", None), "value", "medium"
            ),
            "source": getattr(notification, "source", ""),
            "timestamp": getattr(notification, "timestamp", 0),
            "tags": getattr(notification, "tags", []),
            "metadata": getattr(notification, "metadata", {}),
        }

        return payload


class ChannelFactory:
    """채널 팩토리"""

    @staticmethod
    def create_email_channel(
        smtp_server: str,
        smtp_port: int = 587,
        username: str = "",
        password: str = "",
        from_address: str = "",
    ) -> EmailChannel:
        """이메일 채널 생성"""
        return EmailChannel(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            username=username,
            password=password,
            from_address=from_address,
        )

    @staticmethod
    def create_slack_channel(webhook_urls: Dict[str, str]) -> SlackChannel:
        """Slack 채널 생성"""
        return SlackChannel(webhook_urls)

    @staticmethod
    def create_webhook_channel(webhook_urls: List[str]) -> WebhookChannel:
        """웹훅 채널 생성"""
        return WebhookChannel(webhook_urls)
