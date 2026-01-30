"""
WebSocket Handler for Real-time Communication
"""

import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import RLock
from typing import Any


class MessageType(Enum):
    """WebSocket Message Types"""

    CONNECT = "connect"
    DISCONNECT = "disconnect"
    SEARCH = "search"
    MONITOR = "monitor"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class WSMessage:
    """WebSocket Message"""

    message_type: MessageType
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    client_id: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "type": self.message_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "client_id": self.client_id,
        }


@dataclass
class ClientConnection:
    """WebSocket Client Connection"""

    client_id: str
    connected_at: float
    last_heartbeat: float
    subscriptions: list[str] = field(default_factory=list)
    authenticated: bool = False
    user_id: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


class WebSocketManager:
    """Manages WebSocket connections and messaging"""

    def __init__(self, max_message_queue: int = 1000):
        """
        Initialize WebSocket manager

        Args:
            max_message_queue: Maximum message queue size
        """
        self.connections: dict[str, ClientConnection] = {}
        self.message_queue: deque = deque(maxlen=max_message_queue)
        self.message_handlers: dict[MessageType, callable] = {}
        self.broadcast_channels: dict[str, list[str]] = {}  # channel -> client_ids

        self._lock = RLock()
        self.logger = logging.getLogger(__name__)

    def register_handler(self, message_type: MessageType, handler: callable):
        """Register message handler"""
        with self._lock:
            self.message_handlers[message_type] = handler
            self.logger.info(f"Registered handler for {message_type.value}")

    def client_connect(self, client_id: str) -> dict[str, Any]:
        """
        Handle client connection

        Args:
            client_id: Client ID

        Returns:
            Connection info
        """
        with self._lock:
            connection = ClientConnection(
                client_id=client_id,
                connected_at=time.time(),
                last_heartbeat=time.time(),
            )

            self.connections[client_id] = connection

            # Send welcome message
            welcome_msg = WSMessage(
                message_type=MessageType.CONNECT,
                data={"message": "Connected to RAG system", "client_id": client_id},
                client_id=client_id,
            )

            self.message_queue.append(welcome_msg)

            self.logger.info(f"Client connected: {client_id}")

            return connection.to_dict()

    def client_disconnect(self, client_id: str) -> bool:
        """
        Handle client disconnection

        Args:
            client_id: Client ID

        Returns:
            Success status
        """
        with self._lock:
            if client_id in self.connections:
                del self.connections[client_id]

                # Remove from channels
                for channel_clients in self.broadcast_channels.values():
                    if client_id in channel_clients:
                        channel_clients.remove(client_id)

                self.logger.info(f"Client disconnected: {client_id}")
                return True

            return False

    def authenticate_client(self, client_id: str, user_id: str) -> bool:
        """
        Authenticate client

        Args:
            client_id: Client ID
            user_id: User ID

        Returns:
            Success status
        """
        with self._lock:
            if client_id in self.connections:
                self.connections[client_id].authenticated = True
                self.connections[client_id].user_id = user_id

                self.logger.info(f"Client authenticated: {client_id} -> {user_id}")
                return True

            return False

    def subscribe_channel(self, client_id: str, channel: str) -> bool:
        """
        Subscribe client to channel

        Args:
            client_id: Client ID
            channel: Channel name

        Returns:
            Success status
        """
        with self._lock:
            if client_id not in self.connections:
                return False

            # Add to subscriptions
            if channel not in self.connections[client_id].subscriptions:
                self.connections[client_id].subscriptions.append(channel)

            # Add to broadcast channel
            if channel not in self.broadcast_channels:
                self.broadcast_channels[channel] = []

            if client_id not in self.broadcast_channels[channel]:
                self.broadcast_channels[channel].append(client_id)

            self.logger.info(f"Client subscribed: {client_id} -> {channel}")

            return True

    def unsubscribe_channel(self, client_id: str, channel: str) -> bool:
        """
        Unsubscribe client from channel

        Args:
            client_id: Client ID
            channel: Channel name

        Returns:
            Success status
        """
        with self._lock:
            if client_id in self.connections:
                if channel in self.connections[client_id].subscriptions:
                    self.connections[client_id].subscriptions.remove(channel)

            if channel in self.broadcast_channels:
                if client_id in self.broadcast_channels[channel]:
                    self.broadcast_channels[channel].remove(client_id)

            return True

    def send_message(self, client_id: str, message: WSMessage) -> bool:
        """
        Send message to client

        Args:
            client_id: Client ID
            message: Message to send

        Returns:
            Success status
        """
        with self._lock:
            if client_id not in self.connections:
                return False

            message.client_id = client_id
            self.message_queue.append(message)

            return True

    def broadcast_message(self, channel: str, message: WSMessage) -> int:
        """
        Broadcast message to channel subscribers

        Args:
            channel: Channel name
            message: Message to broadcast

        Returns:
            Number of clients that received message
        """
        with self._lock:
            if channel not in self.broadcast_channels:
                return 0

            client_ids = self.broadcast_channels[channel]
            count = 0

            for client_id in client_ids:
                if self.send_message(client_id, message):
                    count += 1

            self.logger.info(
                f"Broadcasted message to {count} clients on channel {channel}"
            )

            return count

    def handle_message(
        self, client_id: str, message_data: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Handle incoming message

        Args:
            client_id: Client ID
            message_data: Message data

        Returns:
            Response
        """
        with self._lock:
            if client_id not in self.connections:
                return {"error": "Client not found"}

            try:
                message_type = MessageType(message_data.get("type", "error"))
                data = message_data.get("data", {})

                # Update last heartbeat
                self.connections[client_id].last_heartbeat = time.time()

                # Call handler if registered
                if message_type in self.message_handlers:
                    handler = self.message_handlers[message_type]
                    response = handler(client_id, data)
                    return response

                return {"status": "ok", "message": "Message processed"}

            except ValueError:
                return {"error": "Invalid message type"}
            except Exception as e:
                self.logger.error(f"Error handling message: {str(e)}")
                return {"error": str(e)}

    def get_connected_clients(self) -> list[dict[str, Any]]:
        """Get list of connected clients"""
        with self._lock:
            return [c.to_dict() for c in self.connections.values()]

    def get_channel_subscribers(self, channel: str) -> list[str]:
        """Get subscribers of channel"""
        with self._lock:
            return self.broadcast_channels.get(channel, [])

    def get_client_info(self, client_id: str) -> dict[str, Any] | None:
        """Get client information"""
        with self._lock:
            if client_id in self.connections:
                return self.connections[client_id].to_dict()
            return None

    def heartbeat_check(self, timeout_seconds: int = 300) -> int:
        """
        Check for disconnected clients (heartbeat timeout)

        Args:
            timeout_seconds: Heartbeat timeout

        Returns:
            Number of disconnected clients
        """
        with self._lock:
            current_time = time.time()
            disconnected = []

            for client_id, connection in self.connections.items():
                if (current_time - connection.last_heartbeat) > timeout_seconds:
                    disconnected.append(client_id)

            # Remove disconnected clients
            for client_id in disconnected:
                self.client_disconnect(client_id)

            if disconnected:
                self.logger.info(f"Disconnected {len(disconnected)} inactive clients")

            return len(disconnected)

    def get_statistics(self) -> dict[str, Any]:
        """Get WebSocket statistics"""
        with self._lock:
            authenticated_count = sum(
                1 for c in self.connections.values() if c.authenticated
            )

            return {
                "total_connections": len(self.connections),
                "authenticated_clients": authenticated_count,
                "total_channels": len(self.broadcast_channels),
                "message_queue_size": len(self.message_queue),
                "channels": {
                    channel: len(clients)
                    for channel, clients in self.broadcast_channels.items()
                },
            }


class StreamingHandler:
    """Handles streaming responses"""

    def __init__(self, ws_manager: WebSocketManager):
        """
        Initialize streaming handler

        Args:
            ws_manager: WebSocket manager instance
        """
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)

    def stream_search_results(
        self, client_id: str, query: str, results: list[dict]
    ) -> bool:
        """
        Stream search results to client

        Args:
            client_id: Client ID
            query: Search query
            results: Search results

        Returns:
            Success status
        """
        try:
            for i, result in enumerate(results):
                message = WSMessage(
                    message_type=MessageType.SEARCH,
                    data={
                        "query": query,
                        "result": result,
                        "index": i,
                        "total": len(results),
                    },
                    client_id=client_id,
                )

                self.ws_manager.send_message(client_id, message)

            return True

        except Exception as e:
            self.logger.error(f"Error streaming results: {str(e)}")
            return False

    def stream_monitoring_data(
        self, channel: str, monitoring_data: dict[str, Any]
    ) -> int:
        """
        Stream monitoring data to subscribers

        Args:
            channel: Monitoring channel
            monitoring_data: Monitoring data

        Returns:
            Number of clients that received data
        """
        try:
            message = WSMessage(
                message_type=MessageType.MONITOR,
                data=monitoring_data,
            )

            return self.ws_manager.broadcast_message(channel, message)

        except Exception as e:
            self.logger.error(f"Error streaming monitoring data: {str(e)}")
            return 0

    def send_notification(self, channel: str, notification: dict[str, Any]) -> int:
        """
        Send notification to channel

        Args:
            channel: Notification channel
            notification: Notification data

        Returns:
            Number of clients that received notification
        """
        try:
            message = WSMessage(
                message_type=MessageType.NOTIFICATION,
                data=notification,
            )

            return self.ws_manager.broadcast_message(channel, message)

        except Exception as e:
            self.logger.error(f"Error sending notification: {str(e)}")
            return 0
