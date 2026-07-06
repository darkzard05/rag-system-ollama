import time
import pytest
from unittest.mock import MagicMock
from src.api.websocket_handler import (
    WebSocketManager,
    StreamingHandler,
    MessageType,
    WSMessage,
    ClientConnection,
)

@pytest.fixture
def ws_manager():
    return WebSocketManager()

@pytest.fixture
def streaming_handler(ws_manager):
    return StreamingHandler(ws_manager)

class TestWebSocketManager:
    def test_client_lifecycle(self, ws_manager):
        """Test case: Client connect, authenticate, and disconnect lifecycle."""
        client_id = "test_client_1"
        user_id = "user_123"

        # Connect
        conn_info = ws_manager.client_connect(client_id)
        assert conn_info["client_id"] == client_id
        assert client_id in ws_manager.connections
        assert ws_manager.connections[client_id].authenticated is False

        # Authenticate
        auth_success = ws_manager.authenticate_client(client_id, user_id)
        assert auth_success is True
        assert ws_manager.connections[client_id].authenticated is True
        assert ws_manager.connections[client_id].user_id == user_id

        # Disconnect
        disconnect_success = ws_manager.client_disconnect(client_id)
        assert disconnect_success is True
        assert client_id not in ws_manager.connections

    def test_channel_subscription_and_broadcast(self, ws_manager):
        """Test case: Channel subscription and broadcast (verify only subscribers receive the message)."""
        client1 = "client_1"
        client2 = "client_2"
        client3 = "client_3"
        channel = "news"

        ws_manager.client_connect(client1)
        ws_manager.client_connect(client2)
        ws_manager.client_connect(client3)

        # Subscribe client1 and client2 to 'news'
        ws_manager.subscribe_channel(client1, channel)
        ws_manager.subscribe_channel(client2, channel)

        # Verify subscriptions
        assert client1 in ws_manager.get_channel_subscribers(channel)
        assert client2 in ws_manager.get_channel_subscribers(channel)
        assert client3 not in ws_manager.get_channel_subscribers(channel)

        # Broadcast message
        msg = WSMessage(message_type=MessageType.NOTIFICATION, data={"text": "hello"})
        count = ws_manager.broadcast_message(channel, msg)

        assert count == 2
        # Check if messages are in queue for subscribers
        # Note: send_message appends to message_queue
        # Since we are testing the manager, we check the queue size or content
        # In this implementation, every send_message appends to the same queue.
        # We expect 2 messages for the broadcast + 3 welcome messages from connects = 5
        assert len(ws_manager.message_queue) == 5

    def test_custom_message_handler(self, ws_manager):
        """Test case: Custom message handler registration and execution via handle_message."""
        client_id = "client_handler"
        ws_manager.client_connect(client_id)

        # Define a handler
        mock_handler = MagicMock(return_value={"status": "handled"})
        ws_manager.register_handler(MessageType.SEARCH, mock_handler)

        # Simulate incoming message
        message_data = {
            "type": "search",
            "data": {"query": "test query"}
        }

        response = ws_manager.handle_message(client_id, message_data)

        assert response == {"status": "handled"}
        mock_handler.assert_called_once_with(client_id, {"query": "test query"})

    def test_heartbeat_timeout(self, ws_manager):
        """Test case: Heartbeat timeout (heartbeat_check) removes inactive clients."""
        client_active = "active_client"
        client_inactive = "inactive_client"

        ws_manager.client_connect(client_active)
        ws_manager.client_connect(client_inactive)

        # Manually manipulate last_heartbeat for the inactive client
        # Set it to 10 minutes ago
        ws_manager.connections[client_inactive].last_heartbeat = time.time() - 600

        # Check heartbeat with 300s timeout
        disconnected_count = ws_manager.heartbeat_check(timeout_seconds=300)

        assert disconnected_count == 1
        assert client_inactive not in ws_manager.connections
        assert client_active in ws_manager.connections

    def test_broadcast_message_count(self, ws_manager):
        """Test case: broadcast_message returns the correct count of reached clients."""
        client1 = "c1"
        client2 = "c2"
        channel = "test_channel"

        ws_manager.client_connect(client1)
        ws_manager.client_connect(client2)
        ws_manager.subscribe_channel(client1, channel)
        ws_manager.subscribe_channel(client2, channel)

        msg = WSMessage(message_type=MessageType.NOTIFICATION, data={"msg": "test"})
        count = ws_manager.broadcast_message(channel, msg)
        assert count == 2

        # Test non-existent channel
        count_empty = ws_manager.broadcast_message("non_existent", msg)
        assert count_empty == 0

        # Test channel with no subscribers (but exists in broadcast_channels)
        ws_manager.broadcast_channels["empty_channel"] = []
        count_empty_sub = ws_manager.broadcast_message("empty_channel", msg)
        assert count_empty_sub == 0

class TestStreamingHandler:
    def test_stream_search_results(self, ws_manager, streaming_handler):
        """Test case: StreamingHandler.stream_search_results produces correct WSMessage sequences."""
        client_id = "stream_client"
        ws_manager.client_connect(client_id)

        query = "test query"
        results = [{"id": 1, "text": "res1"}, {"id": 2, "text": "res2"}]

        success = streaming_handler.stream_search_results(client_id, query, results)
        assert success is True

        # Check message queue for the streamed messages
        # 1 welcome message + 2 search messages = 3
        assert len(ws_manager.message_queue) == 3

        # Verify the content of the last message
        last_msg = ws_manager.message_queue[-1]
        assert last_msg.message_type == MessageType.SEARCH
        assert last_msg.data["query"] == query
        assert last_msg.data["index"] == 1
        assert last_msg.data["total"] == 2
        assert last_msg.data["result"] == {"id": 2, "text": "res2"}

    def test_stream_monitoring_data(self, ws_manager, streaming_handler):
        """Test monitoring data streaming."""
        channel = "monitor_channel"
        client1 = "c1"
        client2 = "c2"
        ws_manager.client_connect(client1)
        ws_manager.client_connect(client2)
        ws_manager.subscribe_channel(client1, channel)
        ws_manager.subscribe_channel(client2, channel)

        monitoring_data = {"cpu": 50, "mem": 40}
        count = streaming_handler.stream_monitoring_data(channel, monitoring_data)

        assert count == 2
        last_msg = ws_manager.message_queue[-1]
        assert last_msg.message_type == MessageType.MONITOR
        assert last_msg.data == monitoring_data

    def test_send_notification(self, ws_manager, streaming_handler):
        """Test notification sending."""
        channel = "notify_channel"
        client1 = "c1"
        ws_manager.client_connect(client1)
        ws_manager.subscribe_channel(client1, channel)

        notification = {"alert": "system reboot"}
        count = streaming_handler.send_notification(channel, notification)

        assert count == 1
        last_msg = ws_manager.message_queue[-1]
        assert last_msg.message_type == MessageType.NOTIFICATION
        assert last_msg.data == notification
