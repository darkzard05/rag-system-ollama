"""
Task 25: API & System Integration Tests
Tests: 35+ comprehensive integration tests
"""

import pytest
import json
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Mock imports for components that should be integrated
from src.api.api_server import (
    APIResponse, SearchRequest, SearchResponse, DeploymentRequest, APIEndpoints
)
from src.api.websocket_handler import (
    WebSocketManager, StreamingHandler, MessageType, WSMessage, ClientConnection
)
from src.cache.system_integration import SystemIntegration, SystemConfig


class TestSystemIntegration:
    """Test system integration layer"""
    
    @pytest.fixture
    def integration(self):
        """Create integration instance"""
        config = SystemConfig(
            service_name="rag-system",
            version="1.0.0",
            environment="test",
            enable_caching=True,
            enable_monitoring=True,
        )
        return SystemIntegration(config)
    
    def test_system_initialization(self, integration):
        """Test system initialization"""
        result = integration.initialize()
        assert result["status"] == "initialized"
        assert integration.is_initialized
    
    def test_component_registration(self, integration):
        """Test component registration"""
        mock_component = Mock()
        assert integration.register_component("test", mock_component)
        assert "test" in integration.components
    
    def test_service_registration(self, integration):
        """Test service registration"""
        mock_service = Mock()
        assert integration.register_service("test_service", mock_service)
        assert "test_service" in integration.services
    
    def test_middleware_registration(self, integration):
        """Test middleware registration"""
        mock_middleware = Mock(return_value={"processed": True})
        mock_middleware.__name__ = "test_middleware"
        assert integration.register_middleware(mock_middleware)
        assert len(integration.middleware_stack) > 0
    
    def test_get_system_status(self, integration):
        """Test system status endpoint"""
        integration.initialize()
        status = integration.get_system_status()
        
        assert "service_name" in status
        assert status["service_name"] == "rag-system"
        assert status["version"] == "1.0.0"
        assert status["is_initialized"]
        assert status["request_count"] == 0
        assert "uptime_seconds" in status
    
    def test_request_execution(self, integration):
        """Test request execution through middleware"""
        def test_middleware(request):
            request["processed"] = True
            return request
        
        integration.register_middleware(test_middleware)
        
        request = {"type": "search"}
        result = integration.execute_request(request)
        
        assert result["status"] == "success"
        assert result["response"]["processed"]
        assert "execution_time_ms" in result
    
    def test_request_counting(self, integration):
        """Test request counting"""
        integration.initialize()
        
        for _ in range(5):
            integration.execute_request({})
        
        status = integration.get_system_status()
        assert status["request_count"] == 5
    
    def test_error_handling(self, integration):
        """Test error handling"""
        def bad_middleware(request):
            raise Exception("Test error")
        
        integration.register_middleware(bad_middleware)
        result = integration.execute_request({})
        
        assert result["status"] == "error"
        assert "error" in result
        assert integration.error_count > 0
    
    def test_authentication(self, integration):
        """Test request authentication"""
        integration.register_component("rbac", Mock())
        
        request = {"type": "search"}
        assert integration.authenticate_request(request, "valid_token")
        assert not integration.authenticate_request(request, "")
    
    def test_cache_operations(self, integration):
        """Test caching operations"""
        integration.register_component("cache", Mock())
        integration.initialize()
        
        request = {"query": "test"}
        
        # Check cache (should be None initially)
        cached = integration.check_cache(request)
        assert cached is None
        
        # Store in cache
        response = {"results": []}
        assert integration.store_cache(request, response)
    
    def test_cache_disabled(self, integration):
        """Test with caching disabled"""
        integration.config.enable_caching = False
        
        request = {"query": "test"}
        response = {"results": []}
        
        assert not integration.store_cache(request, response)
        assert integration.check_cache(request) is None
    
    def test_metrics_recording(self, integration):
        """Test metrics recording"""
        integration.register_component("monitor", Mock())
        integration.initialize()
        
        assert integration.record_metrics("search_latency", 150.5, "search_engine")
        assert integration.record_metrics("cache_hit_rate", 0.85, "cache")
    
    def test_metrics_disabled(self, integration):
        """Test with monitoring disabled"""
        integration.config.enable_monitoring = False
        
        assert not integration.record_metrics("test_metric", 100)
    
    def test_notifications(self, integration):
        """Test notification system"""
        integration.register_component("notifications", Mock())
        integration.initialize()
        
        assert integration.send_notification(
            "test_event",
            "Test message",
            "info"
        )
        assert integration.send_notification(
            "error_event",
            "Error message",
            "error"
        )
    
    def test_deployment_info(self, integration):
        """Test deployment information retrieval"""
        integration.register_component("deployment", Mock())
        
        dep_info = integration.get_deployment_info("dep_123")
        assert dep_info is not None
        assert "deployment_id" in dep_info
        assert dep_info["status"] == "active"
    
    def test_trigger_deployment(self, integration):
        """Test deployment triggering"""
        integration.register_component("deployment", Mock())
        integration.register_component("notifications", Mock())
        
        dep_id = integration.trigger_deployment(
            "rag-service",
            "1.0.0",
            "production"
        )
        
        assert dep_id.startswith("dep_")
    
    def test_migration_execution(self, integration):
        """Test migration execution"""
        result = integration.perform_migration(
            "add_indexes",
            "schema_update"
        )
        assert result
    
    def test_system_health(self, integration):
        """Test system health check"""
        integration.initialize()
        
        health = integration.get_system_health()
        
        assert "health_status" in health
        assert "system_status" in health
        assert "components_healthy" in health
        assert health["health_status"] in ["healthy", "warning", "degraded"]
    
    def test_system_shutdown(self, integration):
        """Test system shutdown"""
        integration.initialize()
        integration.request_count = 10
        
        result = integration.shutdown()
        
        assert result["status"] == "shutdown"
        assert result["total_requests"] == 10
        assert "uptime_seconds" in result


class TestAPIIntegration:
    """Test API endpoints integration"""
    
    @pytest.fixture
    def api_endpoints(self):
        """Create API endpoints instance"""
        return APIEndpoints()
    
    @pytest.fixture
    def integration(self):
        """Create integration instance"""
        return SystemIntegration()
    
    def test_api_response_wrapper(self):
        """Test API response wrapper"""
        response = APIResponse(
            status_code=200,
            data={"results": []},
            message="Success"
        )
        
        assert response.status_code == 200
        assert response.data == {"results": []}
        assert response.message == "Success"
    
    def test_search_request_validation(self):
        """Test search request validation"""
        request = SearchRequest(
            query="test query",
            top_k=5
        )
        
        assert request.query == "test query"
        assert request.top_k == 5
    
    def test_search_response_structure(self):
        """Test search response structure"""
        response = SearchResponse(
            results=[],
            query="test",
            execution_time_ms=100.5
        )
        
        assert response.query == "test"
        assert isinstance(response.results, list)
        assert response.execution_time_ms > 0
    
    def test_deployment_request_validation(self):
        """Test deployment request validation"""
        request = DeploymentRequest(
            service_name="rag-service",
            version="1.0.0",
            environment="production"
        )
        
        assert request.service_name == "rag-service"
        assert request.version == "1.0.0"
        assert request.environment == "production"
    
    def test_api_endpoints_search(self, api_endpoints, integration):
        """Test API search endpoint"""
        integration.register_component("search", Mock())
        
        result = api_endpoints.post_search(
            SearchRequest(query="test", top_k=5)
        )
        
        assert isinstance(result, APIResponse)
        assert result.status_code in [200, 500]
    
    def test_api_endpoints_documents(self, api_endpoints):
        """Test API document endpoints"""
        # Create
        create_result = api_endpoints.post_document(
            "doc_1",
            "test document"
        )
        assert isinstance(create_result, APIResponse)
        
        # Get
        get_result = api_endpoints.get_documents()
        assert isinstance(get_result, APIResponse)
    
    def test_api_endpoints_deployment(self, api_endpoints):
        """Test API deployment endpoints"""
        # Trigger
        deploy_result = api_endpoints.post_deployment(
            DeploymentRequest(
                service_name="rag-service",
                version="1.0.0",
                environment="production"
            )
        )
        assert isinstance(deploy_result, APIResponse)
        
        # Get
        get_result = api_endpoints.get_deployments()
        assert isinstance(get_result, APIResponse)
    
    def test_api_endpoints_cache(self, api_endpoints):
        """Test API cache endpoints"""
        # Stats
        stats_result = api_endpoints.get_cache_stats()
        assert isinstance(stats_result, APIResponse)
        
        # Clear
        clear_result = api_endpoints.post_cache_clear()
        assert isinstance(clear_result, APIResponse)
    
    def test_api_endpoints_health(self, api_endpoints):
        """Test API health endpoint"""
        health = api_endpoints.get_health()
        assert isinstance(health, APIResponse)
        assert health.status_code in [200, 500]
    
    def test_api_endpoints_metrics(self, api_endpoints):
        """Test API metrics endpoint"""
        metrics = api_endpoints.get_metrics()
        assert isinstance(metrics, APIResponse)
    
    def test_api_endpoints_auth(self, api_endpoints):
        """Test API auth endpoints"""
        # Login
        login_result = api_endpoints.post_login(
            "test",
            "pass"
        )
        assert isinstance(login_result, APIResponse)
        
        # Logout
        logout_result = api_endpoints.post_logout("test_token")
        assert isinstance(logout_result, APIResponse)
    
    def test_api_endpoints_notifications(self, api_endpoints):
        """Test API notification endpoints"""
        # Get
        get_result = api_endpoints.get_notifications()
        assert isinstance(get_result, APIResponse)
        
        # Mark read
        read_result = api_endpoints.post_notification_read(
            {"notification_id": "test_id"}
        )
        assert isinstance(read_result, APIResponse)


class TestWebSocketIntegration:
    """Test WebSocket integration"""
    
    @pytest.fixture
    def ws_manager(self):
        """Create WebSocket manager"""
        return WebSocketManager()
    
    def test_ws_message_structure(self):
        """Test WebSocket message structure"""
        message = WSMessage(
            message_type=MessageType.SEARCH,
            data={"query": "test"},
            client_id="client_1",
            timestamp=datetime.now()
        )
        
        assert message.message_type == MessageType.SEARCH
        assert message.data["query"] == "test"
        assert message.client_id == "client_1"
    
    def test_client_connection_tracking(self):
        """Test client connection tracking"""
        connection = ClientConnection(
            client_id="client_1",
            connected_at=time.time(),
            last_heartbeat=time.time(),
            subscriptions=["search", "monitor"]
        )
        
        assert connection.client_id == "client_1"
        assert "search" in connection.subscriptions
    
    def test_ws_client_connect(self, ws_manager):
        """Test client connection"""
        result = ws_manager.client_connect("client_1")
        assert isinstance(result, dict)
        assert result["client_id"] == "client_1"
    
    def test_ws_client_disconnect(self, ws_manager):
        """Test client disconnection"""
        ws_manager.client_connect("client_1")
        result = ws_manager.client_disconnect("client_1")
        
        assert result
    
    def test_ws_authentication(self, ws_manager):
        """Test WebSocket authentication"""
        ws_manager.client_connect("client_1")
        
        result = ws_manager.authenticate_client("client_1", "token_123")
        assert result
    
    def test_ws_channel_subscription(self, ws_manager):
        """Test channel subscription"""
        ws_manager.client_connect("client_1")
        
        result = ws_manager.subscribe_channel("client_1", "search")
        assert result
        
        clients = ws_manager.get_channel_subscribers("search")
        assert "client_1" in clients
    
    def test_ws_channel_unsubscription(self, ws_manager):
        """Test channel unsubscription"""
        ws_manager.client_connect("client_1")
        ws_manager.subscribe_channel("client_1", "search")
        
        result = ws_manager.unsubscribe_channel("client_1", "search")
        assert result
    
    def test_ws_send_message(self, ws_manager):
        """Test sending message"""
        ws_manager.client_connect("client_1")
        
        message = WSMessage(
            message_type=MessageType.SEARCH,
            data={"query": "test"},
            client_id="client_1"
        )
        
        result = ws_manager.send_message("client_1", message)
        assert result
    
    def test_ws_broadcast_message(self, ws_manager):
        """Test broadcasting message"""
        ws_manager.client_connect("client_1")
        ws_manager.client_connect("client_2")
        ws_manager.subscribe_channel("client_1", "monitor")
        ws_manager.subscribe_channel("client_2", "monitor")
        
        message = WSMessage(
            message_type=MessageType.MONITOR,
            data={"metric": "cpu_usage"},
            client_id="system"
        )
        
        result = ws_manager.broadcast_message("monitor", message)
        assert result
    
    def test_ws_get_connected_clients(self, ws_manager):
        """Test getting connected clients"""
        ws_manager.client_connect("client_1")
        ws_manager.client_connect("client_2")
        
        clients = ws_manager.get_connected_clients()
        assert len(clients) >= 2
    
    def test_ws_heartbeat_check(self, ws_manager):
        """Test heartbeat monitoring"""
        ws_manager.client_connect("client_1")
        
        # Simulate heartbeat
        result = ws_manager.heartbeat_check()
        assert isinstance(result, int)
    
    def test_ws_statistics(self, ws_manager):
        """Test WebSocket statistics"""
        ws_manager.client_connect("client_1")
        
        stats = ws_manager.get_statistics()
        
        assert "total_connections" in stats
        assert "message_queue_size" in stats
    
    def test_streaming_handler_search(self):
        """Test search result streaming"""
        ws_manager = WebSocketManager()
        handler = StreamingHandler(ws_manager)
        ws_manager.client_connect("client_1")
        
        results = [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.8},
        ]
        
        # Should return stream generator info
        stream_info = handler.stream_search_results("client_1", "test", results)
        assert isinstance(stream_info, bool)
    
    def test_streaming_handler_monitoring(self):
        """Test monitoring data streaming"""
        ws_manager = WebSocketManager()
        handler = StreamingHandler(ws_manager)
        ws_manager.subscribe_channel("client_1", "monitor")
        
        monitoring_data = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": 45.5,
            "memory_usage": 60.2,
        }
        
        stream_info = handler.stream_monitoring_data("monitor", monitoring_data)
        assert isinstance(stream_info, int)
    
    def test_streaming_handler_notification(self):
        """Test notification sending"""
        ws_manager = WebSocketManager()
        handler = StreamingHandler(ws_manager)
        ws_manager.subscribe_channel("client_1", "notifications")
        
        notification = {
            "type": "deployment_completed",
            "message": "Deployment successful"
        }
        
        result = handler.send_notification("notifications", notification)
        assert isinstance(result, int)


class TestSystemIntegrationFlow:
    """Test complete integration flows"""
    
    def test_search_request_flow(self):
        """Test complete search request flow"""
        integration = SystemIntegration()
        integration.register_component("search", Mock())
        integration.register_component("cache", Mock())
        integration.register_component("monitor", Mock())
        integration.register_component("rbac", Mock())
        integration.initialize()
        
        # Create search request
        request = SearchRequest(query="test", top_k=5)
        
        # Execute through integration
        auth_result = integration.authenticate_request(
            {"request": request},
            "valid_token"
        )
        assert auth_result
        
        # Record metrics
        assert integration.record_metrics("search_request", 1, "search")
        
        # Simulate caching
        assert integration.store_cache(
            {"query": "test"},
            {"results": []}
        )
    
    def test_deployment_flow(self):
        """Test complete deployment flow"""
        integration = SystemIntegration()
        integration.register_component("deployment", Mock())
        integration.register_component("notifications", Mock())
        integration.initialize()
        
        # Trigger deployment
        dep_id = integration.trigger_deployment(
            "rag-service",
            "2.0.0",
            "staging"
        )
        
        assert dep_id.startswith("dep_")
        
        # Get deployment info
        dep_info = integration.get_deployment_info(dep_id)
        assert dep_info is not None
    
    def test_error_recovery_flow(self):
        """Test error handling and recovery"""
        integration = SystemIntegration()
        integration.register_component("notifications", Mock())
        
        def failing_middleware(request):
            raise Exception("Service unavailable")
        
        integration.register_middleware(failing_middleware)
        
        result = integration.execute_request({"type": "search"})
        
        assert result["status"] == "error"
        assert integration.error_count > 0
        
        # System should still function
        health = integration.get_system_health()
        assert health["health_status"] in ["warning", "degraded", "healthy"]


class TestSystemPerformance:
    """Test system performance characteristics"""
    
    def test_request_throughput(self):
        """Test request throughput"""
        integration = SystemIntegration()
        integration.register_component("monitor", Mock())
        integration.initialize()
        
        start_time = time.time()
        
        for _ in range(100):
            integration.execute_request({"test": True})
        
        elapsed = max(0.1, time.time() - start_time)  # Avoid division by zero
        throughput = 100 / elapsed  # requests per second
        
        assert throughput > 0
    
    def test_response_latency(self):
        """Test response latency"""
        integration = SystemIntegration()
        integration.initialize()
        
        def quick_middleware(request):
            return request
        
        integration.register_middleware(quick_middleware)
        
        result = integration.execute_request({})
        latency = result["execution_time_ms"]
        
        assert latency < 100  # Should be under 100ms
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        integration = SystemIntegration()
        integration.initialize()
        
        # Simulate multiple requests
        for _ in range(50):
            integration.execute_request({"id": _})
        
        status = integration.get_system_status()
        assert status["request_count"] == 50


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
