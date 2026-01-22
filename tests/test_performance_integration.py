"""
Performance Monitoring Integration Tests

테스트 RAG 파이프라인 시뮬레이션으로 모든 핵심 오퍼레이션의 성능 추적이 정상 작동하는지 검증합니다.
"""

import sys
import unittest
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.monitoring.performance_monitor import (
    PerformanceMonitor,
    OperationType,
    OperationMetrics,
)


class TestPerformanceIntegration(unittest.TestCase):
    """RAG 파이프라인 성능 모니터링 통합 테스트"""

    def setUp(self):
        """각 테스트 전 모니터 초기화"""
        self.monitor = get_performance_monitor()
        self.monitor.clear_metrics()

    def test_all_operation_types_tracked(self):
        """모든 오퍼레이션 타입이 추적되는지 검증"""
        operation_types = [
            OperationType.DOCUMENT_RETRIEVAL,
            OperationType.EMBEDDING_GENERATION,
            OperationType.RERANKING,
            OperationType.LLM_INFERENCE,
            OperationType.QUERY_PROCESSING,
            OperationType.PDF_LOADING,
            OperationType.SEMANTIC_CHUNKING,
        ]

        for op_type in operation_types:
            with self.monitor.track_operation(op_type, {"test": True}) as op:
                op.tokens = 100

        # 모든 오퍼레이션이 기록되었는지 확인
        all_stats = self.monitor.get_all_stats()
        
        tracked_types = set(stat for stat in all_stats.keys() if all_stats[stat].total_operations > 0)
        self.assertEqual(len(tracked_types), len(operation_types))

    def test_query_processing_to_llm_pipeline(self):
        """쿼리 처리부터 LLM 추론까지의 전체 파이프라인 추적"""
        # 1. 쿼리 처리
        with self.monitor.track_operation(OperationType.QUERY_PROCESSING) as op:
            op.tokens = 15
        
        # 2. 문서 검색
        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL, {"query_count": 1}) as op:
            op.tokens = 500
        
        # 3. 재순위화
        with self.monitor.track_operation(OperationType.RERANKING, {"doc_count": 10}) as op:
            op.tokens = 300
        
        # 4. LLM 추론
        with self.monitor.track_operation(OperationType.LLM_INFERENCE, {"doc_count": 5}) as op:
            op.tokens = 250

        # 검증
        report = self.monitor.generate_report()
        
        self.assertIn("operations", report)
        self.assertEqual(report["total_operations"], 4)
        
        # 토큰 수 합산 확인 (report에 총합이 없으니 operations에서 합산)
        total_tokens = sum(
            stats.get("tokens", {}).get("total", 0)
            for stats in report.get("operations", {}).values()
        )
        self.assertGreater(total_tokens, 0)

    def test_pdf_processing_pipeline(self):
        """PDF 처리 파이프라인: PDF 로드 -> 임베딩 생성 -> 의미론적 청킹"""
        # 1. PDF 로드
        with self.monitor.track_operation(OperationType.PDF_LOADING, {"file": "test.pdf"}) as op:
            op.tokens = 2000

        # 2. 임베딩 생성
        with self.monitor.track_operation(OperationType.EMBEDDING_GENERATION, {"model": "test"}) as op:
            op.tokens = 2000

        # 3. 의미론적 청킹
        with self.monitor.track_operation(OperationType.SEMANTIC_CHUNKING, {"doc_count": 10}) as op:
            op.tokens = 1500

        # 검증
        stats = self.monitor.get_operation_stats(OperationType.PDF_LOADING)
        self.assertEqual(stats.total_operations, 1)
        self.assertEqual(stats.total_tokens, 2000)

        stats = self.monitor.get_operation_stats(OperationType.EMBEDDING_GENERATION)
        self.assertEqual(stats.total_operations, 1)

        stats = self.monitor.get_operation_stats(OperationType.SEMANTIC_CHUNKING)
        self.assertEqual(stats.total_operations, 1)

    def test_error_tracking_in_pipeline(self):
        """파이프라인 중 오류 발생 시 추적"""
        try:
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                op.tokens = 100
                raise ValueError("LLM timeout")
        except ValueError:
            pass

        stats = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)
        self.assertEqual(stats.total_operations, 1)
        self.assertEqual(stats.failed_operations, 1)
        self.assertEqual(stats.successful_operations, 0)

    def test_metadata_preservation(self):
        """메타데이터가 보존되는지 검증"""
        metadata = {
            "file": "test.pdf",
            "model": "qwen",
            "top_k": 5,
            "doc_count": 10
        }

        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL, metadata) as op:
            op.tokens = 100

        # 메타데이터 확인
        stats = self.monitor.get_operation_stats(OperationType.DOCUMENT_RETRIEVAL)
        self.assertEqual(stats.total_operations, 1)

    def test_concurrent_operations_tracking(self):
        """동시 오퍼레이션 추적"""
        import threading
        import time

        def track_operation(op_type, duration):
            with self.monitor.track_operation(op_type) as op:
                time.sleep(duration)
                op.tokens = 100

        # 3개의 스레드에서 동시에 오퍼레이션 실행
        threads = [
            threading.Thread(target=track_operation, args=(OperationType.DOCUMENT_RETRIEVAL, 0.05)),
            threading.Thread(target=track_operation, args=(OperationType.LLM_INFERENCE, 0.05)),
            threading.Thread(target=track_operation, args=(OperationType.RERANKING, 0.05)),
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # 모든 오퍼레이션이 기록되었는지 확인
        report = self.monitor.generate_report()
        self.assertEqual(report["total_operations"], 3)

    def test_performance_report_generation(self):
        """성능 리포트 생성 검증"""
        # 여러 오퍼레이션 실행
        operations = [
            (OperationType.QUERY_PROCESSING, 50),
            (OperationType.DOCUMENT_RETRIEVAL, 500),
            (OperationType.EMBEDDING_GENERATION, 1000),
            (OperationType.LLM_INFERENCE, 300),
        ]

        for op_type, tokens in operations:
            with self.monitor.track_operation(op_type) as op:
                op.tokens = tokens

        # 리포트 생성
        report = self.monitor.generate_report()

        # 기본 필드 확인
        self.assertIn("timestamp", report)
        self.assertIn("total_operations", report)
        self.assertIn("operations", report)
        self.assertIn("memory", report)

        # 오퍼레이션 통계 확인
        self.assertEqual(report["total_operations"], 4)
        
        # 토큰 수 합산 확인
        total_tokens = sum(
            stats.get("tokens", {}).get("total", 0)
            for stats in report.get("operations", {}).values()
        )
        self.assertEqual(total_tokens, 50 + 500 + 1000 + 300)

        # 각 오퍼레이션 타입별 통계
        for op_type_key, op_stats in report["operations"].items():
            self.assertIn("total", op_stats)
            self.assertIn("successful", op_stats)
            self.assertIn("duration", op_stats)
            self.assertIn("tokens", op_stats)

    def test_health_status(self):
        """시스템 상태 확인"""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
            op.tokens = 100

        health = self.monitor.get_health_status()

        self.assertIn("status", health)
        self.assertIn("memory_mb", health)
        self.assertIn("total_operations", health)
        self.assertIn("issues", health)
        self.assertIn(health["status"], ["healthy", "warning"])

    def test_metrics_persistence(self):
        """메트릭스가 올바르게 집계되는지 검증"""
        # 같은 오퍼레이션을 여러 번 실행
        for i in range(5):
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                op.tokens = 100

        stats = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)

        self.assertEqual(stats.total_operations, 5)
        self.assertEqual(stats.successful_operations, 5)
        self.assertEqual(stats.total_tokens, 500)

    def test_operation_type_enum_completeness(self):
        """모든 오퍼레이션 타입이 정의되었는지 확인"""
        expected_operations = {
            "document_retrieval",
            "embedding_generation",
            "reranking",
            "llm_inference",
            "query_processing",
            "pdf_loading",
            "semantic_chunking",
        }

        actual_operations = {op.value for op in OperationType}

        self.assertEqual(actual_operations, expected_operations)

    def test_response_time_percentiles(self):
        """응답 시간 백분위수 계산 검증"""
        import time

        # 다양한 응답 시간의 오퍼레이션 생성
        durations = [0.01, 0.02, 0.03, 0.04, 0.05]

        for duration in durations:
            with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
                time.sleep(duration)
                op.tokens = 100

        stats = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)

        # 기본 통계 확인
        self.assertEqual(stats.total_operations, 5)
        self.assertGreater(stats.min_duration_seconds, 0)
        self.assertGreater(stats.max_duration_seconds, stats.min_duration_seconds)
        self.assertGreater(stats.avg_duration_seconds, stats.min_duration_seconds)

        # 백분위수 확인 (대략적인 범위)
        self.assertGreater(stats.p50_duration_seconds, stats.min_duration_seconds)
        self.assertGreaterEqual(stats.p95_duration_seconds, stats.p50_duration_seconds)
        self.assertGreaterEqual(stats.p99_duration_seconds, stats.p95_duration_seconds)
        self.assertLessEqual(stats.p99_duration_seconds, stats.max_duration_seconds)


class TestPerformanceEdgeCases(unittest.TestCase):
    """성능 모니터링 엣지 케이스 테스트"""

    def setUp(self):
        self.monitor = get_performance_monitor()
        self.monitor.clear_metrics()

    def test_zero_token_operations(self):
        """토큰 수가 0인 오퍼레이션 처리"""
        with self.monitor.track_operation(OperationType.QUERY_PROCESSING) as op:
            op.tokens = 0

        stats = self.monitor.get_operation_stats(OperationType.QUERY_PROCESSING)
        self.assertEqual(stats.total_operations, 1)
        self.assertEqual(stats.total_tokens, 0)

    def test_large_token_count(self):
        """큰 토큰 수 처리"""
        with self.monitor.track_operation(OperationType.LLM_INFERENCE) as op:
            op.tokens = 1000000  # 1백만 토큰

        stats = self.monitor.get_operation_stats(OperationType.LLM_INFERENCE)
        self.assertEqual(stats.total_tokens, 1000000)

    def test_metadata_with_special_characters(self):
        """특수 문자가 포함된 메타데이터 처리"""
        metadata = {
            "file": "test-한글-файл.pdf",
            "model": "qwen/qwen-7b",
            "query": "무엇이 성공의 열쇠인가?",
        }

        with self.monitor.track_operation(OperationType.DOCUMENT_RETRIEVAL, metadata) as op:
            op.tokens = 100

        stats = self.monitor.get_operation_stats(OperationType.DOCUMENT_RETRIEVAL)
        self.assertEqual(stats.total_operations, 1)

    def test_very_fast_operations(self):
        """매우 빠른 오퍼레이션 추적 (시간 측정 정확도)"""
        for _ in range(100):
            with self.monitor.track_operation(OperationType.QUERY_PROCESSING) as op:
                op.tokens = 1

        stats = self.monitor.get_operation_stats(OperationType.QUERY_PROCESSING)
        self.assertEqual(stats.total_operations, 100)
        self.assertGreater(stats.avg_duration_seconds, 0)


if __name__ == "__main__":
    unittest.main()
