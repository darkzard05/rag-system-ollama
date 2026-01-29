"""
Task 18: 분산 임베딩 테스트
- 분산 임베딩 생성, 클러스터 관리, 동기화, 일관성 검증

Target: 16+ 테스트 (100% 통과)
"""

import pytest
import time
from typing import List

from src.services.distributed.distributed_embedder import (
    DistributedEmbedder,
    TaskOrchestrator,
    JobStatus,
)
from src.services.distributed.cluster_manager import (
    ClusterManager,
    NodeStatus,
    HealthCheckConfig,
)
from src.services.distributed.embedding_synchronizer import (
    EmbeddingSynchronizer,
    ConsistencyValidator,
)


# ============================================================================
# 헬퍼 함수 및 픽스처
# ============================================================================


def mock_embedding_func(texts: List[str], model_name: str) -> List[List[float]]:
    """모의 임베딩 함수."""
    # 간단한 구현: 텍스트 길이 기반 임베딩
    embeddings = []
    for text in texts:
        embedding = [float(ord(c)) / 255.0 for c in text[:128]]
        # 벡터 크기를 384로 고정
        while len(embedding) < 384:
            embedding.append(0.0)
        embeddings.append(embedding[:384])
    return embeddings


@pytest.fixture
def embedder():
    """분산 임베딩 생성기."""
    return DistributedEmbedder(
        embedding_func=mock_embedding_func,
        num_workers=2,
        use_ray=False,  # 테스트에서는 Ray 미사용
    )


@pytest.fixture
def cluster_manager():
    """클러스터 관리자."""
    config = HealthCheckConfig(interval_seconds=1.0)
    return ClusterManager(num_nodes=3, config=config)


@pytest.fixture
def synchronizer():
    """임베딩 동기화기."""
    return EmbeddingSynchronizer()


# ============================================================================
# Test Group 1: 분산 임베딩 생성 (4개)
# ============================================================================


class TestDistributedEmbedding:
    """분산 임베딩 테스트."""

    def test_01_submit_job(self, embedder):
        """작업 제출."""
        texts = ["hello world", "test text", "embedding task"]
        job = embedder.submit_job(
            job_id="job-1",
            texts=texts,
            model_name="test-model",
        )

        assert job is not None
        assert job.job_id == "job-1"
        assert job.status == JobStatus.PENDING
        assert len(job.texts) == 3

    def test_02_process_job_sync(self, embedder):
        """작업 동기식 처리."""
        texts = ["hello", "world"]
        embedder.submit_job(
            job_id="job-2",
            texts=texts,
            model_name="test-model",
        )

        result = embedder.process_job_sync("job-2")

        assert result is not None
        assert len(result) == 2
        assert all(len(emb) == 384 for emb in result)

        # 작업 상태 확인
        assert embedder.get_job_status("job-2") == JobStatus.COMPLETED

    def test_03_process_distributed(self, embedder):
        """분산 처리."""
        embedder.submit_job("job-3", ["text1"], "model")
        embedder.submit_job("job-4", ["text2", "text3"], "model")

        results = embedder.process_jobs_distributed(["job-3", "job-4"])

        assert len(results) == 2
        assert "job-3" in results
        assert "job-4" in results

    def test_04_job_priority(self, embedder):
        """작업 우선순위."""
        embedder.submit_job("low", ["a"], "model", priority=0)
        embedder.submit_job("high", ["b"], "model", priority=2)
        embedder.submit_job("normal", ["c"], "model", priority=1)

        pending = embedder.get_pending_jobs()

        # 우선순위 확인
        assert pending[0] == "high"  # 가장 높은 우선순위


# ============================================================================
# Test Group 2: 클러스터 관리 (5개)
# ============================================================================


class TestClusterManager:
    """클러스터 관리자 테스트."""

    def test_05_cluster_init(self, cluster_manager):
        """클러스터 초기화."""
        metrics = cluster_manager.get_all_nodes_metrics()

        assert len(metrics) == 3
        assert all(node_id.startswith("node-") for node_id in metrics.keys())

    def test_06_health_monitoring(self, cluster_manager):
        """헬스 모니터링."""
        time.sleep(1.5)  # 모니터링 주기 대기

        metrics = cluster_manager.get_all_nodes_metrics()
        for node in metrics.values():
            assert node.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED]

    def test_07_cluster_status(self, cluster_manager):
        """클러스터 상태."""
        status = cluster_manager.get_cluster_status()

        assert "total_nodes" in status
        assert "healthy_nodes" in status
        assert status["total_nodes"] == 3

    def test_08_select_best_node(self, cluster_manager):
        """최적 노드 선택."""
        best_node = cluster_manager.select_best_node()

        assert best_node is not None
        assert best_node.startswith("node-")

    def test_09_node_load(self, cluster_manager):
        """노드 부하."""
        load = cluster_manager.get_node_load("node-0")

        assert load is not None
        assert 0.0 <= load <= 1.0


# ============================================================================
# Test Group 3: 임베딩 동기화 (4개)
# ============================================================================


class TestEmbeddingSynchronizer:
    """임베딩 동기화 테스트."""

    def test_10_register_local(self, synchronizer):
        """로컬 임베딩 등록."""
        embeddings = [[0.1, 0.2], [0.3, 0.4]]
        texts = ["text1", "text2"]

        checksum = synchronizer.register_local_embeddings(
            "job-1",
            embeddings,
            "test-model",
            texts,
        )

        assert checksum is not None
        assert checksum.job_id == "job-1"

    def test_11_consistency_check(self, synchronizer):
        """일관성 확인."""
        embeddings = [[0.1] * 384, [0.2] * 384]
        texts = ["text1", "text2"]

        synchronizer.register_local_embeddings("job-2", embeddings, "model", texts)

        # 같은 데이터로 검증
        is_consistent, error = synchronizer.verify_consistency(
            "job-2",
            embeddings,
            "model",
            texts,
        )

        assert is_consistent
        assert error is None

    def test_12_register_remote(self, synchronizer):
        """원격 임베딩 등록."""
        local_emb = [[0.1] * 384, [0.2] * 384]
        texts = ["a", "b"]

        synchronizer.register_local_embeddings("job-3", local_emb, "model", texts)

        remote_emb = [[0.15] * 384, [0.25] * 384]
        result = synchronizer.register_remote_embeddings("job-3", "node-1", remote_emb)

        assert result is True

    def test_13_synchronize(self, synchronizer):
        """임베딩 동기화."""
        local_emb = [[0.1] * 384, [0.2] * 384]
        texts = ["text1", "text2"]

        synchronizer.register_local_embeddings("job-4", local_emb, "model", texts)

        remote_emb = [[0.15] * 384, [0.25] * 384]
        synchronizer.register_remote_embeddings("job-4", "node-1", remote_emb)

        merged = synchronizer.synchronize_embeddings("job-4", "average")

        assert merged is not None
        assert len(merged) == 2


# ============================================================================
# Test Group 4: 작업 조율 (2개)
# ============================================================================


class TestTaskOrchestrator:
    """작업 조율기 테스트."""

    def test_14_job_group(self, embedder):
        """작업 그룹."""
        orchestrator = TaskOrchestrator(embedder)

        job1 = embedder.submit_job("j1", ["text1"], "model")
        job2 = embedder.submit_job("j2", ["text2"], "model")

        result = orchestrator.create_job_group("group-1", [job1, job2])

        assert result is True

    def test_15_process_group(self, embedder):
        """그룹 처리."""
        orchestrator = TaskOrchestrator(embedder)

        job1 = embedder.submit_job("j3", ["a"], "model")
        job2 = embedder.submit_job("j4", ["b"], "model")

        orchestrator.create_job_group("group-2", [job1, job2])
        results = orchestrator.process_group("group-2")

        assert len(results) == 2


# ============================================================================
# Test Group 5: 일관성 검증 및 캐시 (2개)
# ============================================================================


class TestConsistency:
    """일관성 검증 테스트."""

    def test_16_validator(self):
        """일관성 검증자."""
        validator = ConsistencyValidator()

        embeddings = [[0.1] * 384 for _ in range(10)]

        is_valid, errors = validator.validate_batch(
            "job-1",
            embeddings,
            expected_count=10,
            expected_dim=384,
        )

        assert is_valid
        assert len(errors) == 0


# ============================================================================
# Test Group 6: 통합 테스트 (1개+)
# ============================================================================


class TestIntegration:
    """통합 테스트."""

    def test_17_end_to_end(self, embedder, cluster_manager, synchronizer):
        """엔드-투-엔드 워크플로우."""
        # 1. 작업 제출
        texts = ["document 1", "document 2", "document 3"]
        embedder.submit_job("integrated", texts, "model")

        # 2. 처리
        results = embedder.process_jobs_distributed(["integrated"])
        assert "integrated" in results

        # 3. 동기화
        embeddings = results["integrated"]
        synchronizer.register_local_embeddings("integrated", embeddings, "model", texts)

        # 4. 클러스터 상태 확인
        status = cluster_manager.get_cluster_status()
        assert status["total_nodes"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
