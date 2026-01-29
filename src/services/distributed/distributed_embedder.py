"""
분산 임베딩 생성 - Ray/Dask 기반 병렬 처리.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Callable
from threading import RLock
import time

try:
    import ray

    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """작업 상태."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class EmbeddingJob:
    """임베딩 작업 정의."""

    job_id: str
    texts: List[str]
    model_name: str
    batch_size: int = 32
    priority: int = 0  # 0=low, 1=normal, 2=high
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[List[List[float]]] = None
    error: Optional[str] = None

    def duration(self) -> Optional[float]:
        """작업 소요 시간."""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        return None

    def is_complete(self) -> bool:
        """작업 완료 여부."""
        return self.status in [
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ]


class DistributedEmbedder:
    """분산 임베딩 생성기."""

    def __init__(
        self,
        embedding_func: Callable[[List[str], str], List[List[float]]],
        num_workers: Optional[int] = None,
        use_ray: bool = True,
    ):
        """초기화.

        Args:
            embedding_func: 임베딩 함수 (texts, model_name) -> List[embeddings]
            num_workers: 워커 수 (기본: CPU 코어 수)
            use_ray: Ray 사용 여부
        """
        self.embedding_func = embedding_func
        self.num_workers = num_workers
        self.use_ray = use_ray and RAY_AVAILABLE

        self._lock = RLock()
        self._jobs: Dict[str, EmbeddingJob] = {}
        self._job_queue: List[EmbeddingJob] = []
        self._results: Dict[str, List[List[float]]] = {}

        if self.use_ray:
            self._init_ray()

    def _init_ray(self):
        """Ray 초기화."""
        if not ray.is_initialized():
            try:
                ray.init(num_cpus=self.num_workers, ignore_reinit_error=True)
                logger.info(f"Ray 초기화 완료 (워커: {self.num_workers})")
            except Exception as e:
                logger.warning(f"Ray 초기화 실패: {e}, CPU 기반 처리로 변경")
                self.use_ray = False

    def submit_job(
        self,
        job_id: str,
        texts: List[str],
        model_name: str,
        batch_size: int = 32,
        priority: int = 0,
    ) -> EmbeddingJob:
        """임베딩 작업 제출.

        Args:
            job_id: 작업 ID (고유값)
            texts: 임베딩할 텍스트 목록
            model_name: 모델 이름
            batch_size: 배치 크기
            priority: 우선순위 (0=low, 1=normal, 2=high)

        Returns:
            제출된 작업 객체
        """
        if not texts:
            logger.warning(f"빈 텍스트 목록: {job_id}")
            return None

        with self._lock:
            # 중복 확인
            if job_id in self._jobs:
                logger.warning(f"이미 존재하는 작업: {job_id}")
                return self._jobs[job_id]

            # 작업 생성
            job = EmbeddingJob(
                job_id=job_id,
                texts=texts,
                model_name=model_name,
                batch_size=batch_size,
                priority=priority,
            )

            self._jobs[job_id] = job
            self._job_queue.append(job)

            # 우선순위 정렬
            self._job_queue.sort(key=lambda j: j.priority, reverse=True)

            logger.info(f"작업 제출: {job_id} (텍스트: {len(texts)}개)")

            return job

    def process_job_sync(self, job_id: str) -> Optional[List[List[float]]]:
        """작업 동기식 처리.

        Args:
            job_id: 작업 ID

        Returns:
            임베딩 결과
        """
        with self._lock:
            if job_id not in self._jobs:
                logger.error(f"작업을 찾을 수 없음: {job_id}")
                return None

            job = self._jobs[job_id]

        try:
            # 작업 시작
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            logger.info(f"작업 처리 시작: {job_id}")

            # 배치 처리
            embeddings = self._process_batches(job)

            # 작업 완료
            job.status = JobStatus.COMPLETED
            job.result = embeddings
            job.completed_at = time.time()

            with self._lock:
                self._results[job_id] = embeddings

            logger.info(
                f"작업 완료: {job_id} "
                f"({job.duration():.2f}s, {len(embeddings)}개 임베딩)"
            )

            return embeddings

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

            logger.error(f"작업 실패: {job_id} - {e}")
            return None

    def _process_batches(self, job: EmbeddingJob) -> List[List[float]]:
        """배치 처리."""
        embeddings = []

        for i in range(0, len(job.texts), job.batch_size):
            batch = job.texts[i : i + job.batch_size]

            try:
                batch_embeddings = self.embedding_func(batch, job.model_name)
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"배치 임베딩 실패 ({i}): {e}")
                raise

        return embeddings

    def process_jobs_distributed(
        self,
        job_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[List[float]]]:
        """여러 작업을 분산 처리.

        Args:
            job_ids: 처리할 작업 ID 목록 (None=대기 중인 모든 작업)

        Returns:
            {job_id: embeddings} 딕셔너리
        """
        if not job_ids:
            with self._lock:
                job_ids = [
                    j.job_id for j in self._job_queue if j.status == JobStatus.PENDING
                ]

        if not job_ids:
            logger.info("처리할 작업이 없습니다")
            return {}

        results = {}

        if self.use_ray:
            results = self._process_with_ray(job_ids)
        else:
            results = self._process_sequential(job_ids)

        return results

    def _process_with_ray(self, job_ids: List[str]) -> Dict[str, List[List[float]]]:
        """Ray를 사용한 분산 처리."""
        results = {}

        # 원격 함수 정의
        @ray.remote
        def process_remote_job(
            job_id: str,
            embedding_func,
            texts: List[str],
            model_name: str,
            batch_size: int,
        ):
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embeddings = embedding_func(batch, model_name)
                embeddings.extend(batch_embeddings)
            return job_id, embeddings

        # Ray 작업 제출
        remote_tasks = []
        for job_id in job_ids:
            with self._lock:
                if job_id not in self._jobs:
                    continue
                job = self._jobs[job_id]

            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            task = process_remote_job.remote(
                job_id,
                self.embedding_func,
                job.texts,
                job.model_name,
                job.batch_size,
            )
            remote_tasks.append((job_id, task))

        # 결과 수집
        for job_id, task in remote_tasks:
            try:
                _, embeddings = ray.get(task)

                with self._lock:
                    job = self._jobs[job_id]
                    job.status = JobStatus.COMPLETED
                    job.result = embeddings
                    job.completed_at = time.time()
                    self._results[job_id] = embeddings

                results[job_id] = embeddings
                logger.info(f"Ray 작업 완료: {job_id}")

            except Exception as e:
                with self._lock:
                    job = self._jobs[job_id]
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = time.time()

                logger.error(f"Ray 작업 실패: {job_id} - {e}")

        return results

    def _process_sequential(self, job_ids: List[str]) -> Dict[str, List[List[float]]]:
        """순차 처리 (Ray 없을 때)."""
        results = {}

        for job_id in job_ids:
            embeddings = self.process_job_sync(job_id)
            if embeddings:
                results[job_id] = embeddings

        return results

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """작업 상태 조회."""
        with self._lock:
            if job_id not in self._jobs:
                return None
            return self._jobs[job_id].status

    def get_job_result(self, job_id: str) -> Optional[List[List[float]]]:
        """작업 결과 조회."""
        with self._lock:
            return self._results.get(job_id)

    def get_job_info(self, job_id: str) -> Optional[Dict]:
        """작업 정보 조회."""
        with self._lock:
            if job_id not in self._jobs:
                return None

            job = self._jobs[job_id]
            return {
                "job_id": job.job_id,
                "status": job.status.value,
                "text_count": len(job.texts),
                "duration": job.duration(),
                "error": job.error,
            }

    def cancel_job(self, job_id: str) -> bool:
        """작업 취소.

        Returns:
            취소 성공 여부
        """
        with self._lock:
            if job_id not in self._jobs:
                return False

            job = self._jobs[job_id]

            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                logger.warning(f"이미 완료된 작업: {job_id}")
                return False

            job.status = JobStatus.CANCELLED
            logger.info(f"작업 취소: {job_id}")

            return True

    def get_pending_jobs(self) -> List[str]:
        """대기 중인 작업 목록."""
        with self._lock:
            return [j.job_id for j in self._job_queue if j.status == JobStatus.PENDING]

    def get_all_jobs_status(self) -> Dict[str, str]:
        """모든 작업 상태."""
        with self._lock:
            return {job_id: job.status.value for job_id, job in self._jobs.items()}

    def shutdown(self):
        """리소스 정리."""
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray 종료")


class TaskOrchestrator:
    """작업 조율기 - 작업 스케줄링 및 조율."""

    def __init__(self, embedder: DistributedEmbedder):
        self.embedder = embedder
        self._lock = RLock()
        self._job_groups: Dict[str, List[str]] = {}
        self._completed_groups: set = set()

    def create_job_group(
        self,
        group_id: str,
        jobs: List[EmbeddingJob],
    ) -> bool:
        """작업 그룹 생성.

        Returns:
            생성 성공 여부
        """
        with self._lock:
            if group_id in self._job_groups:
                logger.warning(f"이미 존재하는 그룹: {group_id}")
                return False

            job_ids = [job.job_id for job in jobs]
            self._job_groups[group_id] = job_ids

            logger.info(f"작업 그룹 생성: {group_id} ({len(job_ids)}개 작업)")

            return True

    def process_group(self, group_id: str) -> Dict[str, List[List[float]]]:
        """작업 그룹 처리.

        Returns:
            {job_id: embeddings} 딕셔너리
        """
        with self._lock:
            if group_id not in self._job_groups:
                logger.error(f"작업 그룹을 찾을 수 없음: {group_id}")
                return {}

            job_ids = self._job_groups[group_id]

        logger.info(f"작업 그룹 처리: {group_id}")

        results = self.embedder.process_jobs_distributed(job_ids)

        with self._lock:
            if len(results) == len(job_ids):
                self._completed_groups.add(group_id)

        return results

    def is_group_complete(self, group_id: str) -> bool:
        """그룹 완료 여부."""
        with self._lock:
            return group_id in self._completed_groups

    def get_group_status(self, group_id: str) -> Dict[str, str]:
        """그룹 상태."""
        with self._lock:
            if group_id not in self._job_groups:
                return {}

            job_ids = self._job_groups[group_id]

        return self.embedder.get_all_jobs_status()
