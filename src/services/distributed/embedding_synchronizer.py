"""
임베딩 동기화 - 분산 환경에서의 일관성 유지.
"""

import logging
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from threading import RLock
import time

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingChecksum:
    """임베딩 체크섬."""

    job_id: str
    content_hash: str  # 텍스트 내용의 해시
    model_hash: str  # 모델 정보의 해시
    embedding_hash: str  # 임베딩 데이터의 해시
    timestamp: float = field(default_factory=time.time)

    def matches(self, other: "EmbeddingChecksum") -> bool:
        """체크섬 일치 확인."""
        return (
            self.content_hash == other.content_hash
            and self.model_hash == other.model_hash
            and self.embedding_hash == other.embedding_hash
        )


class EmbeddingSynchronizer:
    """임베딩 동기화 - 여러 노드 간의 일관성 유지."""

    def __init__(self):
        self._lock = RLock()
        self._local_embeddings: Dict[str, List[List[float]]] = {}
        self._remote_embeddings: Dict[str, Dict[str, List[List[float]]]] = {}
        self._checksums: Dict[str, EmbeddingChecksum] = {}
        self._sync_status: Dict[str, Dict] = {}

    def register_local_embeddings(
        self,
        job_id: str,
        embeddings: List[List[float]],
        model_name: str,
        texts: List[str],
    ) -> EmbeddingChecksum:
        """로컬 임베딩 등록.

        Args:
            job_id: 작업 ID
            embeddings: 임베딩 데이터
            model_name: 모델 이름
            texts: 원본 텍스트

        Returns:
            생성된 체크섬
        """
        with self._lock:
            # 체크섬 계산
            content_hash = self._hash_texts(texts)
            model_hash = self._hash_string(model_name)
            embedding_hash = self._hash_embeddings(embeddings)

            checksum = EmbeddingChecksum(
                job_id=job_id,
                content_hash=content_hash,
                model_hash=model_hash,
                embedding_hash=embedding_hash,
            )

            self._local_embeddings[job_id] = embeddings
            self._checksums[job_id] = checksum
            self._sync_status[job_id] = {
                "local": True,
                "nodes": {},
                "last_sync": time.time(),
            }

            logger.info(f"로컬 임베딩 등록: {job_id}")

            return checksum

    def register_remote_embeddings(
        self,
        job_id: str,
        node_id: str,
        embeddings: List[List[float]],
    ) -> bool:
        """원격 임베딩 등록.

        Args:
            job_id: 작업 ID
            node_id: 노드 ID
            embeddings: 임베딩 데이터

        Returns:
            등록 성공 여부
        """
        with self._lock:
            if job_id not in self._local_embeddings:
                logger.warning(f"로컬 임베딩이 없음: {job_id}")
                return False

            if job_id not in self._remote_embeddings:
                self._remote_embeddings[job_id] = {}

            self._remote_embeddings[job_id][node_id] = embeddings

            # 동기화 상태 업데이트
            if job_id in self._sync_status:
                self._sync_status[job_id]["nodes"][node_id] = True

            logger.info(f"원격 임베딩 등록: {job_id} <- {node_id}")

            return True

    def verify_consistency(
        self,
        job_id: str,
        embeddings: List[List[float]],
        model_name: str,
        texts: List[str],
    ) -> Tuple[bool, Optional[str]]:
        """임베딩 일관성 검증.

        Args:
            job_id: 작업 ID
            embeddings: 검증할 임베딩
            model_name: 모델 이름
            texts: 원본 텍스트

        Returns:
            (일관성_여부, 오류_메시지)
        """
        with self._lock:
            # 텍스트 크기 확인
            if len(embeddings) != len(texts):
                return False, f"임베딩 수 불일치: {len(embeddings)} != {len(texts)}"

            # 체크섬 계산
            content_hash = self._hash_texts(texts)
            model_hash = self._hash_string(model_name)
            embedding_hash = self._hash_embeddings(embeddings)

            new_checksum = EmbeddingChecksum(
                job_id=job_id,
                content_hash=content_hash,
                model_hash=model_hash,
                embedding_hash=embedding_hash,
            )

            # 기존 체크섬과 비교
            if job_id in self._checksums:
                stored_checksum = self._checksums[job_id]

                if not new_checksum.matches(stored_checksum):
                    return False, "체크섬 불일치"

            # 임베딩 벡터 크기 확인
            if embeddings:
                vec_dim = len(embeddings[0])
                for i, vec in enumerate(embeddings):
                    if len(vec) != vec_dim:
                        return (
                            False,
                            f"임베딩 차원 불일치 ({i}번째): {len(vec)} != {vec_dim}",
                        )

            return True, None

    def synchronize_embeddings(
        self,
        job_id: str,
        resolution_strategy: str = "average",
    ) -> Optional[List[List[float]]]:
        """여러 노드의 임베딩 동기화.

        Args:
            job_id: 작업 ID
            resolution_strategy: 불일치 해결 전략
                - 'average': 평균
                - 'local': 로컬 우선
                - 'majority': 다수결

        Returns:
            동기화된 임베딩 (또는 None)
        """
        with self._lock:
            if job_id not in self._local_embeddings:
                logger.error(f"로컬 임베딩이 없음: {job_id}")
                return None

            local_embeddings = self._local_embeddings[job_id]
            remote_embeddings_dict = self._remote_embeddings.get(job_id, {})

            # 원격 임베딩이 없으면 로컬 반환
            if not remote_embeddings_dict:
                return local_embeddings

            # 해결 전략 적용
            if resolution_strategy == "average":
                return self._merge_by_average(local_embeddings, remote_embeddings_dict)
            elif resolution_strategy == "local":
                return local_embeddings
            elif resolution_strategy == "majority":
                return self._merge_by_majority(local_embeddings, remote_embeddings_dict)
            else:
                logger.warning(f"알 수 없는 해결 전략: {resolution_strategy}")
                return local_embeddings

    def _merge_by_average(
        self,
        local: List[List[float]],
        remote_dict: Dict[str, List[List[float]]],
    ) -> List[List[float]]:
        """평균을 이용한 병합."""
        all_embeddings = [local] + list(remote_dict.values())

        num_embeddings = len(local)
        if not num_embeddings:
            return local

        merged = []
        for i in range(num_embeddings):
            vectors = [emb[i] for emb in all_embeddings]

            # 벡터 차원
            vec_dim = len(vectors[0]) if vectors else 0
            if not vec_dim:
                continue

            # 각 차원의 평균
            avg_vector = [
                sum(vec[d] for vec in vectors) / len(vectors) for d in range(vec_dim)
            ]

            merged.append(avg_vector)

        return merged

    def _merge_by_majority(
        self,
        local: List[List[float]],
        remote_dict: Dict[str, List[List[float]]],
    ) -> List[List[float]]:
        """다수결을 이용한 병합 (임베딩이 유사한 경우 우선)."""
        # 간단한 구현: 로컬 임베딩이 대부분 일치하면 로컬 사용
        all_embeddings = [local] + list(remote_dict.values())

        if len(all_embeddings) <= 1:
            return local

        # 로컬과 원격의 유사도 계산
        similarity_scores = []
        for remote in all_embeddings[1:]:
            sim = self._calculate_similarity(local, remote)
            similarity_scores.append(sim)

        # 평균 유사도가 높으면 로컬 사용
        if similarity_scores and sum(similarity_scores) / len(similarity_scores) > 0.95:
            return local
        else:
            # 아니면 평균 사용
            return self._merge_by_average(local, remote_dict)

    def _calculate_similarity(
        self,
        embeddings1: List[List[float]],
        embeddings2: List[List[float]],
    ) -> float:
        """두 임베딩 집합의 유사도 (0-1)."""
        if len(embeddings1) != len(embeddings2):
            return 0.0

        if not embeddings1:
            return 1.0

        similarities = []
        for v1, v2 in zip(embeddings1, embeddings2):
            # 코사인 유사도
            dot_product = sum(a * b for a, b in zip(v1, v2))
            norm1 = sum(a**2 for a in v1) ** 0.5
            norm2 = sum(b**2 for b in v2) ** 0.5

            if norm1 > 0 and norm2 > 0:
                sim = dot_product / (norm1 * norm2)
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def get_sync_status(self, job_id: str) -> Optional[Dict]:
        """동기화 상태 조회."""
        with self._lock:
            return self._sync_status.get(job_id)

    def get_all_sync_status(self) -> Dict[str, Dict]:
        """모든 동기화 상태."""
        with self._lock:
            return dict(self._sync_status)

    def clear_embeddings(self, job_id: str):
        """임베딩 데이터 정리."""
        with self._lock:
            self._local_embeddings.pop(job_id, None)
            self._remote_embeddings.pop(job_id, None)
            self._checksums.pop(job_id, None)
            self._sync_status.pop(job_id, None)

            logger.info(f"임베딩 정리: {job_id}")

    @staticmethod
    def _hash_texts(texts: List[str]) -> str:
        """텍스트 리스트의 해시."""
        content = "".join(texts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_string(s: str) -> str:
        """문자열 해시."""
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_embeddings(embeddings: List[List[float]]) -> str:
        """임베딩 데이터의 해시."""
        if not embeddings:
            return hashlib.sha256(b"").hexdigest()[:16]

        # 임베딩을 문자열로 변환
        content = str(embeddings)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class ConsistencyValidator:
    """일관성 검증자."""

    def __init__(self):
        self._lock = RLock()
        self._validation_results: Dict[str, Dict] = {}

    def validate_batch(
        self,
        job_id: str,
        embeddings: List[List[float]],
        expected_count: int,
        expected_dim: int,
    ) -> Tuple[bool, List[str]]:
        """배치 유효성 검증.

        Returns:
            (유효_여부, 오류_메시지_리스트)
        """
        errors = []

        # 임베딩 개수 확인
        if len(embeddings) != expected_count:
            errors.append(f"임베딩 개수 불일치: {len(embeddings)} != {expected_count}")

        # 임베딩 차원 확인
        for i, embedding in enumerate(embeddings):
            if len(embedding) != expected_dim:
                errors.append(
                    f"임베딩 {i}의 차원 불일치: {len(embedding)} != {expected_dim}"
                )
                break

        # 임베딩 값 범위 확인
        for i, embedding in enumerate(embeddings):
            for j, value in enumerate(embedding):
                if not isinstance(value, (int, float)):
                    errors.append(f"임베딩 {i}[{j}]의 타입 오류")
                    break

        is_valid = len(errors) == 0

        with self._lock:
            self._validation_results[job_id] = {
                "valid": is_valid,
                "errors": errors,
                "timestamp": time.time(),
            }

        return is_valid, errors

    def get_validation_result(self, job_id: str) -> Optional[Dict]:
        """검증 결과 조회."""
        with self._lock:
            return self._validation_results.get(job_id)

    def is_valid(self, job_id: str) -> bool:
        """유효성 확인."""
        result = self.get_validation_result(job_id)
        return result and result.get("valid", False) if result else False


class CacheSync:
    """분산 캐시 동기화."""

    def __init__(self):
        self._lock = RLock()
        self._local_cache: Dict[str, List[List[float]]] = {}
        self._cache_version: Dict[str, int] = {}

    def put_in_cache(self, key: str, embeddings: List[List[float]]):
        """캐시에 임베딩 저장."""
        with self._lock:
            self._local_cache[key] = embeddings
            self._cache_version[key] = self._cache_version.get(key, 0) + 1

    def get_from_cache(self, key: str) -> Optional[List[List[float]]]:
        """캐시에서 임베딩 조회."""
        with self._lock:
            return self._local_cache.get(key)

    def invalidate_cache(self, key: str):
        """캐시 무효화."""
        with self._lock:
            self._local_cache.pop(key, None)
            self._cache_version[key] = 0

    def get_cache_size(self) -> int:
        """캐시 크기."""
        with self._lock:
            return len(self._local_cache)

    def clear_all_cache(self):
        """전체 캐시 정리."""
        with self._lock:
            self._local_cache.clear()
            self._cache_version.clear()
