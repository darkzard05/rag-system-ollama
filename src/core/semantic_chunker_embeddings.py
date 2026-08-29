"""
임베딩 생성/캐시 관심사 모듈 — ``SemanticChunkerEmbeddingsMixin``.

배치 임베딩 생성(``_get_embeddings``), 캐시 벡터 검증(``_as_valid_vector``),
차원 복구 재임베딩(``_reembed_single``), 표준 차원 결정(``_resolve_expected_dim``),
그리고 ``split_text``의 buffer 기반 combined embedding/거리 계산
(``_buffer_combined_embeddings``, ``_compute_distances``)을 담당합니다.

[R2-10] ``semantic_chunker.py`` 모노리스에서 관심사별로 분리한 모듈입니다.
"""

import asyncio
import logging
from typing import Any, cast

import numpy as np
import xxhash
from langchain_core.embeddings import Embeddings

from services.optimization.caching_optimizer import CacheManager

logger = logging.getLogger(__name__)


def _is_ollama_embedder(embedder: Embeddings) -> bool:
    """Ollama 기반 임베더인지 판별합니다.

    model_loader가 이미 langchain_ollama를 로드한 뒤 호출되므로 추가 임포트
    비용 없이 정확한 isinstance 판별이 가능합니다. langchain_ollama가
    설치되지 않은 환경에서는 False를 반환합니다.
    """
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        return False
    return isinstance(embedder, OllamaEmbeddings)


class SemanticChunkerEmbeddingsMixin:
    """
    임베딩 생성/캐시 관심사 믹스인.

    아래 속성들은 ``EmbeddingBasedSemanticChunker.__init__``에서 설정됩니다.
    """

    embedder: Embeddings
    model_name: str
    cache_manager: CacheManager
    batch_size: int
    buffer_size: int

    async def _get_embeddings(
        self, texts: list[str], normalize: bool = False
    ) -> np.ndarray:
        """
        텍스트 리스트의 임베딩을 생성합니다 (배치 처리 및 캐싱 강화).
        [수정] 차원 불일치/None 캐시 벡터는 제거하지 않고 해당 텍스트를
        재임베딩하여 복구하며, 반환 행렬의 행 수는 항상 입력 텍스트 수와 동일합니다.
        """
        if not texts:
            return np.array([]).reshape(0, 0)

        all_results: list[np.ndarray | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        text_to_output_indices: dict[str, list[int]] = {}
        newly_embedded: list[tuple[int, str, np.ndarray]] = []

        # 1. 정제 및 캐시 확인
        for i, text in enumerate(texts):
            norm_text = " ".join(text.split())
            cache_key = (
                f"emb:{self.model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"
            )

            cached_raw = await self.cache_manager.get(cache_key)
            if cached_raw is not None:
                cached_vec = (
                    cached_raw["vector"]
                    if isinstance(cached_raw, dict) and "vector" in cached_raw
                    else cached_raw
                )
                cached_vector = self._as_valid_vector(cached_vec)
                if cached_vector is not None:
                    all_results[i] = cached_vector
                    continue

            if norm_text not in text_to_output_indices:
                missing_texts.append(norm_text)
            text_to_output_indices.setdefault(norm_text, []).append(i)
            missing_indices.append(i)

        # 2. 누락분 배치 임베딩 수행 (캐시 저장은 모든 배치 성공 후 단일 패스)
        if missing_texts:
            logger.debug(
                f"[Chunker] {len(missing_texts)}개 문장 신규 임베딩 생성 중 (Batch Size: {self.batch_size})..."
            )

            # [최적화] Ollama 임베더는 embed_documents가 입력 전체를 단일 HTTP
            # 요청으로 전송하므로, 누락분을 한 번의 embed_documents 호출로 모두
            # 묶어 HTTP 왕복 횟수를 1로 고정한다. (배치 사이즈로 쪼개면 왕복만
            # 늘어나 성능이 떨어짐) 비-Ollama(HuggingFace 등)는 기존처럼
            # batch_size 단위로 분할하여 메모리/디바이스 제약을 존중한다.
            if _is_ollama_embedder(self.embedder):
                embed_batches: list[list[str]] = [missing_texts]
            else:
                embed_batches = [
                    missing_texts[b : b + self.batch_size]
                    for b in range(0, len(missing_texts), self.batch_size)
                ]

            # [수정] 배치 루프 동안 메모리에만 수집, 캐시 저장은 루프 이후 단일 패스로 수행
            for b_idx, batch in enumerate(embed_batches):
                batch_indices = missing_indices[b_idx : b_idx + len(batch)]

                try:
                    # [최적화] 동기 임베딩 생성을 비동기 스레드로 분리 (embed 동안 모델 pin)
                    from core.resource_manager import get_resource_manager

                    coordinator = get_resource_manager()
                    async with coordinator.use_embedder(embedder=self.embedder):
                        batch_vecs = await asyncio.to_thread(
                            self.embedder.embed_documents, batch
                        )

                    for batch_text, vec in zip(batch, batch_vecs, strict=False):
                        vec_np = np.array(vec, dtype="float32")
                        for output_idx in text_to_output_indices.get(batch_text, []):
                            all_results[output_idx] = vec_np
                            newly_embedded.append(
                                (output_idx, texts[output_idx], vec_np)
                            )

                except Exception as e:
                    logger.error(
                        f"[Chunker] 배치 {b_idx} 임베딩 생성 중 오류 발생: {e}",
                        exc_info=True,
                    )
                    # 배치 실패 시 해당 인덱스의 결과 제거
                    for idx in batch_indices:
                        all_results[idx] = None
                    raise

            # [수정] 모든 배치 성공 후 단일 패스로 캐시 저장 (부분 캐시 저장 방지)
            try:
                for norm_text, vec_np in {
                    text: vec for _, text, vec in newly_embedded
                }.items():
                    cache_key = f"emb:{self.model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"
                    await asyncio.wait_for(
                        self.cache_manager.set(
                            cache_key,
                            {"vector": vec_np.tolist(), "cache_version": "1.0"},
                            persist_to_disk=True,
                        ),
                        timeout=30.0,
                    )
            except asyncio.TimeoutError:
                logger.error("[Chunker] 캐시 저장 타임아웃 (일부 항목 미저장 가능)")
                raise

        # 3. [수정] 차원 불일치/누락 벡터는 개별 재임베딩으로 복구 (제거하지 않음)
        expected_dim = self._resolve_expected_dim(all_results, newly_embedded)
        if expected_dim is None:
            raise ValueError(
                "[Chunker] 임베딩 결과가 하나도 없어 벡터 행렬을 구성할 수 없습니다."
            )

        for i, res_vec in enumerate(all_results):
            if res_vec is None or int(res_vec.shape[0]) != expected_dim:
                logger.warning(
                    f"[Chunker] 캐시된 벡터 차원 불일치 감지 (Index: {i}, "
                    f"Dim: {None if res_vec is None else res_vec.shape[0]} "
                    f"!= Expected: {expected_dim}). 해당 문장을 재임베딩하여 복구합니다."
                )
                all_results[i] = await self._reembed_single(texts[i], expected_dim)

        # 4. [수정] 행렬 조립 — 행 수는 항상 입력 텍스트 수와 동일해야 함
        valid_vectors = [cast(np.ndarray, v) for v in all_results]
        if len(valid_vectors) != len(texts):
            raise ValueError(
                f"[Chunker] 임베딩 행렬의 행 수({len(valid_vectors)})가 "
                f"입력 텍스트 수({len(texts)})와 일치하지 않습니다."
            )

        embeddings_matrix = np.stack(valid_vectors).astype("float32")
        if normalize:
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            embeddings_matrix = np.divide(
                embeddings_matrix,
                norms,
                out=np.zeros_like(embeddings_matrix),
                where=norms > 1e-9,
            )
        return embeddings_matrix

    @staticmethod
    def _as_valid_vector(cached_vec: Any) -> np.ndarray | None:
        """캐시된 값이 1차원 벡터로 해석 가능하면 np.ndarray로, 아니면 None을 반환합니다."""
        try:
            vec_np = np.asarray(cached_vec, dtype="float32")
        except (TypeError, ValueError):
            return None
        if vec_np.ndim != 1 or vec_np.shape[0] == 0:
            return None
        return vec_np

    async def _reembed_single(self, text: str, expected_dim: int) -> np.ndarray:
        """차원 불일치/None 캐시 벡터에 대해 해당 텍스트를 재임베딩하여 복구합니다."""
        norm_text = " ".join(text.split())
        try:
            from core.resource_manager import get_resource_manager

            coordinator = get_resource_manager()
            async with coordinator.use_embedder(embedder=self.embedder):
                vecs = await asyncio.to_thread(
                    self.embedder.embed_documents, [norm_text]
                )
        except Exception as e:
            raise ValueError(f"[Chunker] 캐시 벡터 복구 재임베딩 실패: {e}") from e
        if not vecs:
            raise ValueError(
                f"[Chunker] 캐시 벡터 복구 재임베딩 결과가 비어 있습니다: {norm_text[:80]!r}"
            )
        vec_np = np.asarray(vecs[0], dtype="float32")
        if vec_np.ndim != 1 or vec_np.shape[0] != expected_dim:
            raise ValueError(
                f"[Chunker] 캐시 벡터 복구 재임베딩 후에도 차원 불일치 "
                f"(text: {norm_text[:80]!r}, expected dim: {expected_dim}, "
                f"got: {vec_np.shape})."
            )

        cache_key = (
            f"emb:{self.model_name}:{xxhash.xxh64(norm_text.encode()).hexdigest()}"
        )
        await asyncio.wait_for(
            self.cache_manager.set(
                cache_key,
                {"vector": vec_np.tolist(), "cache_version": "1.0"},
                persist_to_disk=True,
            ),
            timeout=30.0,
        )
        return vec_np

    @staticmethod
    def _resolve_expected_dim(
        all_results: list[np.ndarray | None],
        newly_embedded: list[tuple[int, str, np.ndarray]],
    ) -> int | None:
        """새로 임베딩된 벡터를 우선, 없으면 캐시 벡터로 표준 차원을 결정합니다."""
        for _, _, vec in newly_embedded:
            return int(vec.shape[0])
        for res_vec in all_results:
            if res_vec is not None:
                return int(res_vec.shape[0])
        return None

    def _buffer_combined_embeddings(self, indiv_embeddings: np.ndarray) -> np.ndarray:
        """문장별 임베딩에 buffer 기반 컨텍스트 윈도우를 적용한 combined 벡터를 계산합니다."""
        combined_embeddings = []
        for i in range(len(indiv_embeddings)):
            start = max(0, i - self.buffer_size)
            end = min(len(indiv_embeddings), i + self.buffer_size + 1)
            window_vectors = indiv_embeddings[start:end]
            combined_vec = np.mean(window_vectors, axis=0)
            norm = np.linalg.norm(combined_vec)
            if norm > 1e-9:
                combined_vec /= norm
            combined_embeddings.append(combined_vec)
        return np.array(combined_embeddings)

    def _compute_distances(self, combined_embeddings_arr: np.ndarray) -> list[float]:
        """인접 combined 벡터 간 유사도를 1-cos_sim 거리로 변환합니다."""
        distances = []
        for i in range(len(combined_embeddings_arr) - 1):
            similarity = np.dot(
                combined_embeddings_arr[i], combined_embeddings_arr[i + 1]
            )
            distances.append(1.0 - float(similarity))
        return distances
