"""
임베딩 기반 의미론적 텍스트 분할기를 구현합니다.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로
일관성 있는 청크를 생성합니다.
"""

import asyncio
import concurrent.futures
import logging
import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import xxhash
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.constants import ChunkingConstants
from services.optimization.caching_optimizer import CacheManager, get_cache_manager

# 순환 참조 방지 및 타입 힌트용
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# [상수] 표준 섹션 키워드 (정제 시 활용)
STANDARD_SECTION_KEYWORDS = [
    "ABSTRACT",
    "INTRODUCTION",
    "RELATED WORK",
    "METHOD",
    "EXPERIMENT",
    "RESULT",
    "CONCLUSION",
    "REFERENCE",
    "ACKNOWLEDGMENT",
]


class EmbeddingBasedSemanticChunker:
    """
    임베딩 기반 의미론적 텍스트 분할기.

    문장 단위로 분할 후, 임베딩 유사도 기반으로 의미 경계를 탐지하여
    일관성 있는 청크를 생성합니다.
    """

    # 복합 헤더 패턴 (Markdown # + 대문자 섹션명)
    HEADER_PATTERN = re.compile(
        r"^(\s*#{1,6}\s+.+|\d+\s+[A-Z]{2,}.*|[A-Z]{3,}(\s+[A-Z]{3,})*)$",
        re.MULTILINE,
    )
    # 간단 헤더 패턴 (breakpoint 탐지용)
    HEADER_SIMPLE = re.compile(r"^\s*#{1,6}\s+.+", re.MULTILINE)

    def __init__(
        self,
        embedder: Embeddings,
        buffer_size: int = 1,  # [추가] 문맥 윈도우 크기
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_value: float = 95.0,
        sentence_split_regex: str = r"(?<=[.?!])\s+",  # [개선] Lookbehind 적용
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        chunk_overlap: int = 1,  # [추가] 겹칠 문장 수 (Context preservation)
        similarity_threshold: float = 0.6,
        batch_size: int = 64,
        cache_manager: CacheManager | None = None,
    ):
        """
        의미론적 청킹 분할기를 초기화합니다.
        """
        self.embedder = embedder
        self.buffer_size = buffer_size  # [추가]
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_value = breakpoint_threshold_value
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap  # [추가]
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        # [최적화] 전역 캐시 관리자 설정
        self.cache_manager = cache_manager or get_cache_manager()

        # [최적화] 모델 식별을 위한 이름 추출 (Ollama 및 HuggingFace 지원 강화)
        self.model_name = getattr(embedder, "model", None) or getattr(
            embedder, "model_name", "default_model"
        )

        # ✅ 정규식 사전 컴파일 (성능 최적화)
        self._sentence_pattern = re.compile(self.sentence_split_regex)

    def _split_sentences(self, text: str) -> list[dict]:
        """
        문장 단위로 텍스트를 분할하고 오프셋 정보를 함께 반환합니다.
        [최적화] 마크다운 테이블 행(|)을 보호하며, 너무 긴 문장은 강제로 분할합니다.
        """
        lines = text.split("\n")
        temp_sentences: list[dict[str, Any]] = []
        current_pos = 0

        # 1. 행 단위로 훑으며 테이블 보호 및 문장 분할
        for line in lines:
            line_len = len(line)
            stripped = line.strip()

            # 테이블 행 감지 ( | 로 시작하고 | 로 끝나는 패턴)
            if stripped.startswith("|") and stripped.endswith("|"):
                self._add_cleaned_sentence(
                    temp_sentences, line, current_pos, current_pos + line_len
                )
            else:
                # 일반 텍스트는 정규식으로 분할
                parts = list(self._sentence_pattern.finditer(line))
                last_pos = 0
                for match in parts:
                    sep_end = match.end()
                    segment_text = line[last_pos:sep_end]
                    if segment_text.strip():
                        self._add_cleaned_sentence(
                            temp_sentences,
                            segment_text,
                            current_pos + last_pos,
                            current_pos + sep_end,
                        )
                    last_pos = sep_end

                if last_pos < len(line):
                    remaining_text = line[last_pos:]
                    if remaining_text.strip():
                        self._add_cleaned_sentence(
                            temp_sentences,
                            remaining_text,
                            current_pos + last_pos,
                            current_pos + len(line),
                        )

            current_pos += line_len + 1  # \n 포함

        # 2. 너무 긴 세그먼트 강제 분할 (OOM 방지 및 하드 스플릿 플래그 처리)
        hard_split_limit: int = ChunkingConstants.MAX_HARD_SPLIT_LEN.value
        if self.max_chunk_size > 0:
            hard_split_limit = min(
                ChunkingConstants.MAX_HARD_SPLIT_LEN.value,
                int(self.max_chunk_size * 1.5),
            )

        final_sentences: list[dict[str, Any]] = []
        for seg in temp_sentences:
            seg_text = str(seg["text"])
            seg_start = int(seg["start"])

            if len(seg_text) <= hard_split_limit:
                final_sentences.append(seg)
            else:
                # 강제 분할 로직
                curr_pos = 0
                while curr_pos < len(seg_text):
                    sub_len: int = int(hard_split_limit)
                    if curr_pos + sub_len < len(seg_text):
                        # 공백 기준으로 끊기 시도
                        last_space = seg_text.rfind(" ", curr_pos, curr_pos + sub_len)
                        if last_space != -1 and last_space > curr_pos + (sub_len // 2):
                            sub_len = int(last_space - curr_pos + 1)

                    sub_text = seg_text[curr_pos : curr_pos + sub_len]
                    is_last_sub = curr_pos + sub_len >= len(seg_text)

                    self._add_cleaned_sentence(
                        final_sentences,
                        sub_text,
                        seg_start + curr_pos,
                        seg_start + curr_pos + len(sub_text),
                        is_hard_split=not is_last_sub,
                    )
                    curr_pos += sub_len

        return final_sentences

    def _add_cleaned_sentence(
        self,
        target_list: list[dict[str, Any]],
        raw_text: str,
        start_offset: int,
        end_offset: int,
        is_hard_split: bool = False,
    ):
        """정제된 문장을 리스트에 추가합니다."""
        l_stripped = raw_text.lstrip()
        n_leading = len(raw_text) - len(l_stripped)

        stripped = l_stripped.rstrip()
        n_trailing = len(l_stripped) - len(stripped)

        final_start = start_offset + n_leading
        final_end = end_offset - n_trailing

        embed_text = stripped.replace("\n", " ")
        if embed_text:
            target_list.append(
                {
                    "text": embed_text,
                    "start": final_start,
                    "end": final_end,
                    "is_hard_split": is_hard_split,
                }
            )

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

            # [수정] 배치 루프 동안 메모리에만 수집, 캐시 저장은 루프 이후 단일 패스로 수행
            for b_idx in range(0, len(missing_texts), self.batch_size):
                batch = missing_texts[b_idx : b_idx + self.batch_size]
                batch_indices = missing_indices[b_idx : b_idx + self.batch_size]

                try:
                    # [최적화] 동기 임베딩 생성을 비동기 스레드로 분리
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
                            persist_to_disk=False,
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
            vecs = await asyncio.to_thread(self.embedder.embed_documents, [norm_text])
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
                persist_to_disk=False,
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

    def _find_breakpoints(
        self, distances: list[float], sentences: list[dict] | None = None
    ) -> list[int]:
        """
        유사도 거리(1-cos_sim) 분포를 분석하여 분할 지점을 찾습니다.
        [고도화] 마크다운 헤더 감지 시 강제 분할 지점으로 추가합니다.
        """
        if not distances:
            return []

        dist_array = np.array(distances)
        threshold = 0.0

        if self.breakpoint_threshold_type == "percentile":
            threshold = float(
                np.percentile(dist_array, self.breakpoint_threshold_value)
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            threshold = float(
                np.mean(dist_array)
                + self.breakpoint_threshold_value * np.std(dist_array)
            )
        elif self.breakpoint_threshold_type == "interquartile":
            # [수정] Mypy 타입 추론 지원을 위해 리스트 형태의 인덱스 전달 시 반환값을 명시적으로 처리
            percentiles = cast(np.ndarray, np.percentile(dist_array, [25, 75]))
            q1, q3 = float(percentiles[0]), float(percentiles[1])
            iqr = q3 - q1
            threshold = float(
                np.mean(dist_array) + self.breakpoint_threshold_value * iqr
            )
        elif self.breakpoint_threshold_type == "gradient":
            # 거리가 급격히 변하는 지점 감지
            threshold = float(
                np.percentile(np.gradient(dist_array), self.breakpoint_threshold_value)
            )
        else:
            threshold = float(self.similarity_threshold)  # 폴백

        # 1. 유사도 기반 분기점 추출
        breakpoints = (np.where(dist_array > threshold)[0] + 1).tolist()

        # 2. [고도화] 마크다운 헤더 감지 기반 강제 분할
        if sentences:
            header_bps = []
            for i, s in enumerate(sentences):
                # 문장의 시작이 헤더 패턴인 경우 (첫 문장 제외)
                if i > 0 and self.HEADER_SIMPLE.match(s["text"].strip()):
                    header_bps.append(i)

            if header_bps:
                logger.info(
                    f"[Chunker] {len(header_bps)}개의 마크다운 헤더를 감지하여 강제 분할 지점으로 설정합니다."
                )
                breakpoints = sorted(set(breakpoints + header_bps))

        # 3. [안전 장치] 너무 긴 청크 방지
        if sentences:
            current_len = 0
            safety_bps = []
            for i, s in enumerate(sentences):
                current_len += len(s["text"])
                if current_len > (self.max_chunk_size * 0.9):
                    # 이미 breakpoints에 이 근처 지점이 있는지 확인
                    if not any(abs(bp - (i + 1)) <= 1 for bp in breakpoints):
                        safety_bps.append(i + 1)
                    current_len = 0
            if safety_bps:
                breakpoints = sorted(set(breakpoints + safety_bps))

        return breakpoints

    def _optimize_chunk_sizes(self, chunks: list[dict]) -> list[dict]:
        """
        생성된 청크들의 크기를 검사하여 병합하며, 벡터도 가중 평균으로 계산합니다.
        [수정] 섹션(Header)이 다르면 절대로 병합하지 않습니다.
        """
        if not chunks:
            return chunks

        optimized = []
        current_chunk = None

        for chunk in chunks:
            if current_chunk is None:
                current_chunk = chunk
                continue

            # 예상 병합 크기 (공백 포함)
            merged_text = current_chunk["text"] + " " + chunk["text"]
            merged_len = len(merged_text)

            # [핵심] 섹션이 다르면 병합 절대 금지
            is_different_section = current_chunk.get("current_section") != chunk.get(
                "current_section"
            )

            # 강제 분할 지점(오프셋 단절 또는 플래그) 확인
            is_at_hard_boundary = current_chunk.get("end") != chunk.get(
                "start"
            ) or current_chunk.get("is_hard_split", False)

            # [수정] 지능적 병합 조건 강화
            should_merge = False

            if not is_different_section and not is_at_hard_boundary:
                # 1. 크기가 극도로 작을 때만 병합 시도
                if len(current_chunk["text"]) < self.min_chunk_size:
                    should_merge = merged_len <= self.max_chunk_size

                # 2. 유사도 기반 지능적 병합
                if not should_merge and merged_len <= self.max_chunk_size:
                    sim = float(np.dot(current_chunk["vector"], chunk["vector"]))
                    if sim > (ChunkingConstants.SIMILARITY_MERGE_THRESHOLD / 100.0):
                        should_merge = True

            if should_merge:
                # 벡터 병합: 각 청크의 길이를 고려한 가중 평균
                len_a = len(current_chunk["text"])
                len_b = len(chunk["text"])
                total_len = len_a + len_b

                if total_len > 0:
                    merged_vector = (
                        current_chunk["vector"] * len_a + chunk["vector"] * len_b
                    ) / total_len
                    norm = np.linalg.norm(merged_vector)
                    if norm > 0:
                        merged_vector /= norm
                else:
                    merged_vector = current_chunk["vector"]

                current_chunk["text"] = merged_text
                current_chunk["end"] = chunk["end"]
                current_chunk["vector"] = merged_vector
                current_chunk["is_hard_split"] = chunk.get("is_hard_split", False)
            else:
                optimized.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            optimized.append(current_chunk)

        return optimized

    def _clean_section_title(self, raw_title: str) -> str:
        """섹션 제목에서 마크다운 기호 및 불필요한 본문 텍스트를 제거합니다."""
        # 1. 마크다운 헤더 기호 및 앞뒤 공백 제거
        clean_title = raw_title.lstrip("# ").strip()

        # 2. 논문의 주요 섹션 키워드 체크 (상수 활용)
        upper_title = clean_title.upper()
        for kw in STANDARD_SECTION_KEYWORDS:
            if upper_title.startswith(kw):
                return kw

        # 3. 너무 긴 경우(100자 이상) 첫 번째 끊김 지점에서 자르기 (제목 보존을 위해 확장)
        if len(clean_title) > 100:
            parts = re.split(r"[\n\r\.\:]", clean_title)
            if parts:
                clean_title = parts[0].strip()

        # 4. 특수문자 제거 및 최종 길이 제한
        clean_title = re.sub(r"[#\*_]", "", clean_title)
        return clean_title[:150]  # 제목 복원을 위해 길이 상향

    def _prune_duplicates(
        self, chunks: list[dict], threshold: float = 0.98
    ) -> list[dict]:
        """임베딩 벡터 기반으로 유사도가 높은 중복 청크를 제거합니다."""
        if not chunks:
            return []

        pruned = [chunks[0]]
        for i in range(1, len(chunks)):
            current_vec = chunks[i]["vector"]
            is_dup = False

            # 최근 3개의 청크와 비교하여 중복 여부 확인 (대량 문서 내 반복 구간 처리)
            for prev in pruned[-3:]:
                sim = float(np.dot(current_vec, prev["vector"]))
                if sim > threshold:
                    is_dup = True
                    break

            if not is_dup:
                pruned.append(chunks[i])
            else:
                logger.debug(f"[Chunker] 중복 청크 제거됨 (유사도 > {threshold})")

        return pruned

    async def split_text(self, text: str) -> list[dict]:
        """
        텍스트를 의미론적으로 분할합니다 (Buffer-based context window 적용).
        [고도화] 마크다운 헤더 뿐만 아니라 대문자 섹션명도 감지하여 구조를 파악합니다.
        """
        if not text or not text.strip():
            return []

        # 1. 문장 분할 (오프셋 포함)
        raw_sentences = self._split_sentences(text)
        if not raw_sentences:
            return []

        # [최적화] 너무 짧은 문장 병합 (오프셋 유지)
        min_merge_len = ChunkingConstants.MIN_MERGE_LEN.value
        # [수정] 헤더 감지 정규식 강화: # 패턴 또는 대문자 시작 섹션 (1 INTRODUCTION 등)
        # [최적화] 클래스 상수 HEADER_PATTERN 사용 (재컴파일 방지)

        sentences = []
        if raw_sentences:
            # [최적화] 병합 전 1회만 헤더 여부를 계산하여 캐싱
            for raw_s in raw_sentences:
                raw_s["is_header"] = bool(
                    self.HEADER_PATTERN.match(raw_s["text"].strip())
                )

            current_s = raw_sentences[0]
            for s in raw_sentences[1:]:
                # [수정] 현재 문장이 헤더거나, 다음 문장이 헤더인 경우 병합 제외 (Clean Section Preservation)
                # [최적화] 미리 캐싱된 is_header 값 재사용 (정규식 중복 호출 제거)
                is_curr_header = current_s.get("is_header", False)
                is_next_header = s.get("is_header", False)

                can_merge = (
                    not is_curr_header  # ✅ 현재 문장이 헤더면 절대 합치지 않음
                    and not is_next_header  # 다음 문장이 헤더면 합치지 않음
                    and not current_s.get("is_hard_split", False)
                    and (len(s["text"]) < min_merge_len)
                    and (len(current_s["text"]) + len(s["text"]) < 1000)
                )

                if can_merge:
                    current_s["text"] += " " + s["text"]
                    current_s["end"] = s["end"]
                    current_s["is_hard_split"] = s.get("is_hard_split", False)
                    # 병합된 문장은 헤더가 아님 (텍스트가 변경되었으므로)
                    current_s["is_header"] = False
                else:
                    sentences.append(current_s)
                    current_s = s
            sentences.append(current_s)

        # [최적화] 헤더 위치 미리 캐싱 (O(1) lookup용) — 이미 위에서 계산 완료

        header_indices = {
            i for i, s in enumerate(sentences) if s.get("is_header", False)
        }

        if len(sentences) <= self.buffer_size:
            # 벡터가 없으므로 계산 필요
            if sentences:
                sentence_texts = [s["text"] for s in sentences]
                embeddings = await self._get_embeddings(sentence_texts)
                if embeddings.shape[0] != len(sentence_texts):
                    raise ValueError(
                        f"[Chunker] 임베딩 행 수({embeddings.shape[0]})가 "
                        f"문장 수({len(sentence_texts)})와 일치하지 않습니다."
                    )
                for s, v in zip(sentences, embeddings, strict=True):
                    s["vector"] = v
            return sentences

        # 2. 개별 문장 임베딩 생성 (캐싱 활용)
        indiv_embeddings = await self._get_embeddings([s["text"] for s in sentences])
        if indiv_embeddings.shape[0] != len(sentences):
            raise ValueError(
                f"[Chunker] 임베딩 행 수({indiv_embeddings.shape[0]})가 "
                f"문장 수({len(sentences)})와 일치하지 않습니다."
            )
        for s, v in zip(sentences, indiv_embeddings, strict=True):
            s["vector"] = v

        # 3. Buffer 기반 Combined Embeddings 생성
        combined_embeddings = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            window_vectors = indiv_embeddings[start:end]
            combined_vec = np.mean(window_vectors, axis=0)
            norm = np.linalg.norm(combined_vec)
            if norm > 1e-9:
                combined_vec /= norm
            combined_embeddings.append(combined_vec)

        combined_embeddings_arr = np.array(combined_embeddings)

        # 4. 거리 계산
        distances = []
        for i in range(len(combined_embeddings_arr) - 1):
            similarity = np.dot(
                combined_embeddings_arr[i], combined_embeddings_arr[i + 1]
            )
            distances.append(1.0 - float(similarity))

        # 5. 분기점 탐색 (헤더 인식 로직 포함됨)
        breakpoints = self._find_breakpoints(distances, sentences=sentences)

        # 6. 그룹화 및 헤더 컨텍스트(Section) 추적
        chunks = []
        start_idx = 0
        all_bps = breakpoints + [len(sentences)]

        current_header = "Front Matter"  # [수정] 기본값

        # [추가] 헤더 병합용 상태
        pending_header = ""
        is_first_header = True

        for i, bp in enumerate(all_bps):
            group_start = start_idx
            group_end = bp
            if group_start >= group_end:
                continue

            # [고도화] 새로운 헤더로 시작하는지 확인 (오버랩 방지용)
            is_new_header_start = sentences[group_start].get("is_header", False)

            # Overlap 적용: 헤더로 시작하는 경우 오버랩 생략 (Clean Section Start)
            actual_start = group_start
            if i > 0 and not is_new_header_start:
                actual_start = max(0, group_start - self.chunk_overlap)

            # [핵심] 현재 청크의 헤더 섹션 결정 (미리 캐싱된 header_indices 사용)
            for s_idx in range(start_idx, group_end):
                if s_idx not in header_indices:  # O(1) set lookup
                    continue
                s = sentences[s_idx]
                text_strip = s["text"].strip()
                new_h = self._clean_section_title(text_strip)

                # [지능형 병합] 전치사나 'OF' 등으로 끝나면 다음 헤더와 합침
                incomplete_markers = ["OF", "AND", "WITH", "IN", "FOR", "THE", "A"]
                if (
                    any(new_h.upper().endswith(w) for w in incomplete_markers)
                    and len(new_h) < 100
                ):
                    pending_header = new_h + " "
                    continue

                if pending_header:
                    new_h = (pending_header + new_h).strip()
                    pending_header = ""

                # 첫 번째 거대한 헤더는 제목으로 처리
                if is_first_header and len(new_h) > 10:
                    current_header = f"TITLE: {new_h}"
                    is_first_header = False
                else:
                    current_header = new_h

            merged_text = " ".join(
                [s["text"] for s in sentences[actual_start:group_end]]
            )
            chunk_vector = np.mean(indiv_embeddings[actual_start:group_end], axis=0)

            chunks.append(
                {
                    "text": merged_text,
                    "start": sentences[group_start]["start"],
                    "end": sentences[group_end - 1]["end"],
                    "vector": chunk_vector,
                    "is_hard_split": sentences[group_end - 1].get(
                        "is_hard_split", False
                    ),
                    "current_section": current_header,
                }
            )
            start_idx = bp

        # 7. 크기 최적화 및 중복 제거
        optimized_chunks = self._optimize_chunk_sizes(chunks)
        return self._prune_duplicates(optimized_chunks)

    def _extract_metadata_for_chunk(
        self, chunk: dict[str, Any], doc_ranges: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        청크에 대한 메타데이터를 추출하고 병합합니다.
        """
        c_start = chunk["start"]
        c_end = chunk["end"]

        overlapping_pages = []
        merged_metadata: dict[str, Any] = {}

        for doc_range in doc_ranges:
            # 범위 겹침 확인: [start1, end1] 과 [start2, end2]
            if max(c_start, doc_range["start"]) < min(c_end, doc_range["end"]):
                # [수정] Mypy 타입 추론 강화를 위해 cast 사용
                metadata = cast(dict[str, Any], doc_range["metadata"])
                page = metadata.get("page")
                if page and page not in overlapping_pages:
                    overlapping_pages.append(page)

                # 기본 메타데이터는 첫 번째 겹치는 문서에서 가져오되, 페이지 정보는 업데이트
                if not merged_metadata:
                    merged_metadata = metadata.copy()

        if overlapping_pages:
            merged_metadata["pages"] = sorted(overlapping_pages)
            # 하위 호환성을 위해 단일 page 필드는 첫 페이지로 유지
            merged_metadata["page"] = overlapping_pages[0]
            # 다중 페이지 여부 표시
            merged_metadata["is_cross_page"] = len(overlapping_pages) > 1

        # [핵심 수정] 추출된 섹션 정보를 메타데이터에 주입
        merged_metadata["current_section"] = chunk.get("current_section", "일반 본문")

        merged_metadata["start_index"] = c_start
        merged_metadata["end_index"] = c_end

        return merged_metadata

    async def split_documents(
        self, docs: list["Document"]
    ) -> tuple[list["Document"], list[np.ndarray]]:
        """
        LangChain Document 객체 리스트를 받아 의미론적 분할을 수행합니다.
        문서들을 통합하여 문맥을 유지하되, 오프셋 매핑을 통해
        각 청크의 원본 메타데이터(페이지 번호 등)를 정확히 보존합니다.
        각 청크에 대해 이미 계산된 벡터를 함께 반환하여 재계산을 방지합니다.
        """

        if not docs:
            logger.warning("split_documents: 입력 문서 리스트가 비어있습니다.")
            return [], []

        # 1. 문서 통합 및 오프셋 매핑 구축 (join()을 사용하여 효율성 개선)
        text_parts = []
        doc_ranges = []
        current_offset = 0

        for doc in docs:
            content = doc.page_content
            if not content:
                continue

            # 문서 사이 공백 추가 (첫 문서 제외)
            if current_offset > 0:
                text_parts.append(" ")
                current_offset += 1

            start = current_offset
            text_parts.append(content)
            end = current_offset + len(content)

            doc_ranges.append({"start": start, "end": end, "metadata": doc.metadata})
            current_offset = end

        full_text = "".join(text_parts)

        if not full_text.strip():
            logger.warning("split_documents: 병합된 텍스트가 비어있습니다.")
            return [], []

        # 2. 청킹 수행 (오프셋 및 벡터 정보 포함)
        chunk_dicts = await self.split_text(full_text)

        # 3. 메타데이터 역매핑 및 문서 객체 생성 (다중 페이지 지원)
        final_docs = []
        final_vectors = []

        def _extract_metadata_batch(
            chunk_dicts: list[dict], doc_ranges: list[dict]
        ) -> list[dict]:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                return list(
                    executor.map(
                        lambda chunk: self._extract_metadata_for_chunk(
                            chunk, doc_ranges
                        ),
                        chunk_dicts,
                    )
                )

        results = await asyncio.to_thread(
            _extract_metadata_batch, chunk_dicts, doc_ranges
        )

        for chunk, merged_metadata in zip(chunk_dicts, results, strict=False):
            final_docs.append(
                Document(page_content=chunk["text"], metadata=merged_metadata)
            )
            final_vectors.append(chunk["vector"])

        logger.info(
            f"의미론적 문서 분할 완료: {len(docs)}개 원본 문서 -> {len(final_docs)}개 청크 생성 (벡터 포함)"
        )
        return final_docs, final_vectors
