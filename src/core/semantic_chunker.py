"""
임베딩 기반 의미론적 텍스트 분할기를 구현합니다.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로
일관성 있는 청크를 생성합니다.
"""

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from common.constants import ChunkingConstants
from services.optimization.caching_optimizer import CacheManager, get_cache_manager

# 순환 참조 방지 및 타입 힌트용
if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class EmbeddingBasedSemanticChunker:
    """
    임베딩 기반 의미론적 텍스트 분할기.

    문장 단위로 분할 후, 임베딩 유사도 기반으로 의미 경계를 탐지하여
    일관성 있는 청크를 생성합니다.
    """

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
        similarity_threshold: float = 0.5,
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

        # [최적화] 모델 식별을 위한 이름 추출 (HuggingFaceEmbeddings 등 지원)
        self.model_name = getattr(embedder, "model_name", "default_model")

        # ✅ 정규식 사전 컴파일 (성능 최적화)
        self._sentence_pattern = re.compile(self.sentence_split_regex)

    def _split_sentences(self, text: str) -> list[dict]:
        """
        문장 단위로 텍스트를 분할하고 오프셋 정보를 함께 반환합니다.
        """
        # ... (기존 정규식 분할 로직)
        parts = list(self._sentence_pattern.finditer(text))
        raw_segments: list[dict[str, Any]] = []

        last_pos = 0
        for match in parts:
            sep_end = match.end()
            segment_text = text[last_pos:sep_end]
            if segment_text.strip():
                raw_segments.append(
                    {"text": segment_text, "start": last_pos, "end": sep_end}
                )
            last_pos = sep_end

        if last_pos < len(text):
            remaining_text = text[last_pos:]
            if remaining_text.strip():
                raw_segments.append(
                    {"text": remaining_text, "start": last_pos, "end": len(text)}
                )

        # 2. 너무 긴 세그먼트 강제 분할 (OOM 방지 및 병합 방지 플래그 추가)
        # [최적화] max_chunk_size의 1.5배까지 여유를 주어 기계적 분할 빈도를 줄임
        hard_split_limit: int = ChunkingConstants.MAX_HARD_SPLIT_LEN.value
        if self.max_chunk_size > 0:
            hard_split_limit = min(
                ChunkingConstants.MAX_HARD_SPLIT_LEN.value,
                int(self.max_chunk_size * 1.5),
            )

        final_sentences: list[dict[str, Any]] = []
        for seg in raw_segments:
            seg_text = str(seg["text"])
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])

            if len(seg_text) <= hard_split_limit:
                self._add_cleaned_sentence(
                    final_sentences, seg_text, seg_start, seg_end
                )
            else:
                # [최적화] 강제 분할은 정상적인 복구 프로세스이므로 레벨을 INFO로 낮춤
                logger.info(
                    f"[Chunker] 긴 세그먼트 발견 ({len(seg_text)}자). 설정된 한계치({hard_split_limit})에 맞춰 강제 분할을 수행합니다."
                )

                curr_pos = 0
                while curr_pos < len(seg_text):
                    sub_len: int = int(hard_split_limit)
                    if curr_pos + sub_len < len(seg_text):
                        last_space = seg_text.rfind(" ", curr_pos, curr_pos + sub_len)
                        if last_space != -1 and last_space > curr_pos + (sub_len // 2):
                            sub_len = int(last_space - curr_pos + 1)

                    sub_text = seg_text[curr_pos : curr_pos + sub_len]
                    # [핵심] 강제 분할된 마지막 문장에 플래그 추가 (다음 문장과 병합 방지)
                    is_last_sub = curr_pos + sub_len >= len(seg_text)
                    self._add_cleaned_sentence(
                        final_sentences,
                        sub_text,
                        seg_start + curr_pos,
                        seg_start + curr_pos + len(sub_text),
                        is_hard_split=not is_last_sub,  # 중간 쪼개짐 지점
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
        self, texts: list[str], normalize: bool = True
    ) -> np.ndarray:
        """
        텍스트 리스트의 임베딩을 생성합니다 (배치 처리 및 캐싱 강화).
        """
        if not texts:
            return np.array([]).reshape(0, 0)

        all_results = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        # 1. 정제 및 캐시 확인
        for i, text in enumerate(texts):
            norm_text = " ".join(text.split())
            cache_key = f"emb:{self.model_name}:{hashlib.sha256(norm_text.encode()).hexdigest()[:16]}"

            cached_vec = await self.cache_manager.get(cache_key)
            if cached_vec is not None:
                all_results[i] = np.array(cached_vec, dtype="float32")
            else:
                missing_indices.append(i)
                missing_texts.append(norm_text)

        # 2. 누락분 배치 임베딩 수행
        if missing_texts:
            logger.debug(
                f"[Chunker] {len(missing_texts)}개 문장 신규 임베딩 생성 중 (Batch Size: {self.batch_size})..."
            )

            for b_idx in range(0, len(missing_texts), self.batch_size):
                batch = missing_texts[b_idx : b_idx + self.batch_size]
                batch_indices = missing_indices[b_idx : b_idx + self.batch_size]

                try:
                    # [최적화] embed_documents 호출
                    batch_vecs = self.embedder.embed_documents(batch)

                    for idx, vec in zip(batch_indices, batch_vecs, strict=False):
                        vec_np = np.array(vec, dtype="float32")
                        all_results[idx] = vec_np

                        # 캐시 저장
                        norm_text = texts[idx]
                        cache_key = f"emb:{self.model_name}:{hashlib.sha256(norm_text.encode()).hexdigest()[:16]}"
                        await self.cache_manager.set(cache_key, vec_np.tolist())

                except Exception as e:
                    logger.error(f"[MODEL] [BATCH] 임베딩 생성 중 오류 발생: {e}")
                    raise

        # 3. 결과 행렬 조립 및 정규화
        embeddings_matrix = np.stack(all_results).astype("float32")
        if normalize:
            norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
            embeddings_matrix = np.divide(
                embeddings_matrix,
                norms,
                out=np.zeros_like(embeddings_matrix),
                where=norms > 1e-9,
            )
        return embeddings_matrix

    def _calculate_similarities(self, normalized_embeddings: np.ndarray) -> list[float]:
        """
        인접 문장 간의 코사인 유사도를 계산합니다.
        [최적화] 이미 정규화된 벡터를 사용하여 내적(Dot Product)만 수행합니다.
        einsum을 사용하여 루프 없이 벡터화된 연산을 수행합니다.
        """
        if len(normalized_embeddings) < 2:
            return []

        # [최적화] einsum을 사용하여 인접 행 간의 내적을 한 번에 계산
        # (n-1, d) 와 (n-1, d) 의 각 행끼리 곱하고 합산
        similarities = np.einsum(
            "ij,ij->i", normalized_embeddings[:-1], normalized_embeddings[1:]
        )

        return similarities.tolist()

    def _find_breakpoints(
        self, distances: list[float], sentences: list[dict] | None = None
    ) -> list[int]:
        """
        유사도 거리(1-cos_sim) 분포를 분석하여 분할 지점을 찾습니다.
        LangChain의 최신 구현 전략(Standard Deviation, Interquartile, Gradient)을 도입합니다.
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

        # 임계값을 넘는 인덱스 추출 (거리가 클수록 의미가 다름)
        breakpoints = (np.where(dist_array > threshold)[0] + 1).tolist()

        # [안전 장치] 너무 긴 청크 방지
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
        [최적화] 크기뿐만 아니라 의미적 유사도를 고려하여 지능적으로 병합합니다.
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

            # [수정] 지능적 병합 조건 강화
            should_merge = False

            # 강제 분할 지점(오프셋 단절 또는 플래그) 확인
            is_at_hard_boundary = current_chunk.get("end") != chunk.get(
                "start"
            ) or current_chunk.get("is_hard_split", False)

            # 1. 크기가 극도로 작을 때만 병합 시도 (단절된 경우 제외)
            if (
                not is_at_hard_boundary
                and len(current_chunk["text"]) < self.min_chunk_size
            ):
                should_merge = merged_len <= self.max_chunk_size

            # 2. 유사도 기반 지능적 병합 (단절된 경우 제외)
            if (
                not should_merge
                and not is_at_hard_boundary
                and merged_len <= self.max_chunk_size
            ):
                sim = float(np.dot(current_chunk["vector"], chunk["vector"]))
                if sim > (ChunkingConstants.SIMILARITY_MERGE_THRESHOLD / 100.0):
                    should_merge = True

            # 3. [추가] 강제 분할 지점 여부 확인 (예: 텍스트가 너무 길어 쪼개진 경우)
            # 여기서는 단순 텍스트 비교 대신 max_chunk_size를 기준으로 방어적 분할 유지
            if merged_len > self.max_chunk_size:
                should_merge = False

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

    async def split_text(self, text: str) -> list[dict]:
        """
        텍스트를 의미론적으로 분할합니다 (Buffer-based context window 적용).
        """
        if not text or not text.strip():
            return []

        # 1. 문장 분할 (오프셋 포함)
        raw_sentences = self._split_sentences(text)
        if not raw_sentences:
            return []

        # [최적화] 너무 짧은 문장 병합 (오프셋 유지)
        min_merge_len = ChunkingConstants.MIN_MERGE_LEN.value

        sentences = []
        if raw_sentences:
            current_s = raw_sentences[0]
            for s in raw_sentences[1:]:
                # [수정] 강제 분할 플래그 확인 및 병합 조건 강화
                can_merge = (
                    not current_s.get("is_hard_split", False)
                    and (len(s["text"]) < min_merge_len)
                    and (len(current_s["text"]) + len(s["text"]) < 1000)
                )

                if can_merge:
                    current_s["text"] += " " + s["text"]
                    current_s["end"] = s["end"]
                    current_s["is_hard_split"] = s.get("is_hard_split", False)
                else:
                    sentences.append(current_s)
                    current_s = s
            sentences.append(current_s)

        if len(sentences) <= self.buffer_size:
            # 벡터가 없으므로 계산 필요
            if sentences:
                sentence_texts = [s["text"] for s in sentences]
                embeddings = await self._get_embeddings(sentence_texts)
                for s, v in zip(sentences, embeddings, strict=False):
                    s["vector"] = v
            return sentences

        # 2. 개별 문장 임베딩 생성 (캐싱 활용)
        # [최적화] 중복 호출 방지를 위해 개별 문장 임베딩을 먼저 구하고 이를 조합하여 문맥 임베딩 생성
        indiv_embeddings = await self._get_embeddings([s["text"] for s in sentences])
        for s, v in zip(sentences, indiv_embeddings, strict=False):
            s["vector"] = v

        if len(sentences) <= self.buffer_size:
            return sentences

        # 3. Buffer 기반 Combined Embeddings 생성 (임베딩 호출 없이 벡터 연산으로 처리)
        # [핵심 개선] 인접한 단일 문장이 아니라 앞뒤 문맥을 합쳐서 비교 (Gregory Kamradt 방식)
        # 텍스트를 합쳐서 다시 임베딩하는 대신, 개별 임베딩의 평균을 사용하여 획기적 속도 향상
        combined_embeddings = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)

            # 윈도우 내 문장들의 벡터 평균 계산
            window_vectors = indiv_embeddings[start:end]
            combined_vec = np.mean(window_vectors, axis=0)

            # 정규화
            norm = np.linalg.norm(combined_vec)
            if norm > 1e-9:
                combined_vec /= norm
            combined_embeddings.append(combined_vec)

        combined_embeddings = np.array(combined_embeddings)

        # 4. 거리 계산 (1 - 코사인 유사도)
        # combined_embeddings[i]와 combined_embeddings[i+1] 간의 거리 측정
        distances = []
        for i in range(len(combined_embeddings) - 1):
            similarity = np.dot(combined_embeddings[i], combined_embeddings[i + 1])
            distances.append(1.0 - float(similarity))

        # 5. 분기점 탐색
        breakpoints = self._find_breakpoints(distances, sentences=sentences)

        # 6. 1차 그룹화 (오프셋 및 벡터 포함, [추가] Overlap 반영)
        chunks = []
        start_idx = 0
        all_bps = breakpoints + [len(sentences)]

        for i, bp in enumerate(all_bps):
            # 현재 그룹의 인덱스 범위 계산
            group_start = start_idx
            group_end = bp

            if group_start >= group_end:
                continue

            # [핵심] Overlap 적용: 이전 그룹의 마지막 N개 문장을 포함시킴
            actual_start = (
                max(0, group_start - self.chunk_overlap) if i > 0 else group_start
            )

            group_sentences = sentences[actual_start:group_end]
            group_embeddings = indiv_embeddings[actual_start:group_end]

            # 병합된 텍스트 및 범위 계산
            merged_text = " ".join([s["text"] for s in group_sentences])
            c_start = group_sentences[0]["start"]
            c_end = group_sentences[-1]["end"]

            chunk_vector = np.mean(group_embeddings, axis=0)

            chunks.append(
                {
                    "text": merged_text,
                    "start": c_start,
                    "end": c_end,
                    "vector": chunk_vector,
                    "is_hard_split": group_sentences[-1].get("is_hard_split", False),
                }
            )
            start_idx = bp

        # 6. 크기 최적화 (오프셋 및 벡터 인식)
        return self._optimize_chunk_sizes(chunks)

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

        for chunk in chunk_dicts:
            c_start = chunk["start"]
            c_end = chunk["end"]
            c_text = chunk["text"]
            c_vector = chunk["vector"]

            # [개선] 청크 범위에 걸쳐 있는 모든 원본 문서(페이지) 찾기
            overlapping_pages = []
            merged_metadata: dict[str, Any] = {}

            for doc_range in doc_ranges:
                # 범위 겹침 확인: [start1, end1] 과 [start2, end2]
                if max(c_start, doc_range["start"]) < min(c_end, doc_range["end"]):
                    page = doc_range["metadata"].get("page")
                    if page and page not in overlapping_pages:
                        overlapping_pages.append(page)

                    # 기본 메타데이터는 첫 번째 겹치는 문서에서 가져오되, 페이지 정보는 업데이트
                    if not merged_metadata:
                        merged_metadata = doc_range["metadata"].copy()

            if overlapping_pages:
                merged_metadata["pages"] = sorted(overlapping_pages)
                # 하위 호환성을 위해 단일 page 필드는 첫 페이지로 유지
                merged_metadata["page"] = overlapping_pages[0]
                # 다중 페이지 여부 표시
                merged_metadata["is_cross_page"] = len(overlapping_pages) > 1

            final_docs.append(Document(page_content=c_text, metadata=merged_metadata))
            final_vectors.append(c_vector)

        logger.info(
            f"의미론적 문서 분할 완료: {len(docs)}개 원본 문서 -> {len(final_docs)}개 청크 생성 (벡터 포함)"
        )
        return final_docs, final_vectors
