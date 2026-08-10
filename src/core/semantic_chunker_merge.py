"""
청크 병합/중복제거/그룹화 관심사 모듈 — ``SemanticChunkerMergeMixin``.
청크 크기 최적화(``_optimize_chunk_sizes``)·중복 제거(``_prune_duplicates``)·
섹션 제목 정제(``_clean_section_title``)·그룹화(``_group_sentences_into_chunks``).
[R2-10] ``semantic_chunker.py`` 모노리스에서 관심사별로 분리한 모듈입니다.
"""

import logging
import re

import numpy as np

from common.constants import ChunkingConstants

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


class SemanticChunkerMergeMixin:
    """
    병합/중복제거/그룹화 관심사 믹스인.

    아래 속성들은 ``EmbeddingBasedSemanticChunker.__init__``에서 설정됩니다.
    """

    min_chunk_size: int
    max_chunk_size: int
    chunk_overlap: int

    def _optimize_chunk_sizes(self, chunks: list[dict]) -> list[dict]:
        """청크 크기를 검사해 병합하며, 벡터는 가중 평균으로 계산합니다.
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

    def _group_sentences_into_chunks(
        self,
        sentences: list[dict],
        header_indices: set[int],
        indiv_embeddings: np.ndarray,
        breakpoints: list[int],
    ) -> list[dict]:
        """분기점 기준으로 문장을 그룹화하고 헤더 컨텍스트(Section)를 추적해 청크를 조립합니다."""
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
            # [R2-02] 청크 벡터 단위노름 — 병합/중복제거의 dot product가 코사인 유사도
            chunk_norm = float(np.linalg.norm(chunk_vector))
            if chunk_norm > 1e-9:
                chunk_vector = chunk_vector / chunk_norm

            # [R2-05] start는 오버랩 포함 실제 텍스트 범위(actual_start) 기준으로 산출.
            # merged_text·벡터가 이미 actual_start를 사용하므로, group_start 기준으로
            # 산출하면 오버랩 문장 구간의 좌표·페이지 메타데이터가 청크 텍스트와 어긋납니다.
            chunks.append(
                {
                    "text": merged_text,
                    "start": sentences[actual_start]["start"],
                    "end": sentences[group_end - 1]["end"],
                    "vector": chunk_vector,
                    "is_hard_split": sentences[group_end - 1].get(
                        "is_hard_split", False
                    ),
                    "current_section": current_header,
                }
            )
            start_idx = bp

        return chunks
