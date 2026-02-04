"""
임베딩 기반 의미론적 텍스트 분할기를 구현합니다.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로
일관성 있는 청크를 생성합니다.
"""

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np
from langchain_core.documents import Document

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
        embedder: Any,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_value: float = 95.0,
        sentence_split_regex: str = r"([.!?]\s+)",  # 기본값을 split 친화적으로 변경 (캡처 그룹 포함 권장)
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        similarity_threshold: float = 0.5,
        batch_size: int = 64,
    ):
        """
        의미론적 청킹 분할기를 초기화합니다.

        Args:
            embedder: 텍스트 임베딩을 생성하는 모델 (HuggingFaceEmbeddings 등)
            breakpoint_threshold_type: 'percentile', 'standard_deviation', 'interquartile', 'gradient'
            breakpoint_threshold_value: threshold 값 (백분위수 또는 표준편차 배수)
            sentence_split_regex: 문장 분할 정규식. re.split을 사용하므로 구분자를 포함하려면 캡처 그룹()을 사용하세요.
            min_chunk_size: 최소 청크 크기 (문자)
            max_chunk_size: 최대 청크 크기 (문자)
            similarity_threshold: 유사도 threshold (0-1)
            batch_size: 임베딩 생성 시 배치 크기
        """
        self.embedder = embedder
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_value = breakpoint_threshold_value
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size

        # ✅ 정규식 사전 컴파일 (성능 최적화)
        self._sentence_pattern = re.compile(self.sentence_split_regex)

    def _split_sentences(self, text: str) -> list[dict]:
        """
        문장 단위로 텍스트를 분할하고 오프셋 정보를 함께 반환합니다.
        [안전성 강화] 구분자가 없는 매우 긴 텍스트의 경우 강제 분할을 수행합니다.
        """
        MAX_HARD_SPLIT_LEN = 1500  # 강제 분할 임계값 (글자 수)

        # 1. 정규식으로 기본 분할
        parts = list(self._sentence_pattern.finditer(text))
        raw_segments: list[dict[str, Any]] = []

        last_pos = 0
        for match in parts:
            sep_start, sep_end = match.span()

            # 구분자 앞쪽 텍스트 + 구분자
            segment_text = text[last_pos:sep_end]
            raw_segments.append(
                {"text": segment_text, "start": last_pos, "end": sep_end}
            )
            last_pos = sep_end

        if last_pos < len(text):
            raw_segments.append(
                {"text": text[last_pos:], "start": last_pos, "end": len(text)}
            )

        # 2. 너무 긴 세그먼트 강제 분할 (OOM 방지)
        final_sentences: list[dict[str, Any]] = []
        for seg in raw_segments:
            seg_text = str(seg["text"])
            seg_start = int(seg["start"])
            seg_end = int(seg["end"])

            if len(seg_text) <= MAX_HARD_SPLIT_LEN:
                # 정상 크기인 경우 정제 후 추가
                self._add_cleaned_sentence(
                    final_sentences, seg_text, seg_start, seg_end
                )
            else:
                # 강제 분할 수행 (공백 기준 또는 글자 수 기준)
                logger.warning(
                    f"[Chunker] 과도하게 긴 세그먼트 발견 ({len(seg_text)}자). 강제 분할을 수행합니다."
                )

                curr_pos = 0
                while curr_pos < len(seg_text):
                    sub_len = MAX_HARD_SPLIT_LEN
                    # 가급적 공백에서 자르기 시도
                    if curr_pos + sub_len < len(seg_text):
                        last_space = seg_text.rfind(" ", curr_pos, curr_pos + sub_len)
                        if last_space != -1 and last_space > curr_pos + (sub_len // 2):
                            sub_len = last_space - curr_pos + 1

                    sub_text = seg_text[curr_pos : curr_pos + sub_len]
                    self._add_cleaned_sentence(
                        final_sentences,
                        sub_text,
                        seg_start + curr_pos,
                        seg_start + curr_pos + len(sub_text),
                    )
                    curr_pos += sub_len
        return final_sentences

    def _add_cleaned_sentence(
        self,
        target_list: list[dict[str, Any]],
        raw_text: str,
        start_offset: int,
        end_offset: int,
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
                {"text": embed_text, "start": final_start, "end": final_end}
            )

    def _get_embeddings(self, texts: list[str], normalize: bool = True) -> np.ndarray:
        """
        텍스트 리스트의 임베딩을 생성합니다.
        [최적화] 중복 제거 및 라이브러리 내부 배치를 활용하여 최고 성능을 냅니다.
        """
        if not texts:
            return np.array([]).reshape(0, 0)

        try:
            # 1. 중복 제거를 통한 연산 최소화
            unique_texts: list[str] = []
            text_to_idx = {}
            mapping = []

            for text in texts:
                norm_text = " ".join(text.lower().split())
                if norm_text not in text_to_idx:
                    text_to_idx[norm_text] = len(unique_texts)
                    unique_texts.append(text)
                mapping.append(text_to_idx[norm_text])

            # 2. 배치 임베딩 (HuggingFaceEmbeddings 내부의 배칭 활용)
            # 수동 배칭(loop)을 제거하여 라이브러리 최적화(AVX/CUDA)가 중단 없이 작동하게 함
            unique_embeddings_list = self.embedder.embed_documents(unique_texts)
            unique_embeddings = np.array(unique_embeddings_list).astype("float32")

            if normalize:
                # [최적화] NumPy 벡터화 연산으로 고속 정규화
                norms = np.linalg.norm(unique_embeddings, axis=1, keepdims=True)
                # 0으로 나누기 방지
                unique_embeddings = np.divide(
                    unique_embeddings,
                    norms,
                    out=np.zeros_like(unique_embeddings),
                    where=norms != 0,
                )

            # 3. 원본 순서로 복원
            return unique_embeddings[mapping]
        except Exception as e:
            logger.error(f"[MODEL] [LOAD] 임베딩 생성 실패 | {e}")
            raise

    def _calculate_similarities(self, normalized_embeddings: np.ndarray) -> list[float]:
        """
        인접 문장 간의 코사인 유사도를 계산합니다.
        [최적화] 이미 정규화된 벡터를 사용하여 내적(Dot Product)만 수행합니다.
        """
        if len(normalized_embeddings) < 2:
            return []

        # 정규화된 벡터 간의 내적은 코사인 유사도와 동일 (einsum으로 메모리 절약)
        similarities = np.einsum(
            "ij,ij->i", normalized_embeddings[:-1], normalized_embeddings[1:]
        )

        return similarities.tolist()

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """
        유사도 분포를 분석하여 분할 지점(breakpoints)을 찾습니다.
        [최적화] np.convolve를 이용한 벡터화 연산으로 로컬 이동 평균을 계산하여 성능을 극대화합니다.
        """
        if not similarities:
            return []

        similarities_array = np.array(similarities)
        n = len(similarities_array)

        # 1. 전역 임계값 계산
        if self.breakpoint_threshold_type == "percentile":
            global_threshold = np.percentile(
                similarities_array, 100 - self.breakpoint_threshold_value
            )
        else:
            global_threshold = self.similarity_threshold

        # 2. [최적화] 로컬 감지 (벡터화된 이동 평균 기반)
        window_size = 3
        breakpoints = []

        # 전역 임계값 미달 지점 (벡터화)
        global_breaks = similarities_array < global_threshold

        # 로컬 급감 지점 계산 (벡터화)
        # np.convolve를 사용하여 이동 평균 계산 (유효한 윈도우만)
        if n >= window_size:
            weights = np.ones(window_size) / window_size
            # 'valid' 모드는 padding 없이 계산하므로 결과 길이가 n - window_size + 1
            local_avgs = np.convolve(similarities_array, weights, mode="valid")

            # 비교를 위해 인덱스 매칭: similarities[i]와 similarities[max(0, i-window_size):i]의 평균 비교
            # 기존 로직: i번째 유사도를 그 직전 window_size개의 평균과 비교함
            # 따라서 local_avgs[0] (0,1,2의 평균)은 similarities[3]과 비교되어야 함

            for i in range(n):
                is_global_break = global_breaks[i]
                is_local_break = False

                if i >= window_size:
                    # i=3일 때 local_avgs[0] (indices 0,1,2의 평균) 사용
                    avg = local_avgs[i - window_size]
                    if similarities_array[i] < avg * 0.8:
                        is_local_break = True

                if is_global_break or is_local_break:
                    breakpoints.append(i + 1)
        else:
            # 데이터가 너무 적으면 전역 임계값만 적용
            breakpoints = [i + 1 for i, b in enumerate(global_breaks) if b]

        logger.debug(
            f"청킹 분석 완료: 전체 분기점 {len(breakpoints)}개 선정 (전역 임계값: {global_threshold:.3f})"
        )

        return breakpoints

    def _optimize_chunk_sizes(self, chunks: list[dict]) -> list[dict]:
        """
        생성된 청크들의 크기를 검사하여 병합하며, 벡터도 가중 평균으로 계산합니다.
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

            # [수정] 의미론적 경계 보존을 위한 조건
            # 현재 청크가 이미 최소 크기(min_chunk_size)를 만족한다면,
            # 굳이 다음 청크(의미적으로 분리된)와 합치지 않고 독립된 청크로 유지합니다.
            should_merge = False
            if len(current_chunk["text"]) < self.min_chunk_size:
                # 현재 청크가 너무 작으면 무조건 병합 시도
                should_merge = merged_len <= self.max_chunk_size
            else:
                # 현재 청크가 이미 최소 크기 이상이라면, 다음 청크와의 병합은 신중해야 함.
                # 여기서는 기본적으로 병합하지 않음 (의미 단위 보존)
                should_merge = False

            if should_merge:
                # 벡터 병합: 각 청크의 길이를 고려한 가중 평균 (단순 평균보다 정확함)
                len_a = len(current_chunk["text"])
                len_b = len(chunk["text"])
                total_len = len_a + len_b

                if total_len > 0:
                    merged_vector = (
                        current_chunk["vector"] * len_a + chunk["vector"] * len_b
                    ) / total_len
                else:
                    merged_vector = current_chunk["vector"]

                current_chunk["text"] = merged_text
                current_chunk["end"] = chunk["end"]  # 끝 위치 업데이트
                current_chunk["vector"] = merged_vector
            else:
                # 현재 청크 저장 후 새로 시작
                optimized.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            optimized.append(current_chunk)

        return optimized

    def split_text(self, text: str) -> list[dict]:
        """
        텍스트를 의미론적으로 분할합니다.
        Returns:
            List[dict]: [{'text': str, 'start': int, 'end': int, 'vector': np.ndarray}, ...]
        """
        if not text or not text.strip():
            return []

        # 1. 문장 분할 (오프셋 포함)
        raw_sentences = self._split_sentences(text)
        if not raw_sentences:
            return []

        # [최적화] 너무 짧은 문장 병합 (오프셋 유지)
        # 30자 미만의 문장은 독립적인 의미를 갖기 어려우므로 앞 문장과 병합
        MIN_MERGE_LEN = 30

        sentences = []
        if raw_sentences:
            current_s = raw_sentences[0]
            for s in raw_sentences[1:]:
                # 현재 문장이 너무 짧거나, 다음 문장이 매우 짧은 경우 병합
                # (단, 합친 길이가 너무 길어지면 안 됨 -> 1000자 제한)
                if (len(s["text"]) < MIN_MERGE_LEN) and (
                    len(current_s["text"]) + len(s["text"]) < 1000
                ):
                    # 병합
                    current_s["text"] += " " + s["text"]
                    current_s["end"] = s["end"]  # 범위 확장
                else:
                    sentences.append(current_s)
                    current_s = s
            sentences.append(current_s)

        if len(sentences) <= 1:
            # 벡터가 없으므로 계산 필요
            if sentences:
                sentence_texts = [s["text"] for s in sentences]
                embeddings = self._get_embeddings(sentence_texts)
                for s, v in zip(sentences, embeddings, strict=False):
                    s["vector"] = v
            return sentences

        # 2. 임베딩 및 유사도 계산 (텍스트만 추출해서 사용)
        sentence_texts = [s["text"] for s in sentences]
        embeddings = self._get_embeddings(sentence_texts)
        similarities = self._calculate_similarities(embeddings)

        # 3. 분기점 탐색
        breakpoints = self._find_breakpoints(similarities)

        # 4. 1차 그룹화 (오프셋 및 벡터 포함)
        chunks = []
        start_idx = 0
        for bp in breakpoints:
            group_sentences = sentences[start_idx:bp]
            group_embeddings = embeddings[start_idx:bp]  # 해당 범위의 벡터들

            if not group_sentences:
                continue

            # 그룹 내 첫 문장의 start와 마지막 문장의 end를 가짐
            merged_text = " ".join([s["text"] for s in group_sentences])
            group_start = group_sentences[0]["start"]
            group_end = group_sentences[-1]["end"]

            # [최적화] 문장 벡터들의 평균을 청크 벡터로 사용
            chunk_vector = np.mean(group_embeddings, axis=0)

            chunks.append(
                {
                    "text": merged_text,
                    "start": group_start,
                    "end": group_end,
                    "vector": chunk_vector,
                }
            )
            start_idx = bp

        # 마지막 그룹
        last_group_sentences = sentences[start_idx:]
        last_group_embeddings = embeddings[start_idx:]
        if last_group_sentences:
            merged_text = " ".join([s["text"] for s in last_group_sentences])
            group_start = last_group_sentences[0]["start"]
            group_end = last_group_sentences[-1]["end"]
            chunk_vector = np.mean(last_group_embeddings, axis=0)

            chunks.append(
                {
                    "text": merged_text,
                    "start": group_start,
                    "end": group_end,
                    "vector": chunk_vector,
                }
            )

        # 5. 크기 최적화 (오프셋 및 벡터 인식)
        return self._optimize_chunk_sizes(chunks)

    def split_documents(
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
        chunk_dicts = self.split_text(full_text)

        # 3. 메타데이터 역매핑 및 문서 객체 생성 (이진 탐색 최적화)
        import bisect

        final_docs = []
        final_vectors = []

        # 탐색을 위한 시작 오프셋 리스트 생성
        doc_starts = [r["start"] for r in doc_ranges]

        for chunk in chunk_dicts:
            c_start = chunk["start"]
            c_end = chunk["end"]
            c_text = chunk["text"]
            c_vector = chunk["vector"]

            # 청크의 중심점 계산
            c_center = (c_start + c_end) // 2

            # 이진 탐색으로 해당 오프셋이 포함된 문서 인덱스 탐색
            # bisect_right: c_center보다 큰 첫 번째 start의 위치 - 1 이 해당 문서
            idx = bisect.bisect_right(doc_starts, c_center) - 1

            # 인덱스 범위 보정
            idx = max(0, min(idx, len(doc_ranges) - 1))
            matched_metadata = (
                doc_ranges[idx]["metadata"].copy()
                if doc_ranges[idx]["metadata"]
                else {}
            )

            final_docs.append(Document(page_content=c_text, metadata=matched_metadata))
            final_vectors.append(c_vector)

        logger.info(
            f"의미론적 문서 분할 완료: {len(docs)}개 원본 문서 -> {len(final_docs)}개 청크 생성 (벡터 포함)"
        )
        return final_docs, final_vectors
