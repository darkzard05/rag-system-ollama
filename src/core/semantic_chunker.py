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
        Returns:
            List[dict]: [{'text': str, 'start': int, 'end': int}, ...]
        """
        # 정규식으로 분할 (re.split과 유사하게 작동하되 오프셋 추적)
        # 정규식: ([.!?]\s+) -> 구분자가 캡처됨
        parts = list(self._sentence_pattern.finditer(text))
        final_sentences = []

        last_pos = 0
        for match in parts:
            # 구분자 매치 정보
            sep_start, sep_end = match.span()
            sep_text = match.group()

            # 구분자 앞쪽 텍스트 (문장 내용)
            content_raw = text[last_pos:sep_start]

            # 내용이 있거나 구분자가 있으면 처리
            full_span_text = content_raw + sep_text

            # 공백 정리 (개행 -> 공백) 하되, 길이는 유지해야 오프셋 계산 가능?
            # 아니오, 오프셋은 '원본 텍스트' 기준이어야 함.
            # 따라서 text.replace("\n", " ")는 여기서 수행하면 안되고,
            # 임베딩용 텍스트(cleaned_text)와 원본 오프셋을 따로 관리해야 함.

            cleaned_text = full_span_text.replace("\n", " ").strip()

            if cleaned_text:
                # strip()으로 인해 앞쪽 공백이 제거되었을 수 있으므로 오프셋 보정
                # 앞쪽 공백 길이 계산
                raw_segment = text[last_pos:sep_end]
                leading_spaces = len(raw_segment) - len(raw_segment.lstrip())

                real_start = last_pos + leading_spaces

                # replace("\n", " ")는 길이를 변화시키지 않음 (1문자->1문자).
                # 하지만 strip()은 길이를 줄임.
                # 정확한 매핑을 위해:
                # start: lstrip() 후의 위치
                # end: rstrip() 후의 위치

                segment_start = last_pos
                segment_end = sep_end

                # 실제 유효 텍스트 범위 찾기
                sub_text = text[segment_start:segment_end]
                l_stripped = sub_text.lstrip()
                n_leading = len(sub_text) - len(l_stripped)

                stripped = l_stripped.rstrip()
                n_trailing = len(l_stripped) - len(stripped)

                final_start = segment_start + n_leading
                final_end = segment_end - n_trailing

                # 임베딩용 텍스트 (내부 개행만 치환)
                embed_text = sub_text[n_leading : len(sub_text) - n_trailing].replace(
                    "\n", " "
                )

                if embed_text:
                    final_sentences.append(
                        {"text": embed_text, "start": final_start, "end": final_end}
                    )

            last_pos = sep_end

        # 남은 텍스트 처리
        if last_pos < len(text):
            remaining = text[last_pos:]
            cleaned = remaining.replace("\n", " ").strip()
            if cleaned:
                sub_text = remaining
                l_stripped = sub_text.lstrip()
                n_leading = len(sub_text) - len(l_stripped)
                stripped = l_stripped.rstrip()
                n_trailing = len(l_stripped) - len(stripped)

                final_start = last_pos + n_leading
                final_end = len(text) - n_trailing
                embed_text = stripped.replace("\n", " ")

                final_sentences.append(
                    {"text": embed_text, "start": final_start, "end": final_end}
                )

        return final_sentences

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """
        텍스트 리스트의 임베딩을 생성합니다 (배치 처리).
        """
        try:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                embeddings.extend(self.embedder.embed_documents(batch))
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"임베딩 생성 중 오류 발생: {e}")
            raise

    def _calculate_similarities(self, embeddings: np.ndarray) -> list[float]:
        """
        인접 문장 간의 코사인 유사도를 계산합니다.
        """
        if len(embeddings) < 2:
            return []

        # 1. 벡터 정규화 (L2 Norm)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # 0으로 나누기 방지
        normalized_embeddings = embeddings / np.where(norms == 0, 1e-10, norms)

        # 2. 인접 벡터 간 내적 (유사도)
        similarities = np.sum(
            normalized_embeddings[:-1] * normalized_embeddings[1:], axis=1
        )

        return similarities.tolist()

    def _find_breakpoints(self, similarities: list[float]) -> list[int]:
        """
        유사도 분포를 분석하여 분할 지점(breakpoints)을 찾습니다.
        """
        if not similarities:
            return []

        similarities_array = np.array(similarities)

        if self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(
                similarities_array, 100 - self.breakpoint_threshold_value
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(similarities_array)
            std = np.std(similarities_array)
            threshold = mean - (self.breakpoint_threshold_value * std)
        elif self.breakpoint_threshold_type == "similarity_threshold":
            threshold = self.breakpoint_threshold_value
        else:
            # 기본값
            threshold = self.similarity_threshold

        # Threshold보다 유사도가 낮은 지점을 분할점으로 선택
        breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]

        logger.debug("청킹 분석: 임계값={threshold:.3f}, 분기점={len(breakpoints)}개")

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

            if merged_len <= self.max_chunk_size:
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
        sentences = []
        if raw_sentences:
            current_s = raw_sentences[0]
            for s in raw_sentences[1:]:
                if len(s["text"]) < 3:
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

        # 1. 문서 통합 및 오프셋 매핑 구축
        full_text = ""
        doc_ranges = []
        current_offset = 0

        for doc in docs:
            content = doc.page_content
            if not content:
                continue

            # 문서 사이 공백 추가 (첫 문서 제외)
            if current_offset > 0:
                full_text += " "
                current_offset += 1

            start = current_offset
            full_text += content
            end = current_offset + len(content)

            doc_ranges.append({"start": start, "end": end, "metadata": doc.metadata})

            current_offset = end

        if not full_text.strip():
            logger.warning("split_documents: 병합된 텍스트가 비어있습니다.")
            return [], []

        # 2. 청킹 수행 (오프셋 및 벡터 정보 포함)
        # split_text는 이제 [{'text': str, 'start': int, 'end': int, 'vector': np.ndarray}]를 반환
        chunk_dicts = self.split_text(full_text)

        # 3. 메타데이터 역매핑 및 문서 객체 생성
        final_docs = []
        final_vectors = []

        for chunk in chunk_dicts:
            c_start = chunk["start"]
            c_end = chunk["end"]
            c_text = chunk["text"]
            c_vector = chunk["vector"]  # 이미 계산된 벡터

            # 청크의 중심점 계산
            c_center = (c_start + c_end) // 2

            # 중심점이 속한 원본 문서 찾기
            matched_metadata = {}

            # 순차 탐색
            found = False
            for i, r in enumerate(doc_ranges):
                if r["start"] <= c_center < r["end"]:
                    matched_metadata = r["metadata"].copy() if r["metadata"] else {}
                    found = True
                    break

            # [Fix] 정확한 범위를 못 찾은 경우 (문서 사이 공백에 중심점이 위치)
            # 가장 가까운 문서를 찾아 매핑 (Gap 보정)
            if not found and doc_ranges:
                # 1. 범위 밖 (맨 앞보다 전, 맨 뒤보다 후)
                if c_center < doc_ranges[0]["start"]:
                    matched_metadata = doc_ranges[0]["metadata"].copy()
                elif c_center >= doc_ranges[-1]["end"]:
                    matched_metadata = doc_ranges[-1]["metadata"].copy()
                else:
                    # 2. 문서 사이 Gap에 위치한 경우
                    # 현재 c_center보다 시작점이 큰 첫 번째 문서를 찾으면, 그 문서(Next)나 그 앞 문서(Prev) 중 가까운 것 선택
                    for i, r in enumerate(doc_ranges):
                        if r["start"] > c_center:
                            # r은 Gap 바로 뒤의 문서
                            prev_r = doc_ranges[i - 1] if i > 0 else r

                            # 거리 비교: (Gap~Next) vs (Prev~Gap)
                            dist_to_next = r["start"] - c_center
                            dist_to_prev = c_center - prev_r["end"]

                            if dist_to_prev <= dist_to_next:
                                matched_metadata = prev_r["metadata"].copy()
                            else:
                                matched_metadata = r["metadata"].copy()
                            break

            # 최후의 안전장치: 여전히 비어있다면 첫 번째 문서 정보 사용
            if not matched_metadata and doc_ranges:
                matched_metadata = doc_ranges[0]["metadata"].copy()

            final_docs.append(Document(page_content=c_text, metadata=matched_metadata))
            final_vectors.append(c_vector)

        logger.info(
            f"의미론적 문서 분할 완료: {len(docs)}개 원본 문서 -> {len(final_docs)}개 청크 생성 (벡터 포함)"
        )
        return final_docs, final_vectors
