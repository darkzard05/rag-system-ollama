"""
임베딩 기반 의미론적 텍스트 분할기를 구현합니다.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를 
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로 
일관성 있는 청크를 생성합니다.
"""

import logging
import re
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingBasedSemanticChunker:
    """
    임베딩 기반 의미론적 텍스트 분할기.
    
    문장 단위로 분할 후, 임베딩 유사도 기반으로 의미 경계를 탐지하여
    일관성 있는 청크를 생성합니다.
    """
    
    def __init__(
        self,
        embedder,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_value: float = 95.0,
        sentence_split_regex: str = r"(?<=[.!?])\s+",
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        similarity_threshold: float = 0.5,
        batch_size: int = 64,
    ):
        """
        의미론적 청킹 분할기를 초기화합니다.
        
        Args:
            embedder: 텍스트 임베딩을 생성하는 모델 (HuggingFaceEmbeddings 등)
            breakpoint_threshold_type: 'percentile' 또는 'standard_deviation'
            breakpoint_threshold_value: threshold 값 (백분위수 또는 표준편차 배수)
            sentence_split_regex: 문장 분할 정규식
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
    
    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 의미론적으로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            
        Returns:
            분할된 청크 리스트
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # 1단계: 문장 단위로 분할
        sentences = self._split_sentences(text)
        if not sentences:
            return [text]
        
        if len(sentences) == 1:
            return sentences
        
        # 2단계: 문장 임베딩 생성
        sentence_embeddings = self._get_embeddings(sentences)
        
        # 3단계: 인접 문장 간 유사도 계산
        similarities = self._calculate_similarities(sentence_embeddings)
        
        # 4단계: 분할 지점 결정 (유사도 threshold 기반)
        breakpoints = self._find_breakpoints(similarities)
        
        # 5단계: 문장을 청크로 그룹화
        chunks = self._group_sentences(sentences, breakpoints)
        
        # 6단계: 크기 기반 최적화
        optimized_chunks = self._optimize_chunk_sizes(chunks)
        
        logger.info(
            f"텍스트 분할 완료: 총 {len(optimized_chunks)}개 청크 "
            f"(평균: {np.mean([len(c) for c in optimized_chunks]):.0f}자)"
        )
        
        return optimized_chunks
    
    def split_documents(self, docs: List["Document"]) -> List["Document"]:
        """
        문서 리스트를 받아서 의미론적 청킹을 수행합니다.
        여러 문서(페이지)를 하나로 합친 후 청킹하고, 다시 원본 메타데이터를 매핑합니다.
        
        Args:
            docs: 분할할 문서(페이지) 리스트
            
        Returns:
            의미론적으로 분할되고 메타데이터가 복원된 문서 리스트
        """
        from langchain_core.documents import Document

        if not docs:
            return []

        # 1. 문서 병합 및 오프셋 매핑 생성
        # split_text 내부에서 개행을 공백으로 치환하므로, 매핑 정확도를 위해 미리 전처리합니다.
        normalized_docs = []
        doc_boundaries = []
        combined_text = ""
        current_idx = 0
        
        for doc in docs:
            # 개행 문자 및 다중 공백 정규화 (split_text 로직과 일치시킴)
            # split_text는 내부적으로 문장을 분리하고 다시 ' '.join() 하므로 단일 공백으로 정규화 필요
            content = re.sub(r"\s+", " ", doc.page_content).strip()
            
            if combined_text:
                # 문맥 분리 방지를 위해 공백으로 연결
                combined_text += " "
                current_idx += 1
            
            start = current_idx
            end = start + len(content)
            doc_boundaries.append((start, end, doc.metadata))
            
            combined_text += content
            current_idx = end

        # 2. 통합된 텍스트로 청킹 수행
        chunk_texts = self.split_text(combined_text)

        # 3. 청크를 문서 객체로 변환 및 메타데이터 복원
        final_docs = []
        current_search_idx = 0
        
        # ✅ 폴백을 위한 초기 메타데이터 설정 (첫 문서 기준)
        last_valid_metadata = docs[0].metadata.copy() if docs else {}
        
        for chunk_text in chunk_texts:
            # 청크의 텍스트 위치 찾기 (순차 검색)
            start_pos = combined_text.find(chunk_text, current_search_idx)
            
            # 혹시라도 못 찾는 경우 대비
            if start_pos == -1:
                logger.warning(f"경고: 병합된 텍스트에서 청크를 찾을 수 없습니다. (폴백 메타데이터 사용)")
                final_docs.append(Document(page_content=chunk_text, metadata=last_valid_metadata.copy()))
                continue
                
            end_pos = start_pos + len(chunk_text)
            current_search_idx = end_pos
            
            # 이 청크가 걸쳐있는 원본 문서들의 메타데이터 수집
            involved_pages = []
            source_file = None
            base_metadata = {}

            for doc_start, doc_end, meta in doc_boundaries:
                # 구간 겹침 확인: max(a_start, b_start) < min(a_end, b_end)
                if max(start_pos, doc_start) < min(end_pos, doc_end):
                    if source_file is None:
                        source_file = meta.get("source")
                        base_metadata = meta.copy() # 첫 매칭된 문서의 메타데이터를 베이스로 사용
                    
                    page = meta.get("page")
                    if page is not None:
                        involved_pages.append(page)

            # 메타데이터 업데이트
            if involved_pages:
                unique_pages = sorted(list(set(involved_pages)))
                if len(unique_pages) == 1:
                    base_metadata["page"] = unique_pages[0]
                else:
                    # 여러 페이지에 걸친 경우 "1-2" 형태로 표기
                    base_metadata["page"] = f"{unique_pages[0]}-{unique_pages[-1]}"
            
            # base_metadata가 비어있다면(매칭 실패 등) 폴백 사용
            if not base_metadata and last_valid_metadata:
                base_metadata = last_valid_metadata.copy()
            
            final_docs.append(Document(page_content=chunk_text, metadata=base_metadata))
            
            # 유효한 메타데이터가 있으면 백업 업데이트
            if base_metadata:
                last_valid_metadata = base_metadata.copy()

        return final_docs
    
    def _split_sentences(self, text: str) -> List[str]:
        """문장 단위로 텍스트를 분할합니다."""
        # 개행 정규화
        text = text.replace("\n", " ")
        
        # ✅ 사전 컴파일된 정규식 사용 (성능 개선)
        sentences = self._sentence_pattern.split(text)
        
        # 공백 제거 및 빈 문장 필터링
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        텍스트 리스트의 임베딩을 생성합니다.
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            형태가 (n, embedding_dim)인 임베딩 배열
        """
        try:
            embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                embeddings.extend(self.embedder.embed_documents(batch))
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _calculate_similarities(self, embeddings: np.ndarray) -> List[float]:
        """
        인접 문장 간의 코사인 유사도를 계산합니다.
        
        Args:
            embeddings: 문장 임베딩 배열 (n, dim)
            
        Returns:
            유사도 리스트 (길이: n-1)
        """
        if len(embeddings) < 2:
            return []

        # 1. 벡터 정규화 (L2 Norm)
        # axis=1은 각 행(문장 벡터)에 대해 계산
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 0으로 나누기 방지 (매우 작은 값 더함)
        norms[norms == 0] = 1e-10
        
        normalized_embeddings = embeddings / norms

        # 2. 인접 벡터 간 내적 (Vectorized Dot Product)
        # normalized_embeddings[:-1] : 0번부터 n-2번까지
        # normalized_embeddings[1:]  : 1번부터 n-1번까지
        # 이 둘을 요소별로 곱한 뒤 합치면 코사인 유사도가 됨
        similarities = np.sum(
            normalized_embeddings[:-1] * normalized_embeddings[1:], 
            axis=1
        )
        
        return similarities.tolist()
    
    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
        """
        유사도 기반으로 분할 지점을 찾습니다.
        
        Args:
            similarities: 인접 문장 간 유사도 리스트
            
        Returns:
            분할할 문장 인덱스 리스트
        """
        if not similarities:
            return []
        
        similarities_array = np.array(similarities)
        
        # Threshold 계산
        if self.breakpoint_threshold_type == "percentile":
            threshold = np.percentile(
                similarities_array, 100 - self.breakpoint_threshold_value
            )
        elif self.breakpoint_threshold_type == "standard_deviation":
            mean = np.mean(similarities_array)
            std = np.std(similarities_array)
            threshold = mean - (self.breakpoint_threshold_value * std)
        elif self.breakpoint_threshold_type == "similarity_threshold":
            # 설정된 절대적 유사도 기준 사용
            threshold = self.breakpoint_threshold_value
        else:
            # 기본값
            threshold = self.similarity_threshold
        
        # Threshold 이하인 지점을 분할점으로 선택
        breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        
        logger.info(f"청킹 수행 (임계값: {threshold:.3f}, 발견된 분기점: {len(breakpoints)}개)")
        
        return breakpoints
    
    def _group_sentences(
        self, sentences: List[str], breakpoints: List[int]
    ) -> List[str]:
        """
        문장을 분할 지점에 따라 청크로 그룹화합니다.
        
        Args:
            sentences: 문장 리스트
            breakpoints: 분할 지점 인덱스 리스트
            
        Returns:
            그룹화된 청크 리스트
        """
        if not breakpoints:
            return [" ".join(sentences)]
        
        chunks = []
        start_idx = 0
        
        for breakpoint in breakpoints:
            chunk = " ".join(sentences[start_idx:breakpoint])
            if chunk.strip():
                chunks.append(chunk)
            start_idx = breakpoint
        
        # 마지막 청크
        if start_idx < len(sentences):
            chunk = " ".join(sentences[start_idx:])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _optimize_chunk_sizes(self, chunks: List[str]) -> List[str]:
        """
        청크 크기를 최적화합니다.
        청크 크기를 최적화하여 너무 작거나 큰 청크를 처리합니다.
        
        최소 크기 미만인 청크는 인접 청크와 병합하고,
        최대 크기를 초과하는 청크는 추가로 분할합니다.
        
        Args:
            chunks: 청크 리스트
            
        Returns:
            최적화된 청크 리스트
        """
        if not chunks:
            return chunks
        
        optimized = []
        buffer = ""
        
        for chunk in chunks:
            # 크기 체크
            chunk_size = len(chunk)
            
            if chunk_size < self.min_chunk_size:
                # 최소 크기 미만: 버퍼에 추가
                buffer += " " + chunk if buffer else chunk
            elif chunk_size <= self.max_chunk_size:
                # 범위 내: 버퍼와 함께 추가
                if buffer:
                    if len(buffer) + 1 + len(chunk) <= self.max_chunk_size:
                        optimized.append(buffer + " " + chunk)
                        buffer = ""
                    else:
                        optimized.append(buffer)
                        optimized.append(chunk)
                        buffer = ""
                else:
                    optimized.append(chunk)
            else:
                # 최대 크기 초과: 문장으로 추가 분할
                sentences = self._sentence_pattern.split(chunk)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                sub_buffer = ""
                for sentence in sentences:
                    if len(sub_buffer) + len(sentence) <= self.max_chunk_size:
                        sub_buffer += " " + sentence if sub_buffer else sentence
                    else:
                        if sub_buffer:
                            if buffer:
                                # 버퍼 병합 시 최대 크기 초과 여부 확인
                                if len(buffer) + 1 + len(sub_buffer) <= self.max_chunk_size:
                                    optimized.append(buffer + " " + sub_buffer)
                                else:
                                    optimized.append(buffer)
                                    optimized.append(sub_buffer)
                                buffer = ""
                            else:
                                optimized.append(sub_buffer)
                        sub_buffer = sentence
                
                if sub_buffer:
                    buffer = sub_buffer
        
        # 남은 버퍼 처리
        if buffer:
            if optimized and len(optimized[-1]) + len(buffer) + 1 <= self.max_chunk_size:
                optimized[-1] += " " + buffer
            else:
                optimized.append(buffer)
        
        return optimized
