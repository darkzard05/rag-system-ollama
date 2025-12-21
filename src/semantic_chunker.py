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
        sentence_split_regex: str = r"[.!?]\s+",
        min_chunk_size: int = 100,
        max_chunk_size: int = 800,
        similarity_threshold: float = 0.5,
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
        """
        self.embedder = embedder
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_value = breakpoint_threshold_value
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        
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
            f"Text split into {len(optimized_chunks)} semantic chunks "
            f"(avg size: {np.mean([len(c) for c in optimized_chunks]):.0f} chars)"
        )
        
        return optimized_chunks
    
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
            embeddings = self.embedder.embed_documents(texts)
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
        similarities = []
        
        for i in range(len(embeddings) - 1):
            # 코사인 유사도 계산
            e1 = embeddings[i]
            e2 = embeddings[i + 1]
            
            norm1 = np.linalg.norm(e1)
            norm2 = np.linalg.norm(e2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = float(np.dot(e1, e2) / (norm1 * norm2))
            
            similarities.append(similarity)
        
        return similarities
    
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
        else:
            threshold = self.similarity_threshold
        
        # Threshold 이하인 지점을 분할점으로 선택
        breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        
        logger.debug(f"Threshold: {threshold:.3f}, Breakpoints: {len(breakpoints)}")
        
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
                    optimized.append(buffer + " " + chunk)
                    buffer = ""
                else:
                    optimized.append(chunk)
            else:
                # 최대 크기 초과: 문장으로 추가 분할
                sentences = re.split(self.sentence_split_regex, chunk)
                sentences = [s.strip() for s in sentences if s.strip()]
                
                sub_buffer = ""
                for sentence in sentences:
                    if len(sub_buffer) + len(sentence) <= self.max_chunk_size:
                        sub_buffer += " " + sentence if sub_buffer else sentence
                    else:
                        if sub_buffer:
                            if buffer:
                                optimized.append(buffer + " " + sub_buffer)
                                buffer = ""
                            else:
                                optimized.append(sub_buffer)
                        sub_buffer = sentence
                
                if sub_buffer:
                    buffer = sub_buffer
        
        # 남은 버퍼 처리
        if buffer:
            if optimized:
                optimized[-1] += " " + buffer
            else:
                optimized.append(buffer)
        
        return optimized
