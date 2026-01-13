"""
임베딩 기반 의미론적 텍스트 분할기를 구현합니다.

이 모듈은 문서를 문장 단위로 우선 분할한 후, 인접 문장 간의 임베딩 유사도를 
계산하여 유사도가 낮은 지점을 경계로 선택합니다. 이를 통해 의미론적으로 
일관성 있는 청크를 생성합니다.
"""

import logging
import re
from typing import List, Optional, Any, TYPE_CHECKING
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
    
    def _split_sentences(self, text: str) -> List[str]:
        """
        문장 단위로 텍스트를 분할합니다.
        re.split을 사용하여 구분자 패턴과 캡처 패턴 모두 호환되도록 처리합니다.
        """
        # 개행 정규화
        text = text.replace("\n", " ")
        
        # [수정] findall -> split으로 변경하여 패턴 호환성 증대
        # split을 사용하면 캡처 그룹이 있는 경우 결과에 포함되고, 없는 경우 구분자로 사용되어 사라짐.
        parts = self._sentence_pattern.split(text)
        
        # 공백 제거 및 빈 문장 필터링
        return [p.strip() for p in parts if p.strip()]

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
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
    
    def _calculate_similarities(self, embeddings: np.ndarray) -> List[float]:
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
            normalized_embeddings[:-1] * normalized_embeddings[1:], 
            axis=1
        )
        
        return similarities.tolist()
    
    def _find_breakpoints(self, similarities: List[float]) -> List[int]:
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
        
        logger.debug(f"청킹 분석: 임계값={{threshold:.3f}}, 분기점={{len(breakpoints)}}개")
        
        return breakpoints

    def _optimize_chunk_sizes(self, chunks: List[str]) -> List[str]:
        """
        생성된 청크들의 크기를 검사하여 병합합니다.
        (주의: max_chunk_size보다 큰 단일 문장은 강제로 분할하지 않고 유지합니다)
        """
        if not chunks:
            return chunks
        
        optimized = []
        current_chunk = ""
        
        for chunk in chunks:
            if not current_chunk:
                current_chunk = chunk
                continue
            
            # 예상 병합 크기 (공백 포함)
            merged_len = len(current_chunk) + 1 + len(chunk)
            
            if merged_len <= self.max_chunk_size:
                # 합쳐도 최대 크기보다 작으면 병합
                current_chunk += " " + chunk
            else:
                # 합치면 너무 커지는 경우, 현재 청크 저장 후 새로 시작
                optimized.append(current_chunk)
                current_chunk = chunk
        
        if current_chunk:
            optimized.append(current_chunk)
        
        return optimized

    def split_text(self, text: str) -> List[str]:
        """
        텍스트를 의미론적으로 분할합니다.
        """
        if not text or not text.strip():
            return []
        
        # 1. 문장 분할
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return sentences
        
        # 2. 임베딩 및 유사도 계산
        embeddings = self._get_embeddings(sentences)
        similarities = self._calculate_similarities(embeddings)
        
        # 3. 분기점 탐색
        breakpoints = self._find_breakpoints(similarities)
        
        # 4. 1차 그룹화
        chunks = []
        start_idx = 0
        for bp in breakpoints:
            group = " ".join(sentences[start_idx:bp])
            if group.strip():
                chunks.append(group)
            start_idx = bp
        
        last_group = " ".join(sentences[start_idx:])
        if last_group.strip():
            chunks.append(last_group)
        
        # 5. 크기 최적화
        return self._optimize_chunk_sizes(chunks)

    def split_documents(self, docs: List["Document"]) -> List["Document"]:
        """
        LangChain Document 객체 리스트를 받아 의미론적 분할을 수행합니다.
        """

        if not docs:
            logger.warning("split_documents: 입력 문서 리스트가 비어있습니다.")
            return []

        # 1. 전체 텍스트 병합
        combined_text = " ".join([d.page_content for d in docs if d.page_content])
        
        if not combined_text.strip():
            logger.warning("split_documents: 병합된 텍스트가 비어있습니다. (빈 문서 또는 텍스트 추출 실패)")
            return []

        # 2. 청킹 수행
        chunk_texts = self.split_text(combined_text)
        
        # 3. 문서 객체 생성
        # 메타데이터 전파 (첫 번째 문서 기준)
        base_metadata = docs[0].metadata.copy() if docs else {}
        
        final_docs = [
            Document(page_content=text, metadata=base_metadata)
            for text in chunk_texts
        ]
        
        logger.info(f"의미론적 문서 분할 완료: {len(docs)}개 -> {len(final_docs)}개 청크")
        return final_docs
