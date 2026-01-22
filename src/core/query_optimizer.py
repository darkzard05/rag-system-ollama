"""
RAG 전용 쿼리 최적화 모듈.
질문의 복잡도를 분석하여 쿼리 확장(Multi-Query) 수행 여부를 결정합니다.
"""

import logging
import re
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

class RAGQueryOptimizer:
    """
    RAG 시스템의 질의 최적화 엔진.
    질문의 길이, 키워드, 구조를 분석하여 최적의 검색 전략을 제안합니다.
    """
    
    # 확장 건너뛰기용 명확한 키워드 목록
    SIMPLE_KEYWORDS = [
        "제목", "저자", "작성자", "날짜", "요약", "결론", 
        "전체 내용", "라이선스", "버전", "목차"
    ]
    
    # 의문사 패턴
    QUESTION_PATTERNS = [
        r"인가요\?", r"뭐야\?", r"누구야\?", r"언제야\?", r"어디야\?"
    ]

    @classmethod
    def is_complex_query(cls, query: str) -> bool:
        """
        질문이 쿼리 확장이 필요한 복잡한 질문인지 판단합니다.
        
        Args:
            query: 사용자의 원본 질문
            
        Returns:
            bool: 확장이 필요하면 True, 단순 질문이면 False
        """
        query = query.strip()
        
        # 1. 길이 기반 판단 (매우 짧은 질문은 단순함)
        if len(query) < 20:
            logger.debug(f"[QueryOptimizer] 짧은 질문({len(query)}자)으로 판단: {query}")
            return False
            
        # 2. 명확한 키워드 포함 여부 확인
        for word in cls.SIMPLE_KEYWORDS:
            if word in query:
                logger.debug(f"[QueryOptimizer] 명확한 키워드('{word}') 포함으로 판단: {query}")
                return False
                
        # 3. 단순 의문문 패턴 확인
        for pattern in cls.QUESTION_PATTERNS:
            if re.search(pattern, query):
                # 단, 문장이 길지 않을 때만 단순 질문으로 간주
                if len(query) < 40:
                    logger.debug(f"[QueryOptimizer] 단순 의문문 패턴 감지: {query}")
                    return False

        # 4. 그 외 40자 이상의 긴 문장은 확장이 유용하다고 판단
        if len(query) >= 40:
            return True
            
        return False

    @classmethod
    def get_optimized_expansion_prompt(cls, query: str) -> str:
        """
        질문에 최적화된 경량화 확장 프롬프트를 반환합니다.
        (향후 질문 타입별로 프롬프트를 다르게 가져갈 때 사용)
        """
        return "질문을 검색용 키워드 3개로 변환하라."