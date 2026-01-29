"""
LLM-based Semantic Router for Intent-based RAG Pipelines.
Optimized for 4B models to achieve < 500ms routing latency.
"""

import logging
import time
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


class RAGQueryOptimizer:
    """
    LLM Intent Classifier.
    Categorizes queries into:
    [A] GREETING: Simple hellos or bot identity questions.
    [B] FACTOID: Specific, clear questions that need direct retrieval.
    [C] RESEARCH: Complex, vague, or multi-step questions that need query expansion.
    """

    ROUTING_PROMPT = """
자 너는 이제부터 질문 분류기야. 사용자의 [질문]을 보고 아래 규칙에 따라 딱 한 글자(A, B, C)로만 대답해.

[규칙]
- [A]: 인사, 자기소개, 감정 표현 등 문서 검색이 아예 필요 없는 질문.
- [B]: "제목이 뭐야?", "저자는 누구야?" 처럼 키워드가 명확하여 바로 검색 가능한 질문.
- [C]: "내용 요약해줘", "전체적인 특징 분석해줘" 처럼 여러 번의 검색이나 확장이 필요한 복잡한 질문.

[질문]: "{input}"
답변(A/B/C):"""

    @classmethod
    async def classify_intent(cls, query: str, llm: Any) -> str:
        """
        Classifies the intent using the LLM with minimal token generation.
        """
        start_time = time.time()

        # 4B 모델용 초경량 체인
        prompt = ChatPromptTemplate.from_template(cls.ROUTING_PROMPT)
        # stop 시퀀스를 설정하여 모델이 길게 말하는 것을 원천 차단 (지연 시간 최적화)
        chain = prompt | llm.bind(stop=["\n", " ", "."]) | StrOutputParser()

        try:
            # 1-토큰 분류 실행
            response = await chain.ainvoke({"input": query})
            intent = response.strip().upper()

            # 결과 정규화 (예외 상황 대비)
            if "A" in intent:
                result = "GREETING"
            elif "C" in intent:
                result = "RESEARCH"
            else:
                result = "FACTOID"  # 기본값

            latency = (time.time() - start_time) * 1000
            logger.info(
                f"[Router] Intent: {result} | Latency: {latency:.2f}ms | Query: {query[:30]}"
            )

            return result
        except Exception as e:
            logger.error(f"[Router] Error: {e}")
            return "FACTOID"  # 에러 시 안전한 기본값

    @classmethod
    def is_complex_query(cls, query: str) -> bool:
        """Legacy helper - defaults to True to be safe."""
        return len(query) > 30
