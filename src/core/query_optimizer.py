"""
LLM-based Semantic Router for Intent-based RAG Pipelines.
Simplified to use direct LLM inference for maximum accuracy and maintainability.
"""

import logging
import time
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


class RAGQueryOptimizer:
    """
    LLM Intent Classifier.
    Simplified to Pure LLM routing with result caching.
    """

    # [최적화] 의도 분석 결과 캐시 (메모리 절감 및 속도 향상)
    _intent_cache: dict[str, str] = {}
    _max_cache_size = 100

    @classmethod
    async def classify_intent(cls, query: str, llm: Any) -> str:
        """
        의도를 분석합니다. (캐시 -> LLM 2단계 하이브리드)
        """
        start_time = time.time()
        clean_q = query.strip().lower()

        # 1. 결과 캐시 확인 (가장 빠름)
        if clean_q in cls._intent_cache:
            return cls._intent_cache[clean_q]

        # 2. LLM 기반 정밀 라우팅 (Reasoning Layer)
        system_msg = (
            "You are a query intent classifier. Output ONLY 'A', 'B', 'C', or 'D'."
        )

        bound_llm = llm
        if hasattr(llm, "bind"):
            # 출력을 엄격히 제한하여 속도 최적화
            bound_llm = llm.bind(
                stop=["\n", " ", "."], options={"temperature": 0.0, "num_predict": 5}
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                (
                    "human",
                    "Classify the intent of the following Korean or English query into one category:\n\n"
                    "A: [GREETING] - 인삿말, 자기소개 요청, 일상적인 대화 (예: 안녕, 너는 누구니)\n"
                    "B: [FACTOID] - 구체적인 사실 확인, 이름, 날짜, 수치, 제목 질문 (예: 저자가 누구야, 매출액 얼마야)\n"
                    "C: [RESEARCH] - 심층 분석, 이유(Why), 방법(How), 비교, 상관관계, 상세 설명 요구 (예: 왜 성능이 좋아, 장단점 비교해줘)\n"
                    "D: [SUMMARY] - 전체 요약, 핵심 주제 정리, 전체 내용 재구성, 전반적인 흐름 파악 (예: 전체 요약해줘, 주요 내용 정리해줘)\n\n"
                    f"Query: {query}\n"
                    "Result (ONLY A, B, C, or D):",
                ),
            ]
        )

        chain = prompt | bound_llm | StrOutputParser()

        try:
            response = await chain.ainvoke({"input": query})
            intent_raw = response.strip().upper()
            # 첫 글자만 추출 (A, B, C, D)
            intent_code = intent_raw[0] if intent_raw else "B"

            intent_map = {
                "A": "GREETING",
                "B": "FACTOID",
                "C": "RESEARCH",
                "D": "SUMMARY",
            }
            result = intent_map.get(intent_code, "FACTOID")

            latency = (time.time() - start_time) * 1000
            logger.info(f"[CHAT] [ROUTER] 의도 분석 완료 | {result} | {latency:.1f}ms")

            # 결과 캐싱
            if len(cls._intent_cache) < cls._max_cache_size:
                cls._intent_cache[clean_q] = result

            return result
        except Exception as e:
            logger.error(f"[CHAT] [ROUTER] 분석 실패 | {e}")
            return "FACTOID"
