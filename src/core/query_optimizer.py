"""
LLM-based Semantic Router for Intent-based RAG Pipelines.
Optimized for 4B models to achieve < 500ms routing latency with caching.
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
    Optimized for 4B models to achieve < 500ms routing latency with caching.
    """

    # [최적화] 의도 분석 결과 및 샘플 벡터 캐시 (메모리 절감 및 속도 향상)
    _intent_cache: dict[str, str] = {}
    _sample_vectors_cache: dict[str, Any] = {}
    _max_cache_size = 100

    ROUTING_PROMPT = """Classify as A (Greeting), B (Fact), or C (Analysis). Answer ONLY with one letter."""

    # [최적화] 의도별 시맨틱 예시 문장
    INTENT_SAMPLES = {
        "GREETING": ["안녕", "반가워", "누구니", "자기소개해줘"],
        "FACTOID": ["제목이 뭐야", "저자가 누구야", "페이지 수 알려줘"],
        "RESEARCH": ["전체 내용 요약해줘", "주요 특징 분석해줘", "결론이 뭐야"],
    }

    @classmethod
    async def classify_intent(cls, query: str, llm: Any) -> str:
        """
        의도를 분석합니다. (룰 -> 캐시된 임베딩 -> LLM 3단계 하이브리드)
        """
        start_time = time.time()

        # 1. 룰 기반 고속 필터링 (Static Layer)
        clean_q = query.strip().lower()
        if len(clean_q) < 10 and any(
            w in clean_q for w in ["안녕", "hi", "hello", "반가"]
        ):
            return "GREETING"

        # 2. 결과 캐시 확인
        if clean_q in cls._intent_cache:
            return cls._intent_cache[clean_q]

        # 3. 임베딩 기반 시맨틱 매칭 (Cached Semantic Layer)
        import numpy as np

        from core.session import SessionManager

        embedder = SessionManager.get("embedder")
        if embedder:
            try:
                # [최적화] 샘플 벡터를 행렬화하여 단 한 번의 행렬 곱으로 모든 유사도 계산
                if not cls._sample_vectors_cache:
                    for intent, samples in cls.INTENT_SAMPLES.items():
                        cls._sample_vectors_cache[intent] = np.array(
                            embedder.embed_documents(samples)
                        ).astype("float32")

                q_vec = np.array(embedder.embed_query(query)).astype("float32")
                q_norm = np.linalg.norm(q_vec)

                best_intent = "FACTOID"
                max_sim = -1.0

                if q_norm > 0:
                    for intent, s_matrix in cls._sample_vectors_cache.items():
                        # s_matrix: (num_samples, dim), q_vec: (dim,)
                        # 행렬-벡터 곱으로 해당 의도의 모든 샘플과의 유사도를 한 번에 계산
                        dot_products = s_matrix @ q_vec
                        s_norms = np.linalg.norm(s_matrix, axis=1)
                        sims = dot_products / (s_norms * q_norm)

                        current_max = np.max(sims)
                        if current_max > max_sim:
                            max_sim = current_max
                            best_intent = intent

                # 유사도가 임계값(0.75) 이상이면 즉시 확정
                if max_sim > 0.75:
                    latency = (time.time() - start_time) * 1000
                    logger.info(
                        f"[ROUTER] 시맨틱 매칭 성공 | {best_intent} | Sim: {max_sim:.2f} | {latency:.1f}ms"
                    )
                    cls._intent_cache[clean_q] = best_intent
                    return best_intent
            except Exception as e:
                logger.debug(f"[ROUTER] 시맨틱 매칭 건너뜀: {e}")

        # 4. LLM 기반 정밀 라우팅 (Reasoning Layer - Final Fallback)
        system_msg = "Output ONLY 'A', 'B', or 'C'. NO thinking. NO explanation."

        bound_llm = llm
        if hasattr(llm, "bind"):
            bound_llm = llm.bind(
                stop=["\n", " "], options={"temperature": 0.0, "num_predict": 2}
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_msg),
                (
                    "human",
                    f"Task: Classify Query\nRules:\nA: Greeting\nB: Simple Fact\nC: Complex Analysis\n\nQuery: {query}\nResult:",
                ),
            ]
        )

        chain = prompt | bound_llm | StrOutputParser()

        try:
            # 1-토큰 분류 실행
            response = await chain.ainvoke({"input": query})
            intent_raw = response.strip().upper()

            # 첫 글자만 추출
            intent = intent_raw[0] if intent_raw else "B"

            if intent == "A":
                result = "GREETING"
            elif intent == "C":
                result = "RESEARCH"
            else:
                result = "FACTOID"

            latency = (time.time() - start_time) * 1000
            logger.info(f"[CHAT] [ROUTER] 의도 분석 완료 | {result} | {latency:.1f}ms")

            if len(cls._intent_cache) < cls._max_cache_size:
                cls._intent_cache[clean_q] = result

            return result
        except Exception as e:
            logger.error(f"[CHAT] [ROUTER] 분석 실패 | {e}")
            return "FACTOID"
