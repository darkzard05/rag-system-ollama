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

    # [최적화] 의도별 시맨틱 예시 문장 (재구성 및 요약 의도 강화)
    INTENT_SAMPLES = {
        "GREETING": ["안녕", "반가워", "누구니", "자기소개해줘"],
        "FACTOID": [
            "제목이 뭐야",
            "저자가 누구야",
            "페이지 수 알려줘",
            "수치가 얼마야",
        ],
        "RESEARCH": [
            "주요 특징 분석해줘",
            "결론이 뭐야",
            "상세히 설명해줘",
            "상관관계가 뭐야",
        ],
        "SUMMARY": [
            "이 문서 전체 내용을 요약해줘",
            "질문과 답변 형식으로 재구성해줘",
            "이 파일의 핵심 주제를 정리해줘",
            "문서 전반에 걸쳐 무엇을 다루고 있니",
            "초록과 결론을 요약해줘",
        ],
    }

    @classmethod
    async def classify_intent(cls, query: str, llm: Any) -> str:
        """
        의도를 분석합니다. (룰 -> 캐시된 임베딩 -> LLM 3단계 하이브리드)
        """
        start_time = time.time()

        # 1. 룰 기반 고속 필터링 (Static Layer - 인사성 질문만 최소한으로 유지)
        clean_q = query.strip().lower()
        if len(clean_q) < 8 and any(
            w in clean_q for w in ["안녕", "hi", "hello", "반가"]
        ):
            return "GREETING"

        # 2. 결과 캐시 확인
        if clean_q in cls._intent_cache:
            return cls._intent_cache[clean_q]

        # 3. 임베딩 기반 시맨틱 매칭 (Cached Semantic Layer - NumPy Matrix Ops)
        import numpy as np

        from core.session import SessionManager

        embedder = SessionManager.get("embedder")
        if embedder:
            try:
                # [최적화] 모든 의도의 샘플을 단 하나의 거대 행렬로 통합 및 노름(Norm) 사전 계산
                if not cls._sample_vectors_cache:
                    all_vectors = []
                    intent_indices = []
                    intent_names = list(cls.INTENT_SAMPLES.keys())

                    for i, intent in enumerate(intent_names):
                        samples = cls.INTENT_SAMPLES[intent]
                        vecs = np.array(embedder.embed_documents(samples)).astype(
                            "float32"
                        )
                        all_vectors.append(vecs)
                        intent_indices.extend([i] * len(samples))

                    full_matrix = np.vstack(all_vectors)
                    # [최적화] 노름을 미리 계산하여 런타임 연산량 감소
                    s_norms = np.linalg.norm(full_matrix, axis=1)

                    cls._sample_vectors_cache = {
                        "matrix": full_matrix,
                        "norms": s_norms,
                        "indices": np.array(intent_indices),
                        "names": intent_names,
                    }

                q_vec = np.array(embedder.embed_query(query)).astype("float32")
                q_norm = np.linalg.norm(q_vec)

                if q_norm > 0:
                    s_matrix = cls._sample_vectors_cache["matrix"]
                    s_norms = cls._sample_vectors_cache["norms"]
                    intent_indices = cls._sample_vectors_cache["indices"]
                    intent_names = cls._sample_vectors_cache["names"]

                    # [최적화] 사전 계산된 노름을 사용하여 코사인 유사도 고속 계산
                    dot_products = s_matrix @ q_vec
                    sims = dot_products / (s_norms * q_norm)

                    best_sample_idx = np.argmax(sims)
                    max_sim = sims[best_sample_idx]
                    best_intent = intent_names[intent_indices[best_sample_idx]]

                    # 매우 높은 유사도(0.8)일 때만 즉시 확정하여 정확도 보장
                    if max_sim > 0.8:
                        latency = (time.time() - start_time) * 1000
                        logger.info(
                            f"[ROUTER] 시맨틱 매칭 성공 | {best_intent} | {latency:.1f}ms"
                        )
                        cls._intent_cache[clean_q] = best_intent
                        return best_intent
            except Exception as e:
                logger.debug(f"[ROUTER] 시맨틱 매칭 건너뜀: {e}")

        # 4. LLM 기반 정밀 라우팅 (Reasoning Layer - Final Fallback)
        system_msg = "Output ONLY 'A', 'B', 'C', or 'D'. NO explanation."

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
                    f"Task: Classify Query Intent\nRules:\n"
                    "A: Greeting (Hello, who are you?)\n"
                    "B: Simple Fact (Specific numbers, names, dates)\n"
                    "C: Complex Analysis (Relationships, why/how, detailed explanation)\n"
                    "D: Global/Summary (Overall summary, transformation, reconstruction of the whole document)\n\n"
                    f"Query: {query}\nResult:",
                ),
            ]
        )

        chain = prompt | bound_llm | StrOutputParser()

        try:
            response = await chain.ainvoke({"input": query})
            intent_raw = response.strip().upper()
            intent_code = intent_raw[0] if intent_raw else "B"

            intent_map = {
                "A": "GREETING",
                "B": "FACTOID",
                "C": "RESEARCH",
                "D": "SUMMARY",
            }
            result = intent_map.get(intent_code, "FACTOID")

            latency = (time.time() - start_time) * 1000
            logger.info(f"[CHAT] [ROUTER] 정밀 분석 완료 | {result} | {latency:.1f}ms")

            if len(cls._intent_cache) < cls._max_cache_size:
                cls._intent_cache[clean_q] = result

            return result
        except Exception as e:
            logger.error(f"[CHAT] [ROUTER] 분석 실패 | {e}")
            return "FACTOID"
