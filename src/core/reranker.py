"""
FlashRank 기반 시맨틱 재순위화 모듈.
ONNX 런타임을 사용하여 로컬 환경에서 고속으로 문서의 관련성을 재평가합니다.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any

from langchain_core.documents import Document

from common.config import RERANKER_CONFIG, RERANKER_ENABLED

logger = logging.getLogger(__name__)


class RerankerStrategy(Enum):
    """재순위지정 전략"""

    SCORE_ONLY = "score_only"  # 점수 기반 (Pass-through)
    SEMANTIC_FLASH = "semantic_flash"  # FlashRank 시맨틱 리랭킹
    DIVERSITY = "diversity"  # 다양성 고려
    MMR = "mmr"  # Maximal Marginal Relevance


@dataclass
class RerankingResult:
    """재순위지정된 결과"""

    doc_id: str
    content: str
    original_score: float
    reranked_score: float
    original_rank: int
    final_rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __lt__(self, other):
        return self.reranked_score > other.reranked_score


@dataclass
class RerankingMetrics:
    """재순위지정 메트릭"""

    total_results: int = 0
    reranked_results: int = 0
    rank_changes: int = 0
    reranking_time: float = 0.0
    strategy_used: str = ""
    avg_score_before: float = 0.0
    avg_score_after: float = 0.0


class FlashReranker:
    """FlashRank 엔진을 관리하는 싱글톤 래퍼."""

    _instance = None
    _lock = Lock()
    _initialized: bool = False

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            from flashrank import Ranker

            model_name = RERANKER_CONFIG.get("model_name", "ms-marco-TinyBERT-L-2-v2")
            # max_length 최적화: 청크 사이즈보다 약간 크게 설정
            max_length = RERANKER_CONFIG.get("max_length", 512)

            logger.info(
                f"FlashRank 모델 로드 중: {model_name} (max_length: {max_length})"
            )
            self.ranker = Ranker(model_name=model_name, max_length=max_length)
            self._initialized = True
            logger.info("FlashRank 엔진 초기화 완료")
        except ImportError:
            logger.error("flashrank 라이브러리가 설치되지 않았습니다.")
            self.ranker = None
        except Exception as e:
            logger.error(f"FlashRank 초기화 실패: {e}")
            self.ranker = None

    def rerank_documents(
        self, query: str, documents: list[Document], top_k: int = 10
    ) -> list[Document]:
        """FlashRank를 사용하여 문서를 재순위화하고 상위 k개를 반환합니다."""
        if not self.ranker or not documents:
            return documents[:top_k]

        from flashrank import RerankRequest

        # 1. FlashRank 포맷 변환 (ID와 Metadata 보존)
        passages = []
        for i, doc in enumerate(documents):
            passages.append(
                {
                    "id": i,
                    "text": doc.page_content,
                    "meta": doc.metadata,
                }
            )

        # 2. 리랭킹 실행
        rerank_request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(rerank_request)

        # 3. 결과 제한 및 Document 객체로 복원
        reranked_docs = []
        for res in results[:top_k]:
            # FlashRank가 계산한 점수를 메타데이터에 추가
            meta = res.get("meta", {}).copy()
            meta["rerank_score"] = float(res.get("score", 0.0))

            reranked_docs.append(Document(page_content=res["text"], metadata=meta))

        return reranked_docs


class DistributedReranker:
    """통합 재순위지정 인터페이스."""

    def __init__(self):
        self.flash_engine = FlashReranker() if RERANKER_ENABLED else None
        self.bypass_threshold = RERANKER_CONFIG.get("bypass_threshold", 0.95)
        self.default_top_k = RERANKER_CONFIG.get("top_k", 10)

    def rerank(
        self,
        results: list[Any],
        query_text: str | None = None,
        strategy: RerankerStrategy = RerankerStrategy.SEMANTIC_FLASH,
        top_k: int | None = None,
        **kwargs,
    ) -> tuple[list[Document] | list[RerankingResult], RerankingMetrics]:
        """
        주어진 결과를 재순위화합니다.
        FlashRank가 활성화되어 있고 관련 쿼리가 있으면 시맨틱 리랭킹을 수행합니다.
        """
        if not results:
            return [], RerankingMetrics(strategy_used=strategy.value)

        start_time = time.time()
        target_top_k = top_k or self.default_top_k
        metrics = RerankingMetrics(
            total_results=len(results), strategy_used=strategy.value
        )

        # [Fast-Path] 스마트 얼리 엑시트 (Smart Early Exit)
        # 1위 점수가 압도적으로 높고 2위와의 점수 차이(Gap)가 클 경우 리랭킹 생략
        if len(results) >= 2:
            first_score = getattr(results[0], "score", 0.0)
            second_score = getattr(results[1], "score", 0.0)
            score_gap = first_score - second_score

            # 임계치: 1위가 0.95 이상이면서 2위와 0.3 이상의 차이가 나면 확정적 정답으로 간주
            if first_score >= self.bypass_threshold and score_gap >= 0.3:
                logger.info(
                    f"[RAG] [RERANK] Early Exit 활성화: 1위({first_score:.3f}), Gap({score_gap:.3f}) "
                    f"-> 리랭킹 생략 (Time saved: ~80ms)"
                )
                metrics.strategy_used = "early_exit_bypass"
                return results[:target_top_k], metrics
        elif len(results) == 1:
            # 결과가 하나뿐이면 리랭킹 무의미
            return results, metrics

        # 시맨틱 리랭킹 수행
        if (
            strategy == RerankerStrategy.SEMANTIC_FLASH
            and self.flash_engine
            and query_text
        ):
            # FlashRank는 Document 객체 리스트를 기대함
            docs = []
            for r in results:
                if isinstance(r, Document):
                    docs.append(r)
                else:
                    # RerankingResult 등 다른 타입일 경우 변환
                    docs.append(
                        Document(
                            page_content=getattr(
                                r, "page_content", getattr(r, "content", "")
                            ),
                            metadata=getattr(r, "metadata", {}),
                        )
                    )

            reranked_docs = self.flash_engine.rerank_documents(
                query_text, docs, top_k=target_top_k
            )

            # [수정] 하드 필터링 제거: 점수가 낮더라도 모든 결과를 반환하여 LLM이 판단하도록 함
            # (사용자 피드백 반영: 과도한 필터링으로 인한 정보 유실 방지)
            metrics.reranked_results = len(reranked_docs)
            metrics.reranking_time = time.perf_counter() - start_time
            return reranked_docs, metrics

        # 폴백: 점수 기반 정렬 (Score Only)
        reranked = sorted(results, key=lambda x: getattr(x, "score", 0.0), reverse=True)
        metrics.reranked_results = len(reranked[:target_top_k])
        metrics.reranking_time = time.time() - start_time
        return reranked[:target_top_k], metrics
