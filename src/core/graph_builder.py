"""
LangGraph를 사용하여 자가 교정(Self-Correction) RAG 워크플로우를 구성합니다.
의도 분류, 캐시 확인, 하이브리드 검색, 문서 평가, 쿼리 재구성, 생성의 단계를 포함합니다.
"""

import asyncio
import copy
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

from api.schemas import AggregatedSearchResult, GraphState
from cache.response_cache import get_response_cache
from common.config import (
    ANALYSIS_PROTOCOL,
    GRADING_CONFIG,
    REWRITING_CONFIG,
)
from common.utils import fast_hash
from core.session import SessionManager

logger = logging.getLogger(__name__)


async def preprocess(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """의도 분류 및 캐시 확인을 수행합니다."""
    query = state["input"].strip()
    logger.info(f"[RAG] [PREPROCESS] 입력 질의: '{query}'")

    import re

    from common.config import DYNAMIC_WEIGHTING_CONFIG, ENSEMBLE_WEIGHTS

    # 1. 의도 분류 및 동적 가중치 결정
    # 기본값 (Config 기반)
    weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
    intent = "rag"

    # [추가] 간단한 인사말 및 일상 대화 감지 (General Intent)
    if len(query) < 10 and any(
        g in query.lower() for g in ["안녕", "hi", "hello", "반가워", "누구"]
    ):
        intent = "general"
        logger.info("[RAG] [PREPROCESS] 일상 대화(General) 의도 감지")

    # [최적화] 설정 기반 동적 가중치 적용
    if DYNAMIC_WEIGHTING_CONFIG.get("enabled", True):
        # A. 키워드 중심(Keyword-Heavy) 질문 감지
        keyword_patterns = DYNAMIC_WEIGHTING_CONFIG.get("keyword_patterns", [])
        is_keyword_heavy = any(re.search(p, query) for p in keyword_patterns)

        # B. 의미 중심(Semantic-Heavy) 질문 감지
        semantic_keywords = DYNAMIC_WEIGHTING_CONFIG.get("semantic_keywords", [])
        is_semantic_heavy = any(k in query for k in semantic_keywords)

        if is_keyword_heavy and not is_semantic_heavy:
            kw_w = DYNAMIC_WEIGHTING_CONFIG.get("keyword_weight", 0.8)
            weights = {"bm25": kw_w, "faiss": round(1.0 - kw_w, 1)}
            logger.info(f"[RAG] [PREPROCESS] 키워드 중심 질의 판단 (BM25: {kw_w})")
        elif is_semantic_heavy and not is_keyword_heavy:
            sm_w = DYNAMIC_WEIGHTING_CONFIG.get("semantic_weight", 0.8)
            weights = {"bm25": round(1.0 - sm_w, 1), "faiss": sm_w}
            logger.info(f"[RAG] [PREPROCESS] 의미 중심 질의 판단 (FAISS: {sm_w})")

    # 2. 캐시 확인
    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "의도 분석 및 지식 확인 중..."})

    cache = get_response_cache()
    cached_res = await cache.get(query, use_semantic=True)
    if cached_res:
        logger.info("[RAG] [PREPROCESS] 시맨틱 캐시 히트")
        SessionManager.add_status_log("캐시된 답변을 발견했습니다.")
        return {
            "response": cached_res.response,
            "thought": cached_res.metadata.get("thought", ""),
            "is_cached": True,
            "search_weights": weights,
        }

    return {
        "intent": intent,
        "is_cached": False,
        "search_weights": weights,
    }


async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """문서 검색 및 재순위화를 수행합니다."""
    if state.get("is_cached") or state.get("intent") == "general":
        return {}

    from core.search_aggregator import AggregationStrategy, SearchResultAggregator

    # [수정] 재시도 시 재구성된 쿼리가 있으면 그것을 사용
    query = state.get("input")
    if state.get("search_queries"):
        query = state["search_queries"][-1]
        logger.info(
            f"[RAG] [RETRIEVE] 재구성된 쿼리 사용: '{query}' (Retry: {state.get('retry_count')})"
        )
        SessionManager.add_status_log(f"재구성된 쿼리로 검색 시도: {query}")
    else:
        logger.info(f"[RAG] [RETRIEVE] 원본 쿼리 기반 검색 시작: '{query}'")

    cfg = config.get("configurable", {})

    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "관련 지식 탐색 중..."})
    SessionManager.add_status_log(f"지식 탐색 중: {query}")

    # 리트리버 획득
    bm25 = cfg.get("bm25_retriever")
    faiss = cfg.get("faiss_retriever")

    # 병렬 검색 (딕셔너리 기반으로 출처 명확화)
    search_tasks = {}
    if bm25:
        search_tasks["bm25"] = asyncio.create_task(bm25.ainvoke(query))
    if faiss:
        search_tasks["faiss"] = asyncio.create_task(faiss.ainvoke(query))

    # 결과 수집
    results = {}
    if search_tasks:
        task_names = list(search_tasks.keys())
        task_results = await asyncio.gather(*search_tasks.values())
        results = dict(zip(task_names, task_results, strict=False))

    logger.debug(
        f"[RAG] [RETRIEVE] 검색 결과 확보 (BM25: {len(results.get('bm25', []))}, Vector: {len(results.get('faiss', []))})"
    )

    # 결과 병합 및 RRF 집계
    all_docs = []
    for source, res in results.items():
        for doc in res:
            all_docs.append(
                AggregatedSearchResult(
                    doc_id=doc.metadata.get("doc_id", fast_hash(doc.page_content)),
                    content=doc.page_content,
                    score=doc.metadata.get("score", 0.5),
                    node_id=source,
                    metadata=doc.metadata,
                )
            )

    from common.config import ENSEMBLE_WEIGHTS

    aggregator = SearchResultAggregator()
    # [수정] 동적 가중치 적용: preprocess 노드에서 결정된 가중치 우선 사용
    weights = state.get("search_weights") or {
        "bm25": ENSEMBLE_WEIGHTS[0],
        "faiss": ENSEMBLE_WEIGHTS[1],
    }

    logger.info(
        f"[RAG] [RETRIEVE] 하이브리드 가중치 적용: BM25({weights['bm25']:.1f}), FAISS({weights['faiss']:.1f})"
    )

    aggregated, _ = aggregator.aggregate_results(
        {"all": all_docs},
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=25,
        weights=weights,
    )

    # [최적화] 리랭킹 후보군 동적 제한 (Pruning)
    # 초기 점수 분포가 조밀할 때만 많은 후보를 리랭킹함
    # 상위 1위와 5위의 점수 차이가 크면 후보군을 좁힘
    top_5_docs = aggregated[:5]
    if len(top_5_docs) >= 5:
        score_gap_5 = top_5_docs[0].aggregated_score - top_5_docs[4].aggregated_score
        # 점수 차이가 0.5 이상이면 상위권이 확실하므로 후보군 10개로 제한
        dynamic_top_k = 10 if score_gap_5 > 0.5 else 25
    else:
        dynamic_top_k = 25

    final_docs = [
        Document(page_content=r.content, metadata=r.metadata)
        for r in aggregated[:dynamic_top_k]
    ]

    if not final_docs:
        q_len = len(query) if query else 0
        logger.warning(
            f"[RAG] [RETRIEVE] 검색 결과가 전혀 없습니다 (Query Length: {q_len})"
        )
        SessionManager.add_status_log("검색된 문서가 없습니다.")
        return {"relevant_docs": []}

    # [최적화] 리랭킹 효율화: 병합 전 원본 청크들로 먼저 정밀 리랭킹 수행
    from core.reranker import DistributedReranker, RerankerStrategy

    reranker = DistributedReranker()
    # 후보군(Dynamic Top-K)에 대해 즉시 리랭킹 수행
    ranked_docs, _ = reranker.rerank(
        final_docs,
        query_text=query,
        strategy=RerankerStrategy.SEMANTIC_FLASH,
        top_k=GRADING_CONFIG.get("top_k", 5),  # 최종 선별 개수는 설정 따름
    )
    logger.info(
        f"[RAG] [RETRIEVE] 리랭킹 선별 완료: {len(final_docs)}개 후보 중 {len(ranked_docs)}개 최종 선별"
    )

    # [최적화] 리랭킹된 상위 문서들에 대해서만 지능적 문맥 병합 수행
    # (최종 생성 모델에 전달할 풍부한 문맥 확보)
    merged_context_docs = _merge_adjacent_chunks(ranked_docs, max_tokens=1200)
    logger.info(
        f"[RAG] [RETRIEVE] 하이브리드 검색 및 문맥 보강 완료: 최종 {len(merged_context_docs)}개 섹션 구성"
    )

    return {"relevant_docs": merged_context_docs}


class GradeResponse(BaseModel):
    """문서의 질문 관련성 평가 결과"""

    is_relevant: bool = Field(
        description="문서가 질문에 답변하기에 충분한 정보를 포함하고 있는지 여부"
    )
    relevant_entities: list[str] = Field(
        default_factory=list,
        description="질문과 관련된 문서 내 핵심 키워드나 고유 명사 목록 (예: 모델명, 기술 용어)",
    )
    reason: str = Field(
        description="결정에 대한 구체적인 근거 (특히 용어 일치 및 문서 맥락 우선 고려 여부 포함)"
    )


class RewriteResponse(BaseModel):
    """검색 쿼리 재구성 결과"""

    optimized_query: str = Field(description="검색 엔진에 최적화된 새로운 검색어")


async def grade_documents(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """검색된 문서들의 관련성을 LLM으로 평가합니다 (견고한 파싱 적용)."""
    if state.get("is_cached") or state.get("intent") == "general":
        return {}

    docs = state.get("relevant_docs", [])
    if not docs:
        logger.info("[RAG] [GRADE] 문서가 없어 즉시 재구성 단계로 이동")
        return {"intent": "transform"}

    query = state["input"]
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")

    import contextlib
    import json
    import re

    with contextlib.suppress(Exception):
        writer({"status": "문서 관련성 검증 중..."})

    # 상위 3개 문서만 정밀 평가
    test_docs = docs[:3]
    context_text = "\n\n".join(
        [f"DOC {i + 1}: {d.page_content}" for i, d in enumerate(test_docs)]
    )

    grade_prompt = (
        f"{GRADING_CONFIG.get('instruction', '')}\n"
        '출력은 반드시 JSON 형식이어야 합니다. (예: {"is_relevant": true, "relevant_entities": ["A", "B"], "reason": "..."})\n\n'
        f"{GRADING_CONFIG.get('template', '').format(query=query, context_text=context_text)}"
    )

    # 내부 노드 호출용 설정
    call_config = (
        {"configurable": {**cfg, "messages": []}}
        if cfg
        else {"configurable": {"messages": []}}
    )

    try:
        # 1차 시도: 구조화된 출력
        try:
            structured_llm = llm.with_structured_output(GradeResponse)
            result = await structured_llm.ainvoke(grade_prompt, config=call_config)
        except Exception as e:
            logger.debug(f"[RAG] [GRADE] 표준 구조화 출력 실패, 수동 파싱 시도: {e}")
            # 2차 시도: 일반 텍스트 생성 후 JSON 추출
            raw_res = await llm.ainvoke(grade_prompt, config=call_config)
            content = raw_res.content if hasattr(raw_res, "content") else str(raw_res)

            # JSON 패턴 추출 ({...})
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                result = GradeResponse(**data)
            else:
                raise ValueError("JSON 패턴을 찾을 수 없습니다.") from e

        if result.is_relevant:
            logger.info(f"[RAG] [GRADE] 관련성 확인: YES ({result.reason})")
            SessionManager.add_status_log("검색된 지식의 관련성이 확인되었습니다.")
            return {"intent": "generate"}
        else:
            logger.info(f"[RAG] [GRADE] 관련성 확인: NO ({result.reason})")
            SessionManager.add_status_log(
                "검색 결과가 부적합하여 질문 재구성을 시도합니다."
            )
            return {"intent": "transform"}

    except Exception as e:
        logger.warning(f"[RAG] [GRADE] 모든 파싱 시도 실패, 기본값(YES) 적용: {e}")
        return {"intent": "generate"}


async def rewrite_query(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """검색에 더 최적화된 형태로 질문을 재구성합니다 (견고한 파싱 적용)."""
    query = state["input"]
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")

    import contextlib
    import json
    import re

    with contextlib.suppress(Exception):
        writer({"status": "검색 쿼리 최적화 중..."})

    rewrite_prompt = (
        f"{REWRITING_CONFIG.get('instruction', '')}\n"
        '출력은 반드시 JSON 형식이어야 합니다. (예: {"optimized_query": "..."})\n\n'
        f"{REWRITING_CONFIG.get('template', '').format(query=query)}"
    )

    call_config = (
        {"configurable": {**cfg, "messages": []}}
        if cfg
        else {"configurable": {"messages": []}}
    )

    try:
        # 1차 시도: 구조화된 출력
        try:
            structured_llm = llm.with_structured_output(RewriteResponse)
            result = await structured_llm.ainvoke(rewrite_prompt, config=call_config)
        except Exception as e:
            logger.debug(f"[RAG] [REWRITE] 표준 구조화 출력 실패, 수동 파싱 시도: {e}")
            # 2차 시도: 수동 추출
            raw_res = await llm.ainvoke(rewrite_prompt, config=call_config)
            content = raw_res.content if hasattr(raw_res, "content") else str(raw_res)
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                result = RewriteResponse(**data)
            else:
                raise ValueError("JSON 패턴 미검출") from e

        new_query = result.optimized_query.strip().strip('"')
        logger.info(f"[RAG] [REWRITE] 쿼리 재구성: '{query}' -> '{new_query}'")
        return {"search_queries": [new_query], "retry_count": 1}

    except Exception as e:
        logger.warning(f"[RAG] [REWRITE] 모든 시도 실패, 원본 유지: {e}")
        return {"retry_count": 1}


def format_context(docs: list[Document]) -> str:
    """검색된 문서들을 LLM이 읽기 좋은 형식의 문자열로 변환합니다."""
    context = ""
    for _i, d in enumerate(docs):
        section = d.metadata.get("current_section", "일반 본문")
        page = d.metadata.get("page", "?")
        context += f"### [섹션: {section}] (P{page})\n{d.page_content}\n\n"
    return context


async def generate(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """최종 답변을 생성합니다."""
    if state.get("is_cached"):
        return {}

    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        return {"response": "LLM 미로드"}

    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "답변 논리 설계 및 생성 중..."})
    SessionManager.add_status_log("답변 논리 설계 및 생성 시작")

    docs = state.get("relevant_docs", [])
    context = format_context(docs)

    sys_msg = SystemMessage(content="전문 문서 분석가로서 한국어로 답변하세요.")
    human_msg = HumanMessage(
        content=f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{state['input']}"
    )

    from core.model_loader import ModelManager

    full_response = ""
    full_thought = ""
    last_metadata = {}

    async with ModelManager.inference_session():
        async for chunk in llm.astream([sys_msg, human_msg], config=config):
            content_chunk, thought_chunk = llm._convert_chunk_to_thought_and_content(
                chunk
            )
            if thought_chunk:
                full_thought += thought_chunk
                with contextlib.suppress(Exception):
                    writer({"thought": thought_chunk})
            if content_chunk:
                full_response += content_chunk
                with contextlib.suppress(Exception):
                    writer({"content": content_chunk})
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                last_metadata = chunk.response_metadata

    input_tokens = last_metadata.get("prompt_eval_count", 0)
    return {
        "response": full_response,
        "thought": full_thought,
        "performance": {
            **last_metadata,
            "input_token_count": input_tokens,
            "relevant_docs_count": len(docs),
        },
    }


def _merge_adjacent_chunks(
    docs: list[Document], max_tokens: int = 1200
) -> list[Document]:
    """같은 페이지의 연속된 청크들을 하나로 합쳐 풍부한 문맥을 제공합니다 (최적화 버전)."""
    if not docs:
        return []
    if len(docs) == 1:
        return docs

    from common.utils import count_tokens_rough

    merged_docs: list[Document] = []

    # 1. 정렬 (원본 보존을 위해 얕은 복사 리스트 사용)
    # [최적화] deepcopy 대신 얕은 복사 후 필요한 값만 가공
    working_docs = sorted(
        docs,  # 굳이 deepcopy할 필요 없음, 아래에서 필요할 때만 복사
        key=lambda x: (
            str(x.metadata.get("source", "")),
            int(x.metadata.get("page", 0)),
            int(x.metadata.get("chunk_index", 0)),
        ),
    )

    # 2. 병합 루프
    # [최적화] 첫 번째 문서는 필요할 때만 복사
    current_doc = Document(
        page_content=working_docs[0].page_content,
        metadata=copy.copy(working_docs[0].metadata),
    )

    # 미리 토큰 수 계산 (반복 계산 방지)
    current_tokens = count_tokens_rough(current_doc.page_content)

    for next_doc in working_docs[1:]:
        curr_m = current_doc.metadata
        next_m = next_doc.metadata

        # 병합 조건 확인: 동일 페이지이며, 실제 텍스트 오프셋이 인접한지 확인
        # (중복 제거로 인해 인덱스가 벌어져도 실제 텍스트가 연속이면 병합 가능)
        is_same_context = curr_m.get("source") == next_m.get("source") and curr_m.get(
            "page"
        ) == next_m.get("page")

        # [최적화] 섹션 일치 여부 확인 (주제 일관성 보장)
        is_same_section = curr_m.get("current_section") == next_m.get("current_section")

        # [개선] 인덱스가 아닌 실제 텍스트 오프셋(시작/끝) 기반 인접도 체크
        curr_end = curr_m.get("end_index")
        next_start = next_m.get("start_index")

        if curr_end is not None and next_start is not None:
            is_actually_consecutive = abs(next_start - curr_end) <= 5
        else:
            is_actually_consecutive = (
                abs(next_m.get("chunk_index", 0) - curr_m.get("chunk_index", 0)) <= 1
            )

        next_tokens = count_tokens_rough(next_doc.page_content)

        # 병합 결정: 페이지, 오프셋, 섹션이 모두 일치해야 함
        if (
            is_same_context
            and is_actually_consecutive
            and is_same_section
            and (current_tokens + next_tokens + 10) <= max_tokens
        ):
            current_doc.page_content += "\n\n" + next_doc.page_content
            current_tokens += next_tokens + 10  # 구분자 토큰 보정
            # 병합된 청크의 끝 지점 업데이트
            current_doc.metadata["end_index"] = next_m.get("end_index", curr_end)
            # [추가] 다음 청크와 병합을 위해 인덱스 갱신
            current_doc.metadata["chunk_index"] = next_m.get(
                "chunk_index", curr_m.get("chunk_index")
            )
        else:
            merged_docs.append(current_doc)
            # 새로운 기준 문서 설정 (얕은 복사)
            current_doc = Document(
                page_content=next_doc.page_content,
                metadata=copy.copy(next_doc.metadata),
            )
            current_tokens = next_tokens

    merged_docs.append(current_doc)
    return merged_docs


def build_graph() -> Any:
    """Self-Correction 루프가 포함된 그래프를 빌드합니다."""
    workflow = StateGraph(GraphState)

    # 노드 등록
    workflow.add_node("preprocess", preprocess)
    workflow.add_node("retrieve", retrieve_and_rerank)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    # [최적화] generate 노드에 전용 태그 부여하여 스트리밍 필터링 안정성 확보
    workflow.add_node("generate", generate)

    # 엣지 및 조건부 로직 설정
    workflow.add_edge(START, "preprocess")

    workflow.add_conditional_edges(
        "preprocess",
        lambda s: (
            "END"
            if s.get("is_cached")
            else ("generate" if s.get("intent") == "general" else "retrieve")
        ),
        {"END": END, "generate": "generate", "retrieve": "retrieve"},
    )

    workflow.add_edge("retrieve", "grade")

    def decide_to_generate(state: GraphState):
        if state.get("intent") == "generate":
            return "generate"
        if state.get("retry_count", 0) >= 1:
            SessionManager.add_status_log(
                "최대 재시도 횟수에 도달하여 현재 지식으로 답변을 생성합니다."
            )
            return "generate"
        return "rewrite"

    workflow.add_conditional_edges(
        "grade",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    # [최적화] 상태 영속성을 위한 체크포인터 추가
    memory = InMemorySaver()

    # [핵심] 특정 노드에 메타데이터/태그 주입 (컴파일 시점이 아닌 노드 정의 시점 권장이나,
    # 현재 구조에서는 astream에서 필터링할 수 있도록 태그를 명시적으로 관리)
    return workflow.compile(checkpointer=memory)
