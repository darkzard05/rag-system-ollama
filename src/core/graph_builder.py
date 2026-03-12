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

    # 초기 상태 설정 (리듀서 필드는 초기값 불필요, 명시적 설정만 수행)
    update = {
        "intent": "rag",
        "is_cached": False,
    }

    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "의도 분석 및 지식 확인 중..."})

    # 1. 의도 분류 (단순 규칙)
    greetings = ["안녕", "hi", "hello", "도움말", "고마워"]
    if any(g in query.lower() for g in greetings) or len(query) < 2:
        logger.info("[RAG] [PREPROCESS] 일상 대화 또는 짧은 질의로 판단")
        update["intent"] = "general"
        return update

    # 2. 캐시 확인
    cache = get_response_cache()
    cached_res = await cache.get(query, use_semantic=True)
    if cached_res:
        logger.info("[RAG] [PREPROCESS] 시맨틱 캐시 히트")
        SessionManager.add_status_log("캐시된 답변을 발견했습니다.")
        return {
            "response": cached_res.response,
            "thought": cached_res.metadata.get("thought", ""),
            "is_cached": True,
        }

    return update


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

    # 병렬 검색
    tasks = []
    if bm25:
        tasks.append(asyncio.create_task(bm25.ainvoke(query)))
    if faiss:
        tasks.append(asyncio.create_task(faiss.ainvoke(query)))

    results = await asyncio.gather(*tasks) if tasks else [[], []]
    logger.debug(
        f"[RAG] [RETRIEVE] 검색 결과 확보 (BM25: {len(results[0])}, Vector: {len(results[1])})"
    )

    # 결과 병합 및 RRF 집계
    all_docs = []
    for i, res in enumerate(results):
        source = "bm25" if i == 0 else "faiss"
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
    weights = {"bm25": ENSEMBLE_WEIGHTS[0], "faiss": ENSEMBLE_WEIGHTS[1]}
    aggregated, _ = aggregator.aggregate_results(
        {"all": all_docs},
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=25,
        weights=weights,
    )

    final_docs = [
        Document(page_content=r.content, metadata=r.metadata) for r in aggregated
    ]

    if not final_docs:
        q_len = len(query) if query else 0
        logger.warning(
            f"[RAG] [RETRIEVE] 검색 결과가 전혀 없습니다 (Query Length: {q_len})"
        )
        SessionManager.add_status_log("검색된 문서가 없습니다.")
        return {"relevant_docs": []}

    # [최적화] 리랭킹 효율화: 병합 전 원본 청크들로 먼저 정밀 리랭킹 수행
    # (텍스트 양이 적어 리랭커 속도가 향상되고 점수 희석 방지)
    from core.reranker import DistributedReranker, RerankerStrategy

    reranker = DistributedReranker()
    # 후보군(Top 25)에 대해 즉시 리랭킹 수행
    ranked_docs, _ = reranker.rerank(
        final_docs, query_text=query, strategy=RerankerStrategy.SEMANTIC_FLASH
    )
    logger.info(
        f"[RAG] [RETRIEVE] 리랭킹 선별 완료: {len(final_docs)}개 후보 중 최상위 선별"
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
    """검색된 문서들의 관련성을 LLM으로 평가합니다."""
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

    with contextlib.suppress(Exception):
        writer({"status": "문서 관련성 검증 중..."})

    # 상위 3개 문서만 정밀 평가 (속도 향상)
    test_docs = docs[:3]
    context_text = "\n\n".join(
        [f"DOC {i + 1}: {d.page_content}" for i, d in enumerate(test_docs)]
    )

    # 설정 파일에서 프롬프트 로드
    grade_prompt = (
        f"{GRADING_CONFIG.get('instruction', '')}\n\n"
        f"{GRADING_CONFIG.get('template', '').format(query=query, context_text=context_text)}"
    )

    try:
        # [최적화] 구조화된 출력(Structured Output) 적용
        structured_llm = llm.with_structured_output(GradeResponse)
        # 내부 노드 호출 시 히스토리 미사용 (얕은 복사 후 특정 필드만 교체하여 pickle 에러 방지)
        call_config = (
            {"configurable": {**cfg, "messages": []}}
            if cfg
            else {"configurable": {"messages": []}}
        )

        result = await structured_llm.ainvoke(grade_prompt, config=call_config)

        if result.is_relevant:
            logger.info(
                f"[RAG] [GRADE] 관련성 확인 결과: YES ({result.reason}) | 키워드: {result.relevant_entities}"
            )
            SessionManager.add_status_log("검색된 지식의 관련성이 확인되었습니다.")
            return {"intent": "generate"}
        else:
            logger.info(
                f"[RAG] [GRADE] 관련성 확인 결과: NO ({result.reason}) | 키워드: {result.relevant_entities}"
            )
            SessionManager.add_status_log(
                "검색 결과가 부적합하여 질문 재구성을 시도합니다."
            )
            return {"intent": "transform"}

    except Exception as e:
        logger.warning(f"[RAG] [GRADE] 구조화된 출력 실패, 기본값(YES) 적용: {e}")
        return {"intent": "generate"}


async def rewrite_query(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """검색에 더 최적화된 형태로 질문을 재구성합니다."""
    query = state["input"]
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")

    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "검색 쿼리 최적화 중..."})

    # 설정 파일에서 프롬프트 로드
    rewrite_prompt = (
        f"{REWRITING_CONFIG.get('instruction', '')}\n\n"
        f"{REWRITING_CONFIG.get('template', '').format(query=query)}"
    )

    try:
        # [최적화] 구조화된 출력(Structured Output) 적용
        structured_llm = llm.with_structured_output(RewriteResponse)
        # 내부 노드 호출 시 히스토리 미사용
        call_config = (
            {"configurable": {**cfg, "messages": []}}
            if cfg
            else {"configurable": {"messages": []}}
        )

        result = await structured_llm.ainvoke(rewrite_prompt, config=call_config)
        new_query = result.optimized_query.strip().strip('"')

        logger.info(f"[RAG] [REWRITE] 쿼리 재구성 완료: '{query}' -> '{new_query}'")
        # [리듀서] 리스트를 반환하면 자동으로 기존 리스트에 추가됨
        return {"search_queries": [new_query], "retry_count": 1}

    except Exception as e:
        logger.warning(f"[RAG] [REWRITE] 구조화된 출력 실패, 원본 유지: {e}")
        # 실패 시에도 재시도 횟수만 1 증가
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
    """같은 페이지의 연속된 청크들을 하나로 합쳐 풍부한 문맥을 제공합니다."""
    if not docs:
        return []

    from common.utils import count_tokens_rough

    merged_docs: list[Document] = []

    working_docs = sorted(
        [copy.deepcopy(d) for d in docs],
        key=lambda x: (
            str(x.metadata.get("source", "")),
            int(x.metadata.get("page", 0)),
            int(x.metadata.get("chunk_index", 0)),
        ),
    )

    current_doc = working_docs[0]
    for next_doc in working_docs[1:]:
        curr_m = current_doc.metadata
        next_m = next_doc.metadata
        is_same_context = curr_m.get("source") == next_m.get("source") and curr_m.get(
            "page"
        ) == next_m.get("page")
        is_consecutive = (
            abs(next_m.get("chunk_index", 0) - curr_m.get("chunk_index", 0)) <= 2
        )
        current_tokens = count_tokens_rough(
            current_doc.page_content + next_doc.page_content
        )
        if is_same_context and is_consecutive and current_tokens <= max_tokens:
            current_doc.page_content += "\n\n" + next_doc.page_content
        else:
            merged_docs.append(current_doc)
            current_doc = next_doc
    merged_docs.append(current_doc)
    return merged_docs


def build_graph() -> Any:
    """Self-Correction 루프가 포함된 그래프를 빌드합니다."""
    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("retrieve", retrieve_and_rerank)
    workflow.add_node("grade", grade_documents)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("generate", generate)

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

    return workflow.compile()
