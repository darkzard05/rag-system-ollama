"""
LangGraph를 사용하여 단순화된 RAG 워크플로우를 구성합니다.
의도 분류, 캐시 확인, 하이브리드 검색, 생성의 핵심 단계를 직선화합니다.
"""

import asyncio
import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter

from api.schemas import AggregatedSearchResult, GraphState
from cache.response_cache import get_response_cache
from common.config import (
    ANALYSIS_PROTOCOL,
    RERANKER_CONFIG,
)
from common.utils import fast_hash
from core.session import SessionManager

logger = logging.getLogger(__name__)


async def preprocess(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """의도 분류 및 캐시 확인을 동시에 수행합니다."""
    query = state["input"].strip()

    import contextlib

    # [표준] StreamWriter 사용 (Python 3.10 호환성 방어적 처리)
    with contextlib.suppress(Exception):
        writer({"status": "의도 분석 및 지식 확인 중..."})

    # 1. 의도 분류 (단순 규칙)
    greetings = ["안녕", "hi", "hello", "도움말", "고마워"]
    if any(g in query.lower() for g in greetings) or len(query) < 2:
        return {"intent": "general", "is_cached": False}

    # 2. 캐시 확인
    cache = get_response_cache()
    cached_res = await cache.get(query, use_semantic=True)
    if cached_res:
        SessionManager.add_status_log("캐시된 답변을 발견했습니다.")
        return {
            "response": cached_res.response,
            "thought": cached_res.metadata.get("thought", ""),
            "is_cached": True,
        }

    return {"intent": "rag", "is_cached": False}


async def retrieve_and_rerank(
    state: GraphState, config: RunnableConfig, writer: StreamWriter
) -> dict[str, Any]:
    """문서 검색 및 재순위화를 수행합니다."""
    if state.get("is_cached") or state.get("intent") == "general":
        return {}

    from core.search_aggregator import AggregationStrategy, SearchResultAggregator

    query = state["input"]
    cfg = config.get("configurable", {})

    # [표준] StreamWriter 사용 (Python 3.10 호환성 방어적 처리)
    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "관련 지식 탐색 중..."})
    SessionManager.add_status_log("문서 저장소에서 관련 지식 탐색 시작")

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
    weights = {
        "bm25": ENSEMBLE_WEIGHTS[0],
        "faiss": ENSEMBLE_WEIGHTS[1],
    }
    aggregated, _ = aggregator.aggregate_results(
        {"all": all_docs},
        strategy=AggregationStrategy.WEIGHTED_RRF,
        top_k=25,
        weights=weights,
    )

    final_docs = [
        Document(page_content=r.content, metadata=r.metadata) for r in aggregated
    ]
    SessionManager.add_status_log(
        f"하이브리드 검색 완료 ({len(final_docs)}개 후보 확보)"
    )

    # [최적화] 적응형 리랭킹 전략 (Adaptive Reranking)
    # 1. 후보군이 너무 적거나 (3개 이하)
    # 2. 최상위 결과가 압도적으로 우수한 경우 리랭킹 생략 (TTFT 절약)
    skip_rerank = False
    if len(final_docs) <= 3:
        skip_rerank = True
        SessionManager.add_status_log("후보 문서가 적어 리랭킹을 생략합니다.")
    elif len(aggregated) > 0 and aggregated[0].aggregated_score > 0.85:
        skip_rerank = True
        SessionManager.add_status_log(
            "명확한 검색 결과가 확인되어 리랭킹을 생략합니다."
        )

    # [최적화] 리랭킹 전 문맥 병합 (Pre-Rerank Merging)
    merged_candidates = _merge_consecutive_chunks(final_docs, max_tokens=1024)
    if len(merged_candidates) < len(final_docs):
        SessionManager.add_status_log(
            f"지능형 문맥 병합 완료: {len(final_docs)}개 파편 → {len(merged_candidates)}개 섹션으로 통합"
        )

    # FlashRank 리랭킹 (선택적 실행)
    if (
        RERANKER_CONFIG.get("enabled", True)
        and not skip_rerank
        and len(merged_candidates) > 1
    ):
        import contextlib

        with contextlib.suppress(Exception):
            writer({"status": "지식 우선순위 정제 중 (FlashRank)"})
        SessionManager.add_status_log("지식의 우선순위 재조정 및 정제 중 (FlashRank)")
        from flashrank import RerankRequest

        from core.model_loader import ModelManager

        ranker = await ModelManager.get_flashranker()
        passages = [
            {"id": i, "text": d.page_content, "meta": d.metadata}
            for i, d in enumerate(merged_candidates)
        ]

        async with ModelManager.inference_session():
            results = await asyncio.to_thread(
                ranker.rerank, RerankRequest(query=query, passages=passages)
            )
            # 상위 10개에서 15개로 약간 확장 (병합된 청크는 정보량이 많음)
            final_docs = [
                Document(page_content=r["text"], metadata=r["meta"])
                for r in results[:15]
            ]

            SessionManager.add_status_log(f"최적의 지식 {len(final_docs)}개 선별 완료")
    else:
        final_docs = merged_candidates

    return {"relevant_docs": final_docs}


def format_context(docs: list[Document]) -> str:
    """검색된 문서들을 LLM이 읽기 좋은 형식의 문자열로 변환합니다."""
    context = ""
    for i, d in enumerate(docs):
        page = d.metadata.get("page", "?")
        context += f"### [자료 {i + 1}] (P{page})\n{d.page_content}\n\n"
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

    # [표준] StreamWriter 사용 (Python 3.10 호환성 방어적 처리)
    import contextlib

    with contextlib.suppress(Exception):
        writer({"status": "답변 논리 설계 및 생성 중..."})
    SessionManager.add_status_log("답변 논리 설계 및 생성 시작")

    # 컨텍스트 포맷팅
    docs = state.get("relevant_docs", [])
    context = format_context(docs)

    # 프롬프트 구성
    sys_msg = SystemMessage(content="전문 문서 분석가로서 한국어로 답변하세요.")
    human_msg = HumanMessage(
        content=f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{state['input']}"
    )

    # [최적화] astream을 사용한 실시간 사고 과정 및 답변 추출
    from core.model_loader import ModelManager

    full_response = ""
    full_thought = ""
    last_metadata = {}

    async with ModelManager.inference_session():
        # config를 전달하여 스트리밍 컨텍스트(messages, custom 등)를 유지함
        async for chunk in llm.astream([sys_msg, human_msg], config=config):
            # 1. 커스텀 래퍼를 통해 사고 과정과 답변 분리
            content_chunk, thought_chunk = llm._convert_chunk_to_thought_and_content(
                chunk
            )

            # 2. 사고 과정 실시간 전달
            if thought_chunk:
                full_thought += thought_chunk
                with contextlib.suppress(Exception):
                    writer({"thought": thought_chunk})

            # 3. 답변 본문 실시간 전달
            if content_chunk:
                full_response += content_chunk
                with contextlib.suppress(Exception):
                    writer({"content": content_chunk})

            # 4. 메타데이터 수집 (마지막 청크에 포함됨)
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                last_metadata = chunk.response_metadata

    # 결과 반환 (누적된 데이터로 Graph State 업데이트)
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


def _merge_consecutive_chunks(
    docs: list[Document], max_tokens: int = 1200
) -> list[Document]:
    """
    같은 페이지의 연속된 청크들을 하나로 합쳐 풍부한 문맥을 제공합니다.
    리랭킹 전에 호출될 경우 리랭커에게 더 넓은 문맥을 제공합니다.
    """
    if not docs:
        return []

    from common.utils import count_tokens_rough

    merged_docs: list[Document] = []
    import copy

    # 원본 보호를 위한 복사 및 정렬 (소스/페이지/인덱스 순)
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

        # 같은 소스, 같은 페이지 확인
        is_same_context = curr_m.get("source") == next_m.get("source") and curr_m.get(
            "page"
        ) == next_m.get("page")

        # 인덱스 연속성 확인
        is_consecutive = True
        if "chunk_index" in curr_m and "chunk_index" in next_m:
            # 리트리버가 건너뛴 청크가 있더라도 인접하면 병합 허용 (차이가 2 이내)
            is_consecutive = abs(next_m["chunk_index"] - curr_m["chunk_index"]) <= 2

        # 토큰 길이 제한 확인
        current_tokens = count_tokens_rough(
            current_doc.page_content + next_doc.page_content
        )
        within_limit = current_tokens <= max_tokens

        if is_same_context and is_consecutive and within_limit:
            # 중복된 문장이 시작/끝에 있을 수 있으므로 처리 (간단한 결합)
            current_doc.page_content += "\n\n" + next_doc.page_content
            # 메타데이터 업데이트 (범위 표시 등)
            if "chunk_index_range" not in current_doc.metadata:
                current_doc.metadata["chunk_index_range"] = [
                    curr_m.get("chunk_index"),
                    next_m.get("chunk_index"),
                ]
            else:
                current_doc.metadata["chunk_index_range"][1] = next_m.get("chunk_index")
        else:
            merged_docs.append(current_doc)
            current_doc = next_doc

    merged_docs.append(current_doc)
    return merged_docs


def build_graph() -> Any:
    """그래프를 빌드하고 컴파일합니다."""
    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess", preprocess)
    workflow.add_node("retrieve", retrieve_and_rerank)
    workflow.add_node("generate", generate)

    workflow.add_edge(START, "preprocess")

    # 조건부 에지: 캐시가 있으면 바로 종료, 일상 대화면 바로 생성, 아니면 검색
    workflow.add_conditional_edges(
        "preprocess",
        lambda s: (
            "END"
            if s.get("is_cached")
            else ("generate" if s.get("intent") == "general" else "retrieve")
        ),
        {"END": END, "generate": "generate", "retrieve": "retrieve"},
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()
