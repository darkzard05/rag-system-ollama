"""
LangGraph 기반 자가 교정(Self-Correction) RAG 워크플로우 배선(wiring) 모듈.

6개 LangGraph 노드(preprocess, retrieve_and_rerank, grade_documents, rewrite_query,
generate, verify_answer)는 ``core.graph`` 패키지의 개별 모듈로 분리되었습니다.
이 모듈은 이제 노드들을 조립하는 ``build_graph()`` 와, 분리 이전 import 경로
(``core.graph.graph_builder.<symbol>``)에 대한 하위 호환 재수출(re-export)만
유지합니다. 모듈 배치 및 제거된 재수출 목록은 아래 주석을 참고.
"""

import logging
from typing import Any

from langchain_core.callbacks.manager import (
    adispatch_custom_event,  # noqa: F401 — re-exports for backward compat
)
from langgraph.graph import END, START, StateGraph

from api.schemas import GraphState
from core.graph._generate import (  # noqa: F401 — re-exports for backward compat
    QUERY_CACHE_ENABLED,
    _apply_ctx_guard,
    _split_injection_docs,
    format_context,
    generate,
)
from core.graph._grade import (  # noqa: F401 — re-exports for backward compat
    grade_documents,
    rewrite_query,
)
from core.graph._grading_glue import (  # noqa: F401 — re-exports for backward compat
    _add_stage_ms,
    _reset_stage_timings,
    _stage_timing_var,
)
from core.graph._graph_cache import (  # noqa: F401 — re-exports for backward compat
    _GRAPH_CACHE_KEY,
    _graph_cache,
    _graph_object_cache,
    delete_graph_thread,
    invalidate_graph_cache,
)
from core.graph._graph_utils import (
    _doc_stable_id,  # noqa: F401 — re-exports for backward compat
    _sanitize_channel_value,  # noqa: F401 — re-exports for backward compat
    get_state_attr,
)
from core.graph._preprocess import preprocess  # noqa: F401 — re-exports (back-compat)
from core.graph._retrieve import (  # noqa: F401 — re-exports for backward compat
    _MIN_CONTEXT_SECTION_LEN,
    _filter_min_section_len,
    _merge_adjacent_chunks,
    retrieve_and_rerank,
)
from core.graph._speculative_gen import (  # noqa: F401 — re-exports for backward compat
    _spec_registry,
    _SpecGenerate,
)
from core.graph._verify import (  # noqa: F401 — re-exports for backward compat
    _validate_cited_doc_ids,
    verify_answer,
)
from core.session import SessionManager  # noqa: F401 — re-exports for backward compat

logger = logging.getLogger(__name__)

# ============================================================================
# Module layout (graph-builder split)
# ----------------------------------------------------------------------------
# The LangGraph nodes/helpers now live in ``core.graph`` submodules. The imports
# above are re-exported for backward compatibility with the pre-split import
# surface (``core.graph.graph_builder.<symbol>``); every re-exported name is
# consumed by src/, tests/, or scripts/ (verified repo-wide — pruning/batch-2).
# build_graph() itself only uses the node functions, _graph_cache, and
# get_state_attr.
#
# Removed in earlier pruning batches (zero consumers, unused in build_graph):
#   _glue helpers, _grading_glue timing emitters, _grade memo helpers,
#   _graph_cache internals, _json_utils, speculative-overlap control names
#   (MAX_CONCURRENT_INFERENCE, _adopt/_cancel/_replay_spec_events,
#   _spec_generate_events, _spec_overlap_enabled, _SpecEvent), _RE_VERIFY_DOC_CITATION
#   and _coerce_chunk_content/_estimate_ctx_tokens.
#
# Speculative-overlap safety (IMPORTANT): speculative generate events are buffered
# via a ContextVar and are NEVER surfaced on the transform route; they only overlap
# when MAX_CONCURRENT_INFERENCE > 1.
# ============================================================================


async def build_graph() -> Any:
    """자가 교정형 RAG 워크플로우를 구성합니다.

    [최적화] asyncio.Lock을 사용하여 동시 빌드 요청 시 이중 컴파일을 방지합니다.
    """
    logger.info("[RAG] [GRAPH] build_graph() 호출됨")
    if _graph_cache.compiled is not None:
        logger.info("[RAG] [GRAPH] 캐시된 그래프 반환")
        return _graph_cache.compiled

    lock = _graph_cache.get_lock()
    async with lock:
        # Double-check after acquiring lock (다른 세션이 이미 빌드 완료했을 수 있음)
        if _graph_cache.compiled is not None:
            logger.info("[RAG] [GRAPH] 캐시 획득 후 캐시된 그래프 반환")
            return _graph_cache.compiled

        logger.info("[RAG] [GRAPH] 그래프 재구성 시작")
        workflow = StateGraph(GraphState)

        # 노드 등록
        workflow.add_node("preprocess", preprocess)
        workflow.add_node("retrieve", retrieve_and_rerank)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("rewrite_query", rewrite_query)
        workflow.add_node("generate", generate)

        # 엣지 설정
        workflow.add_edge(START, "preprocess")

        workflow.add_conditional_edges(
            "preprocess",
            lambda s: (
                "generate" if get_state_attr(s, "intent") == "general" else "retrieve"
            ),
            {"generate": "generate", "retrieve": "retrieve"},
        )

        workflow.add_edge("retrieve", "grade_documents")

        workflow.add_conditional_edges(
            "grade_documents",
            # R1a-06: 라우팅은 intent(분류 의미)가 아닌 route(전용 라우팅 채널)로 결정.
            # route 미설정 시 명시적 기본값 generate (grade가 항상 route를 반환하므로 방어적 폴백).
            lambda s: get_state_attr(s, "route", "generate"),
            {"generate": "generate", "transform": "rewrite_query"},
        )

        workflow.add_edge("rewrite_query", "retrieve")

        # Phase 1.5: 검증 노드 추가 (기능 플래그로 제어)
        from common.config import VERIFICATION_ENABLED

        if VERIFICATION_ENABLED:
            workflow.add_node("verify_answer", verify_answer)
            workflow.add_edge("generate", "verify_answer")
            # verify_answer에서 END 또는 regenerate로 라우팅
            workflow.add_conditional_edges(
                "verify_answer",
                lambda s: get_state_attr(s, "verification_route", "end"),
                {"end": END, "regenerate": "generate"},
            )
        else:
            workflow.add_edge("generate", END)

        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

        # R1a-05: pickle_fallback=False — msgpack 직렬화 불가 객체는 조용히 pickle로
        # 강등하지 않고 명시적 예외를 발생시킨다. 상태 채널은 _sanitize_channel_value로
        # 순수 타입만 저장되도록 위생화한다. InMemorySaver는 실험/테스트용 저장소이므로
        # 운영 전환 시 퇴거 정책이 있는 영속 체크포인터로 교체해야 한다.
        # P4: allowed_json_modules 명시 — langchain_core.messages 타입(상태에 저장될 수
        # 있는 객체형)만 reviver allowlist에 올린다. 순수 타입(int/str/float/bool/None/
        # list/dict)은 생성자 재구성이 필요 없어 allowlist 대상이 아니다.
        memory = InMemorySaver(
            serde=JsonPlusSerializer(
                pickle_fallback=False,
                allowed_json_modules=[
                    ("langchain_core", "messages", "HumanMessage"),
                    ("langchain_core", "messages", "AIMessage"),
                    ("langchain_core", "messages", "SystemMessage"),
                ],
            )
        )

        _graph_cache.compiled = workflow.compile(checkpointer=memory)
        # R1a-02/R1b-02: 세션 삭제 시 delete_graph_thread가 해당 thread를 정리할 수
        # 있도록 saver를 전역에서 참조 가능하게 노출한다.
        _graph_cache.checkpointer = memory
        return _graph_cache.compiled
