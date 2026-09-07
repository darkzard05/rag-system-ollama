"""최종 답변 생성 LangGraph 노드와 컨텍스트 구성 헬퍼 모듈.

``graph_builder.py`` 에서 추출됨 (graph-builder split plan Step 5, pure move);
``graph_builder.py`` 는 하위 호환을 위해 이 이름을 re-export 한다.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from api.schemas import AnswerStructure, GraphState
from common.config import (
    ANALYSIS_PROTOCOL,
    DEFAULT_OLLAMA_MODEL,
    GENERATE_PROMPT_CONFIG,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
    PROMPT_TEMPLATES_CONFIG,
    QUERY_CACHE_ENABLED,
    QUERY_CACHE_MIN_CONF,
    QUERY_CACHE_TTL,
    TOKEN_ESTIMATION_PROXY,
)
from common.utils import count_tokens_rough
from core.graph._glue import _dispatch_event, _get_session_id
from core.graph._grading_glue import (
    _emit_query_timing,
    _enter_stage,
    _stage_timing_var,
)
from core.graph._graph_utils import (
    _doc_stable_id,
    _ensure_query_cache_embedder,
    _sanitize_channel_value,
    get_state_attr,
)
from core.graph._json_utils import (
    _extract_partial_answer,
    _recover_citations,
    _repair_json,
    _strip_json_fence,
)
from core.graph._speculative_gen import _adopt_speculative_generate
from core.model_loader import ModelManager
from core.resource_manager import get_resource_manager
from core.session import SessionManager
from services.monitoring.performance_monitor import OperationType
from services.optimization.caching_optimizer import get_cache_manager

logger = logging.getLogger(__name__)

# 쿼리 응답 캐시 저장값 구조: {"response": str, "confidence": float}
# SemanticCache.get()은 내부 유사도 임계값을 통과한 entry.value(저장값) 또는 None만 반환하며,
# 유사도 점수나 히트 객체를 노출하지 않으므로 신뢰도는 저장 시 함께 직렬화한다.
_QUERY_CACHE_VALUE_VERSION = "1.0"


def format_context(docs: list[Document]) -> str:
    """검색된 문서들을 LLM이 읽기 좋은 형식의 문자열로 변환합니다.

    Phase 1.4: 구조화된 답변을 위한 인용 포맷
    [doc:<stable_id>] [section:X] [page:Y] [score:Z]

    [수정] 인용 토큰은 enumerate 위치(i) 대신 안정 식별자(stable id)를 사용합니다.
    위치 기반 토큰은 rerank/retry 시 문서 순서가 바뀌면 인용이 엉뚱한 청크를
    가리키는 버그를 유발하므로, doc_id(또는 content 해시)를 씁니다.
    """
    context = ""
    for d in docs:
        stable_id = _doc_stable_id(d)
        section = d.metadata.get("current_section", "일반 본문")
        page = d.metadata.get("page", "?")
        score = d.metadata.get("rerank_score", d.metadata.get("score", 0.0))
        context += f"[doc:{stable_id}] [section:{section}] [page:{page}] [score:{score:.3f}]\n{d.page_content}\n\n"
    return context


def _estimate_ctx_tokens(docs: list, query: str) -> int:
    """Estimate prompt tokens for the analysis-context template.

    Uses a SINGLE call on the assembled full-context string so the estimate
    matches the historical semantics (and is robust to call-count-based mocks
    such as a constant ``count_tokens_rough``); the value is the initial `est`
    before per-removal decrements. O(n) overall: the context is assembled
    once here and the guard loop only decrements per removal.

    The context string is assembled inline (mirroring :func:`format_context`)
    rather than by calling the module-level ``format_context``, so this helper
    does not inflate the ``format_context`` call count observed by
    ``_apply_ctx_guard`` (invoked at most twice: pre-guard and post-loop).
    """
    if not docs:
        return count_tokens_rough(
            f"{TOKEN_ESTIMATION_PROXY}\n\n[Context]\n\n[Question]\n{query}"
        )
    context = "".join(
        f"[doc:{_doc_stable_id(d)}] "
        f"[section:{d.metadata.get('current_section', '일반 본문')}] "
        f"[page:{d.metadata.get('page', '?')}] "
        f"[score:{float(d.metadata.get('rerank_score', d.metadata.get('score', 0.0))):.3f}]"
        f"\n{d.page_content}\n\n"
        for d in docs
    )
    return count_tokens_rough(
        f"{TOKEN_ESTIMATION_PROXY}\n\n[Context]\n{context}\n\n[Question]\n{query}"
    )


def _apply_ctx_guard(docs: list, query: str) -> tuple[list, str, int]:
    """Enforce the num_ctx token budget on the retrieval context.

    Returns ``(trimmed_docs, context_str, removed_count)``. Documents are
    dropped from the lowest ``rerank_score`` end of a descending-ranked copy
    until the estimated token cost fits the budget, while always preserving a
    minimum of 2 documents. On each removal the remaining docs are re-formatted
    and the full context string is re-counted, so ``format_context`` is invoked
    at most once per removal (1 + removals, which is bounded by the number of
    documents).
    """
    # Pre-guard context format (initial render of all docs).
    context = format_context(docs) if docs else "일상적인 대화입니다."

    if not docs:
        return docs, context, 0

    token_budget = int((OLLAMA_NUM_CTX - OLLAMA_NUM_PREDICT) * 0.85)
    est = _estimate_ctx_tokens(docs, query)

    removed = 0
    if est > token_budget:
        # rerank_score 내림차순(최상위 문서 우선) 사본에서 낮은 점수 문서부터 제거
        ranked = sorted(
            docs,
            key=lambda d: float(d.metadata.get("rerank_score", 0.0)),
            reverse=True,
        )
        while est > token_budget and len(ranked) > 2:
            ranked.pop()
            removed += 1
            context = format_context(ranked)
            est = count_tokens_rough(
                f"{TOKEN_ESTIMATION_PROXY}\n\n[Context]\n{context}\n\n[Question]\n{query}"
            )
        docs = ranked
        logger.info(f"[RAG] [CTX] trimmed {removed} docs, est tokens={est}")
    return docs, context, removed


# R4-04: 간접 프롬프트 인젝션 패턴 — OWASP RAG Cheat Sheet §3이 스캔을 권장하는
# "SYSTEM:", "INSTRUCTION:", "ignore previous" 계열. SYSTEM/INSTRUCTION은 콜론이
# 뒤따라야만 매칭해 일반 명사("system" 등) 오탐을 줄인다.
_INJECTION_PATTERNS = re.compile(
    r"(?:SYSTEM|INSTRUCTION)\s*:|ignore\s+(?:all\s+)?previous(?:\s+instructions?)?",
    re.IGNORECASE,
)


def _split_injection_docs(
    docs: list[Document],
) -> tuple[list[Document], list[Document]]:
    """검색 청크에서 간접 프롬프트 인젝션 패턴을 스캔해 위험 청크를 분리합니다.

    검색된 원문에 "지시를 무시하고..." 같은 명령이 포함되면 LLM이 데이터를 지시로
    오인할 수 있다. 감지된 청크는 (clean, flagged)로 나누어 컨텍스트에서 제외한다.
    """
    clean: list[Document] = []
    flagged: list[Document] = []
    for d in docs:
        content = d.page_content if isinstance(d.page_content, str) else ""
        if _INJECTION_PATTERNS.search(content):
            flagged.append(d)
        else:
            clean.append(d)
    return clean, flagged


def _coerce_chunk_content(raw: Any) -> str:
    """스트리밍 청크의 content를 항상 문자열로 정규화합니다 (R4-08).

    일부 LLM 벤더(Anthropic 스타일 등)는 content를 복합 콘텐츠 리스트로 반환한다.
    리스트는 텍스트 블록만 병합하고, 그 외 타입은 str()로 폴백한다.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in raw:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(raw)


async def generate(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """최종 답변을 생성합니다."""
    # PHASE 2: If grade_documents pre-started this generate speculatively, adopt
    # the already-warm task (replay its buffered events, return its result) so
    # the grade round-trip is hidden behind generation. Single LLM call, just
    # begun earlier — routing/answer quality unchanged. The speculative instance
    # itself skips adoption via its _is_speculative tag.
    if not getattr(asyncio.current_task(), "_is_speculative", False):
        adoption = _adopt_speculative_generate(config)
        if adoption is not None:
            spec_task, spec_buffer = adoption
            logger.info("[RAG] [SPEC] warm generate 채택 — speculative 결과 반환")
            for ev in spec_buffer:
                await adispatch_custom_event(ev.name, ev.data, config=ev.config)
            return await spec_task

    # [TIMING] 쿼리당 버퍼를 로컬에 캡처 (재시도로 retrieve/grade가 재실행되어도 누적 유지)
    _query_timings = dict(_stage_timing_var.get({}))
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        _emit_query_timing(_query_timings)
        return {"response": "LLM not loaded"}

    if writer is not None:
        await _dispatch_event(
            "graph_status",
            {"status": "답변 설계 및 생성 중..."},
            writer=writer,
            config=config,
        )
    # R1a-08: generate 진입 상태 로그는 단일 호출당 1회만 기록한다.
    # SessionManager.add_status_log(manager.py:381-382)가 연속 동일 로그를 중복 제거하므로
    # 노드 재실행이 발생해도 타임라인에 중복이 쌓이지 않는다. (retrieve/grade 재실행 로그는
    # T1/T4 소유 노드 영역 — 라우팅 결정 시점 로그로 개선하는 것은 별도 워크스트림)
    SessionManager.add_status_log(
        "답변 논리 설계 및 생성 시작", session_id=_get_session_id(config)
    )

    docs = get_state_attr(state, "relevant_docs") or []
    logger.info(f"[RAG] [GENERATE] 관련 문서 수: {len(docs) if docs else 0}")
    if docs:
        for i, d in enumerate(docs):
            logger.info(f"[RAG] [GENERATE] 문서 {i} 길이: {len(d.page_content)}")
    no_info_msg = "제공된 문서에서 질문과 관련된 정보를 찾을 수 없습니다. 다른 질문을 입력하거나 문서 내용을 확인해 주세요."
    if not docs and get_state_attr(state, "intent") != "general":
        logger.info("[RAG] [GENERATE] 관련 문서 없음 -> 사용자 안내 메시지 생성")
        if writer is not None:
            await _dispatch_event(
                "response_chunk", {"content": no_info_msg}, writer=writer, config=config
            )
        _emit_query_timing(_query_timings)
        return {"response": no_info_msg}

    # 쿼리 캐시 단축 경로: preprocess에서 is_cached=True로 라우팅된 경우.
    # LLM 스트리밍을 완전히 건너뛴다. cached path에는 relevant_docs가 없으므로
    # 바로 아래 인젝션/CTX 가드는 사실상 no-op라 안전이 보존된다.
    if get_state_attr(state, "is_cached"):
        cached_response = get_state_attr(state, "cached_response")
        if cached_response:
            logger.info("[RAG] [GENERATE] 캐시 응답 스트리밍 (LLM 미호출)")
            if writer is not None:
                await _dispatch_event(
                    "response_chunk",
                    {"content": cached_response},
                    writer=writer,
                    config=config,
                )
            _emit_query_timing(_query_timings)
            return {"response": cached_response}
        # cached_response가 비어 있으면(오염된 캐시 항목 등) 정상 생성 경로로
        # 폴백하지 않고 빈 컨텍스트로 LLM을 호출하지 않도록 무정보 안내를 반환한다.
        # (Tier 2 방어: 기존 오염 캐시가 있어도 환각 생성 차단)
        logger.warning(
            "[RAG] [GENERATE] 캐시 히트이나 응답이 비어 있음 — 무정보 안내 반환"
        )
        if writer is not None:
            await _dispatch_event(
                "response_chunk",
                {"content": no_info_msg},
                writer=writer,
                config=config,
            )
        _emit_query_timing(_query_timings)
        return {"response": no_info_msg}

    # R4-04: 검색 청크 인젝션 패턴 스캔 — 발견된 청크는 컨텍스트에서 제외하고 경고를 남긴다.
    # 격리 결과도 R1a-03과 동일한 원칙으로 노드 반환을 통해 최종 상태에 반영한다.
    state_docs: list[Document] | None = None
    if docs:
        clean_docs, flagged_docs = _split_injection_docs(docs)
        if flagged_docs:
            logger.warning(
                f"[RAG] [INJECTION] 프롬프트 인젝션 패턴 감지 → "
                f"{len(flagged_docs)}개 청크 격리 (총 {len(docs)}개 중)"
            )
            SessionManager.add_status_log(
                "검색 문서 중 프롬프트 인젝션 패턴이 감지되어 답변 생성에서 제외되었습니다.",
                session_id=_get_session_id(config),
            )
            docs = clean_docs
            state_docs = clean_docs
        if not docs:
            logger.warning(
                "[RAG] [INJECTION] 전체 검색 문서가 인젝션 패턴으로 격리됨 — 안내 메시지 반환"
            )
            if writer is not None:
                await _dispatch_event(
                    "response_chunk",
                    {"content": no_info_msg},
                    writer=writer,
                    config=config,
                )
            _emit_query_timing(_query_timings)
            return {"response": no_info_msg, "relevant_docs": []}

    # [CTX 가드] num_ctx 대비 prompt 추정 토큰이 예산을 초과하면 rerank_score가 낮은
    # 문서부터 제거하여 컨텍스트 초과(overflow)를 방지합니다. (최소 2문서 유지)
    # R4-02: num_predict(출력 예산)를 예약한 후 85%만 입력에 허용한다.
    # 입력+출력 합계가 num_ctx를 넘지 않도록 상한 = (num_ctx - num_predict) * 0.85.
    # O(n^2) 재포맷 방지를 위해 추출한 _apply_ctx_guard에서 per-doc 합산 추정 + 단일
    # 재포맷을 수행한다. (R1a-03 원칙에 따라 트림 결과는 반환 dict로 전달)
    query = get_state_attr(state, "input") or ""
    docs, context, removed = _apply_ctx_guard(docs, query)
    if removed:
        state_docs = docs

    # Phase 1.3: 구조화된 답변 출력을 위한 프롬프트 템플릿 사용
    structured_prompt = PROMPT_TEMPLATES_CONFIG.get(
        "structured_output", ANALYSIS_PROTOCOL
    )
    prompt_version = PROMPT_TEMPLATES_CONFIG.get("version", "1.0")

    # 컨텍스트와 질문을 템플릿에 주입
    query = get_state_attr(state, "input")
    # JSON 스키마의 중호({})가 format() 플레이스홀더로 해석되지 않게 보호
    # {context}와 {query}만 남기고 나머지는 이스케이프
    safe_prompt = structured_prompt.replace("{context}", "{{context}}").replace(
        "{query}", "{{query}}"
    )
    safe_prompt = safe_prompt.replace("{", "{{").replace("}", "}}")
    safe_prompt = safe_prompt.replace("{{context}}", "{context}").replace(
        "{{query}}", "{query}"
    )
    formatted_prompt = safe_prompt.format(context=context, query=query)

    # B8: verify → regenerate 재시도 시 이전 검증 실패 사유를 프롬프트에 주입한다.
    # 사유가 없으면(정상 경로) formatted_prompt를 건드리지 않아 byte-identical 유지.
    verification_issues = get_state_attr(state, "verification_issues", None)
    if verification_issues:
        issue_lines = "\n".join(f"- {issue}" for issue in verification_issues)
        issues_section = (
            "이전 답변이 아래 사유로 검증 실패했습니다. "
            "위 사유를 해결하여 다시 답변하십시오:\n"
            f"{issue_lines}"
        )
        formatted_prompt = f"{formatted_prompt}\n\n[{issues_section}]"

    sys_msg = SystemMessage(content=formatted_prompt)
    human_msg = HumanMessage(
        content=GENERATE_PROMPT_CONFIG.get(
            "human_message",
            "Think step by step, then output ONLY the JSON object.",
        )
    )

    full_response = ""
    full_thought = ""
    last_metadata = {}

    gen_start = time.perf_counter()
    ttft_ms: float | None = None
    _generate_op = _enter_stage(OperationType.LLM_INFERENCE)
    _generate_op.__enter__()

    # 구조화된 출력 모드인지 확인 (PROMPT_TEMPLATES_CONFIG에 structured_output이 있으면 구조화 모드)
    use_structured_output = "structured_output" in PROMPT_TEMPLATES_CONFIG

    async with ModelManager.inference_session():
        # JSON 모드 강제 바인딩(grade 노드 1053행과 동일) — qwen3:4b가 값 내부에
        # 이스케이프 없는 큰따옴표를 넣어 json.loads가 실패(Expecting ',' delimiter)하는
        # 깨진 JSON 출력을 막는다.
        json_llm = llm.bind(response_format={"type": "json_object"})
        coordinator = get_resource_manager()
        async with coordinator.use_llm(model_name=DEFAULT_OLLAMA_MODEL):
            async for chunk in json_llm.astream([sys_msg, human_msg], config=config):
                if hasattr(llm, "_convert_chunk_to_thought_and_content"):
                    content_chunk, thought_chunk = (
                        llm._convert_chunk_to_thought_and_content(chunk)
                    )
                else:
                    # Fallback for LLMs without thought/content separation.
                    # R4-08: content가 리스트(복합 콘텐츠)면 텍스트 블록을 병합해
                    # str+list TypeError를 방지한다 (hasattr 분기 구조는 유지).
                    content_chunk = _coerce_chunk_content(
                        chunk.content if hasattr(chunk, "content") else str(chunk)
                    )
                    thought_chunk = ""
                if content_chunk and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - gen_start) * 1000

                if thought_chunk:
                    full_thought += thought_chunk
                if content_chunk:
                    full_response += content_chunk
                if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                    last_metadata = chunk.response_metadata

                # 원시 JSON 청크도 항상 UI로 전송한다.
                # 구조화 모드(raw_json=True)에서는 파싱 전 원시 토큰을 먼저 띄우고,
                # 파싱 후 final_answer로 대체되도록 한다.
                if (content_chunk or thought_chunk) and writer is not None:
                    await _dispatch_event(
                        "response_chunk",
                        {
                            "content": content_chunk,
                            "thought": thought_chunk,
                            "raw_json": use_structured_output,
                        },
                        writer=writer,
                        config=config,
                    )

    generate_ms = (time.perf_counter() - gen_start) * 1000
    logger.info(
        f"[RAG] [GENERATE][TIMING] generate_ms={generate_ms:.1f} "
        f"ttft_ms={(ttft_ms or 0.0):.1f} output_chars={len(full_response)}"
    )
    _query_timings["generate_total_ms"] = (
        _query_timings.get("generate_total_ms", 0.0) + generate_ms
    )
    _query_timings["ttft_ms"] = _query_timings.get("ttft_ms", 0.0) + float(
        ttft_ms or 0.0
    )
    _generate_op.__exit__(None, None, None)

    # Phase 1.2: 구조화된 답변 파싱 (JSON 출력 기대)
    parsed_answer = None
    parse_failed = False
    json_str = _strip_json_fence(full_response)

    def _extract_partial(json_blob: str) -> dict[str, Any]:
        """깨진 JSON에서 final_answer/citations를 비파괴 복구한다.

        모듈 레벨 escape 인식 헬퍼로 값을 추출해 메타데이터 손실(D1)·본문 절단
        (D2)·답변 절단(D3)·교차 매칭(D4)을 방지한다. 복구 불가능하면 빈 dict 반환.
        """
        recovered: dict[str, Any] = {}
        final_answer = _extract_partial_answer(json_blob)
        if final_answer is not None:
            recovered["final_answer"] = final_answer
        citations = _recover_citations(json_blob)
        if citations:
            recovered["citations"] = citations
        return recovered

    try:
        # P1-4: 깨진 JSON 우선 복구 시도 (값 내부 escape 누락 큰따옴표 보정)
        repaired = _repair_json(json_str)
        if repaired is not None:
            json_str = repaired
        parsed_data = json.loads(json_str)

        # LLM 출력 필드명 매핑: thinking → reasoning (일부 모델 호환성)
        if "thinking" in parsed_data and "reasoning" not in parsed_data:
            parsed_data["reasoning"] = parsed_data.pop("thinking")
            logger.debug("[RAG] [GENERATE] 'thinking' 필드를 'reasoning'으로 매핑함")

        parsed_answer = AnswerStructure(**parsed_data)
        logger.info(
            f"[RAG] [GENERATE] 구조화된 답변 파싱 성공: prompt_version={prompt_version}"
        )

        # 구조화된 출력 모드일 때: 파싱된 final_answer를 UI로 스트리밍
        if (
            use_structured_output
            and parsed_answer
            and parsed_answer.reasoning
            and writer is not None
        ):
            await _dispatch_event(
                "response_chunk",
                {"content": "", "thought": parsed_answer.reasoning},
                writer=writer,
                config=config,
            )
        # P3: 인라인 [doc:...] 폴백과 별도로 citations[] 자체를 스트림에 태워
        # 안정적 doc_id 기반 렌더링을 가능케 한다.
        if parsed_answer and writer is not None:
            await _dispatch_event(
                "citations",
                {"citations": [c.model_dump() for c in parsed_answer.citations]},
                writer=writer,
                config=config,
            )
    except (json.JSONDecodeError, ValueError) as e:
        parse_failed = True
        # 진단: 재현/원인 분석을 위해 원시 출력을 기록 (blind spot 해소)
        logger.warning(
            f"[RAG] [GENERATE] JSON 파싱 실패, 폴백 사용: {e} "
            f"(output_chars={len(full_response)})"
        )
        logger.debug(f"[RAG] [GENERATE] 파싱 실패 원시 출력:\n{json_str}")
        # 강건화: 깨진 JSON에서 final_answer/citations 부분 복구 시도
        recovered = _extract_partial(json_str)
        if recovered:
            cites = recovered.get("citations", [])
            # P3-1: 복구 완전성 가시화 — 메타데이터 보존 여부를 명시
            meta_preserved = sum(
                1
                for c in cites
                if c.get("section") != "" or c.get("page") != 0 or c.get("score") != 0.0
            )
            logger.info(
                f"[RAG] [GENERATE] 부분 복구 성공: "
                f"final_answer={'Y' if recovered.get('final_answer') else 'N'}, "
                f"citations={len(cites)} (메타데이터 보존 {meta_preserved}/{len(cites)})"
            )
            if cites and meta_preserved < len(cites):
                logger.warning(
                    "[RAG] [GENERATE] citation 메타데이터 일부 누락 — "
                    "하이라이팅 좌표 정확도 저하 가능"
                )
        parsed_answer = AnswerStructure(
            reasoning=full_thought or "추론 과정 파싱 실패",
            final_answer=recovered.get("final_answer", full_response),
            citations=recovered.get("citations", []),
            confidence=0.5,
        )

    input_tokens = last_metadata.get("prompt_eval_count", 0)
    output_tokens = last_metadata.get("eval_count", 0)
    result: dict[str, Any] = {
        "response": parsed_answer.final_answer if parsed_answer else full_response,
        "thought": parsed_answer.reasoning if parsed_answer else full_thought,
        "citations": [c.model_dump() for c in parsed_answer.citations]
        if parsed_answer
        else [],
        "confidence": parsed_answer.confidence if parsed_answer else 0.5,
        "prompt_version": prompt_version,
        "parse_failed": parse_failed,
        # R1a-05: response_metadata 등 임의 객체가 섞이면 체크포인트 전체가 pickle로
        # 강등되는 경로를 차단하기 위해 순수 타입만 저장한다.
        "performance": _sanitize_channel_value(
            {
                **last_metadata,
                "input_token_count": input_tokens,
                "output_token_count": output_tokens,
                "relevant_docs_count": len(docs),
                "ttft_ms": ttft_ms or 0.0,
                "generate_ms": generate_ms,
            }
        ),
    }
    # R1a-03: CTX 트림/인젝션 격리로 LLM이 실제로 본 문서 목록이 상태와 다르면
    # 반환을 통해 overwrite 리소스로 최종 상태에 반영한다. 변경이 없으면 건드리지
    # 않아 retrieve가 전달한 기존 상태를 보존한다.
    if state_docs is not None:
        result["relevant_docs"] = state_docs

    # 쿼리 응답 캐시 저장 (선택적 opt-in). 캐시 히트 경로가 아니라 이번에 새로
    # 생성한 경우에만 저장한다. 신뢰도가 기준 미달이면 저장하지 않는다.
    if QUERY_CACHE_ENABLED and not get_state_attr(state, "is_cached"):
        query_for_cache = get_state_attr(state, "input", "") or ""
        conf = float(result.get("confidence", 0.0))
        if query_for_cache and conf >= QUERY_CACHE_MIN_CONF:
            sid = _get_session_id(config)
            if bool(SessionManager.get("file_hash", session_id=sid, default=None)):
                await _ensure_query_cache_embedder()
                try:
                    await get_cache_manager().set(
                        query_for_cache,
                        {
                            "response": result["response"],
                            "confidence": conf,
                            "version": _QUERY_CACHE_VALUE_VERSION,
                        },
                        ttl_seconds=QUERY_CACHE_TTL,
                        use_semantic=True,
                        persist_to_disk=False,
                    )
                except Exception as e:  # noqa: BLE001 - 캐시 저장 실패는 무시
                    logger.warning(f"[RAG] [CACHE] 저장 실패 — 무시: {e}")

    _emit_query_timing(_query_timings)
    return result
