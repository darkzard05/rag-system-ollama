"""채점·검증 노드 단일 진원 (Phase 2A 병합).

구 ``_grade`` (grade_documents + rewrite_query + grading-memo 헬퍼)와
구 ``_verify`` (verify_answer + 인용 검증 + 결정적 샘플링)를 흡수했다.
내부 로직은 변경 없이 이동했으며 — 특히 grade short-circuit의
``min_score_to_skip``/``min_top_gap_to_skip`` 분기는 byte-identical —
구경로는 alias shim으로 유지한다.
"""

import hashlib
import logging
import re
import sys
import time
from typing import Any

from langchain_core.callbacks.manager import adispatch_custom_event
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field

from api.schemas import GraphState
from common.config import (
    DEFAULT_OLLAMA_MODEL,
    GRADE_PROMPT_CONFIG,
    GRADING_CONFIG,
    GRADING_ENABLED,
    VERIFICATION_ENABLED,
    VERIFICATION_SAMPLE_RATE,
    VERIFY_PROMPT_CONFIG,
)
from common.constants import GRADE_MEMO_KEY
from common.utils import fast_hash
from core.graph._generate import format_context
from core.graph._graph_core import (
    _add_stage_ms,
    _doc_stable_id,
    _enter_stage,
    _get_session_id,
    _safe_invoke,
    _start_speculative_generate,
    _strip_json_fence,
    get_state_attr,
)
from core.graph._speculative_gen import _cancel_speculative_generate
from core.model_loader import ModelManager
from core.resource_manager import get_resource_manager
from core.session import SessionManager
from services.monitoring.performance_monitor import OperationType

logger = logging.getLogger(__name__)


# ============================================================================
# _grade 섹션 — grade_documents + rewrite_query + grading-memo 헬퍼 (pure move)
# ============================================================================


# ----------------------------------------------------------------------------
# Wave 3 grade reduction — memoization contract (T8).
# 공용 키는 ``common.constants.GRADE_MEMO_KEY``에서 단일 정의로 공유한다.
# pipeline_builder는 동일 키로 무효화, 본 모듈은 저장만 담당한다.
# (이전에는 순환 import 회피를 위해 문자열 리터럴을 중복 정의했으나,
#  leaf 모듈인 common.constants로 이동해 단일 정의로 통합했다.)
_GRADE_MEMO_KEY = GRADE_MEMO_KEY


def _grade_memo_key(state: Any, docs: list) -> str:
    """안정 해시로 메모 키를 생성합니다 (query + 정렬된 doc_id 집합).

    동일 (query, doc-set) 조합이면 항상 동일 키 → 같은 세션 내 반복 질의 시
    LLM 호출을 건너뛰고 이전 판단을 재사용한다. doc_id가 하나라도 바뀌면
    해시가 달라져(staleness miss) 오래된 메모가 재사용되지 않는다.
    """
    query = get_state_attr(state, "input") or ""
    # 재구성된 쿼리(search_queries[-1])가 실제 평가에 쓰인 경우 그걸 사용.
    search_queries = get_state_attr(state, "search_queries")
    if search_queries:
        query = search_queries[-1]
    doc_ids = sorted(_doc_stable_id(d) for d in docs)
    return fast_hash(f"GRADE|{query}|" + "|".join(doc_ids))


def _store_grade_memo(state: Any, docs: list, decision: dict, sid: str) -> None:
    """실제 LLM 판단을 세션 메모(dict)에 저장합니다.

    실패해도 채점 흐름을 깨뜨리지 않도록 내부에서 흡수한다 (최대한 best-effort).
    저장 실패 시 다음 반복에서 단순히 miss → LLM 경로로 폴백한다.
    """
    try:
        memo_key = _grade_memo_key(state, docs)
        existing = SessionManager.get(_GRADE_MEMO_KEY, session_id=sid, default={})
        if not isinstance(existing, dict):
            existing = {}
        existing[memo_key] = decision
        SessionManager.set(_GRADE_MEMO_KEY, existing, session_id=sid)
    except Exception as e:  # noqa: BLE001 - 메모 저장 실패는 치명적이지 않음
        logger.debug(f"[RAG] [GRADE] 메모 저장 실패 (무시): {e}")


class UnifiedGradeRewriteResponse(BaseModel):
    """문서 평가 + 쿼리 재구성을 단일 호출로 통합."""

    action: str = Field(description="'generate' 또는 'rewrite'")
    is_relevant: bool = Field(
        description="문서가 질문에 답변하기에 충분한 정보를 포함하고 있는지 여부"
    )
    relevant_entities: list[str] = Field(
        default_factory=list,
        description="질문과 관련된 문서 내 핵심 키워드나 고유 명사 목록",
    )
    reason: str = Field(description="결정에 대한 구체적인 근거")
    optimized_query: str | None = Field(
        default=None,
        description="rewrite 시에만 채울 최적화된 검색어",
    )


async def grade_documents(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """검색된 문서들의 관련성을 LLM으로 평가하고, 부적절 시 재검색 쿼리를 동시에 생성합니다."""
    # 운영자 제어 채점 단계 opt-out (GRADING_ENABLED=False): LLM 호출 없이
    # 즉시 generate로 직행. T8 메모이제이션/is_cached 단축 경로보다 우선.
    if not GRADING_ENABLED:
        logger.info("[RAG] [GRADE] 채점 단계 비활성화 — generate로 직행")
        return {"intent": "generate", "route": "generate"}

    if (
        get_state_attr(state, "is_cached")
        or get_state_attr(state, "intent") == "general"
    ):
        # R1a-06: 라우팅은 intent가 아닌 route 채널로 결정하므로 여기서도 명시적으로 설정
        return {"route": "generate"}

    grade_start = time.perf_counter()
    _grade_op = _enter_stage(OperationType.LLM_INFERENCE)
    _grade_op.__enter__()
    _grade_op_exited = False

    retry_count = get_state_attr(state, "retry_count", 0)
    max_retries = GRADING_CONFIG.get("max_retries", 2)
    if retry_count >= max_retries:
        logger.info(
            f"[RAG] [GRADE] 최대 재시도 횟수({retry_count}/{max_retries}) 도달. 즉시 생성 단계로 이동."
        )
        _grade_op.__exit__(None, None, None)
        _grade_op_exited = True
        return {"intent": "generate", "route": "generate"}

    docs = get_state_attr(state, "relevant_docs")
    if not docs:
        logger.info("[RAG] [GRADE] 문서가 없어 즉시 재구성 단계로 이동")
        # R1a-01: 증가는 grade 단일 지점에서만. "문서 없음" 경로가 루프 종료를
        # rewrite 폴백의 간접 증가에 의존하던 취약 결합을 명시적 델타 +1로 해소.
        _grade_op.__exit__(None, None, None)
        _grade_op_exited = True
        return {
            "intent": "transform",
            "route": "transform",
            "retry_count": 1,
        }

    # [F4] 초단문 쿼리 strict 가드: 5자 미만 쿼리는 그레이더가 임의로 확장 해석
    # (예: "cm3" -> CM3 모델)할 위험이 커 환각 소지가 있다. 검색된 문서 중 어디에도
    # 원문 토큰이 존재하지 않으면 오타/타 문서로 판단해 재검색(transform)으로
    # 라우팅한다. 정상 매칭(예: CM3 논문에 "CM3" 존재) 시에는 통과시켜 기존 흐름 유지.
    if get_state_attr(state, "short_query", False):
        import re

        _q = str(get_state_attr(state, "input", "")).strip().lower()
        # 관련성은 LLM grade가 보던 상위 grade_top_n 문서 기준으로만 판정한다.
        # (전체 docs는 상위 3 밖의 노이즈 문서가 포함될 수 있어 긍정 오판 여지)
        # 단어 경계 매칭으로 서브스트링 오탐("cm3"가 "ACM3" 등에 걸림)을 방지한다.
        _top_n = int(GRADING_CONFIG.get("grade_top_n", 3))
        _pat = re.compile(rf"\b{re.escape(_q)}\b")
        _hit = _q and any(
            _pat.search((d.page_content or "").lower()) for d in docs[:_top_n]
        )
        if not _hit:
            logger.info(
                f"[RAG] [GRADE] 초단문 쿼리 '{_q}' 가 검색 문서에 무존재 — 재검색 라우팅"
            )
            _grade_op.__exit__(None, None, None)
            _grade_op_exited = True
            return {"intent": "transform", "route": "transform", "retry_count": 1}
        # fast-path: 원문 토큰이 상위 문서에 단어 경계로 존재하면 관련성이 확실하므로
        # LLM grade 호출을 생략하고 바로 generate 로 직행한다 (5자 미만 키워드 환각 방지
        # 가드는 _hit 검사를 통과한 시점에 이미 충족 — 토큰 존재가 답변 근거를 보장).
        grade_ms = (time.perf_counter() - grade_start) * 1000
        logger.info(
            f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_query_fast_path=True"
        )
        _add_stage_ms("grade_ms", grade_ms)
        _grade_op.__exit__(None, None, None)
        _grade_op_exited = True
        return {"intent": "generate", "route": "generate"}

    # Wave 3 grade reduction: 동일 세션 내 동일 (query, doc-set) 반복 질의 시
    # 이전 LLM 판단을 재사용한다 (메모 해시 miss 시에만 LLM 호출).
    # 메모 저장 실패는 채점을 깨뜨리지 않도록 LLM 경로로 폴백한다.
    sid = _get_session_id(config)
    try:
        memo_key = _grade_memo_key(state, docs)
        grade_memo = SessionManager.get(_GRADE_MEMO_KEY, session_id=sid, default={})
        if isinstance(grade_memo, dict) and memo_key in grade_memo:
            decision = grade_memo[memo_key]
            reused_route = (
                decision.get("route", "generate")
                if isinstance(decision, dict)
                else "generate"
            )
            logger.info("[RAG] [GRADE] 메모이즈된 판단 재사용")
            SessionManager.add_status_log(
                "이전 평가 결과를 재사용합니다.", session_id=sid
            )
            grade_ms = (time.perf_counter() - grade_start) * 1000
            logger.info(f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} memo_hit=True")
            _add_stage_ms("grade_ms", grade_ms)
            _grade_op.__exit__(None, None, None)
            _grade_op_exited = True
            # [FIX] 메모 히트 시에도 재시도 카운터를 누적해야 한다.
            # 메모 재사용 경로가 retry_count 를 반환하지 않으면 reset_or_add
            # 리듀서가 카운터를 증가시키지 않아 hardcap(>= max_retries)에
            # 도달하지 못하고 무한 재귀(GraphRecursionError)가 발생한다.
            return {"intent": "generate", "route": reused_route, "retry_count": 1}
    except Exception as e:  # noqa: BLE001 - 메모 실패는 치명적이지 않음
        logger.debug(f"[RAG] [GRADE] 메모 조회 실패, LLM 경로로 진행: {e}")

    # Short-circuit: 리랭킹 점수가 충분히 높으면 LLM 검증 생략
    max_rerank_score = max(
        (d.metadata.get("rerank_score", 0.0) for d in docs), default=0.0
    )
    # [R3b-02] rerank_score 스케일은 엔진에 따라 다르다 — FlashRank 시그모이드(실측 0.06~0.91)와
    # bi-encoder 코사인(실측 0.32~0.57). 활성 엔진에 따라 임계값을 분기한다.
    from core.async_reranker import get_active_rerank_engine

    if get_active_rerank_engine() == "semantic":
        min_score_to_skip = GRADING_CONFIG.get("min_score_to_skip_semantic", 0.45)
    else:
        min_score_to_skip = GRADING_CONFIG.get("min_score_to_skip", 0.85)
    # [FIX] 상위-차상위 격차가 충분하지 않으면 점수가 애매한 문서 집합으로 판단해
    # short-circuit을 막고 LLM 검증을 수행한다 (모델 교체 시 wrong-answer 잠재 위험 차단).
    min_top_gap = GRADING_CONFIG.get("min_top_gap_to_skip", 0.05)
    sorted_scores = sorted(
        (float(d.metadata.get("rerank_score", 0.0)) for d in docs), reverse=True
    )
    top_gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else 1.0
    if max_rerank_score >= min_score_to_skip and top_gap >= min_top_gap:
        logger.info(
            f"[RAG] [GRADE] Short-circuit 활성화 (Max Rerank Score: {max_rerank_score:.3f} >= {min_score_to_skip})"
        )
        SessionManager.add_status_log(
            "High-confidence knowledge found. Generating the answer now.",
            session_id=_get_session_id(config),
        )
        grade_ms = (time.perf_counter() - grade_start) * 1000
        logger.info(f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=True")
        _add_stage_ms("grade_ms", grade_ms)
        _grade_op.__exit__(None, None, None)
        _grade_op_exited = True
        return {"intent": "generate", "route": "generate"}

    query = get_state_attr(state, "input")
    cfg = config.get("configurable", {})
    llm = cfg.get("llm")

    import json
    import re

    if writer is not None:
        await adispatch_custom_event(
            "graph_status", {"status": "문서 관련성 검증 중..."}, config=config
        )

    test_docs = docs[: int(GRADING_CONFIG.get("grade_top_n", 3))]
    context_text = "\n\n".join(
        [f"DOC {i + 1}: {d.page_content}" for i, d in enumerate(test_docs)]
    )

    grade_system = GRADE_PROMPT_CONFIG.get(
        "system_message",
        "당신은 문서 관련성 평가자이자 검색 쿼리 최적화 전문가입니다.",
    )
    grade_template = GRADE_PROMPT_CONFIG.get("human_message_template", "")
    unified_prompt = (
        f"{grade_system}\n\n" + grade_template.format(query=query, context=context_text)
        if grade_template
        else (
            f"{grade_system}\n\n"
            f"[질문]\n{query}\n\n"
            f"[검색된 문서 (상위 3개)]\n{context_text}\n\n"
            "[작업]\n"
            "1. 위 문서들이 질문에 답하기에 충분한지 평가하세요 (is_relevant: true/false)\n"
            "2. 충분하지 않다면, 더 나은 검색 결과를 위한 최적화된 쿼리를 작성하세요 "
            "(optimized_query)\n"
            "3. 판단 근거(reason)와 관련 엔티티(relevant_entities)도 포함하세요.\n\n"
            '출력은 반드시 JSON 형식이어야 합니다. (예: {"action": "generate", '
            '"is_relevant": true, "relevant_entities": ["A"], "reason": "...", '
            '"optimized_query": null})'
        )
    )

    call_config = (
        {"configurable": {**cfg, "messages": []}}
        if cfg
        else {"configurable": {"messages": []}}
    )

    # PHASE 2: Eagerly start generate (buffered) so its LLM round-trip overlaps
    # with the grade LLM call below. Adopted by generate on the common
    # route=generate path; cancelled if grade routes to transform.
    try:
        _start_speculative_generate(state, config, writer)
        if llm is None:
            raise ValueError("LLM is not initialized")

        # JSON 모드 강제 (구조화 출력 대신) — 단일 호출로 완성
        try:
            async with ModelManager.inference_session():
                json_llm = llm.bind(response_format={"type": "json_object"})
                result = await _safe_invoke(
                    json_llm,
                    unified_prompt,
                    call_config,
                    model_name=DEFAULT_OLLAMA_MODEL,
                )
            content = result.content if hasattr(result, "content") else str(result)
            data = json.loads(content)
            parsed = UnifiedGradeRewriteResponse(**data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"[RAG] [GRADE] JSON 모드 실패, 수동 파싱 시도: {e}")
            async with ModelManager.inference_session():
                raw_res = await _safe_invoke(
                    llm, unified_prompt, call_config, model_name=DEFAULT_OLLAMA_MODEL
                )
            raw_content = (
                raw_res.content if hasattr(raw_res, "content") else str(raw_res)
            )
            match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            if match:
                data = json.loads(match.group())
                parsed = UnifiedGradeRewriteResponse(**data)
            else:
                raise ValueError("JSON 패턴을 찾을 수 없습니다.") from None

        if parsed.action == "generate" or parsed.is_relevant:
            logger.info(f"[RAG] [GRADE] 관련성 확인: YES ({parsed.reason})")
            SessionManager.add_status_log(
                "검색된 지식의 관련성이 확인되었습니다.",
                session_id=_get_session_id(config),
            )
            _store_grade_memo(state, docs, {"route": "generate"}, sid)
            grade_ms = (time.perf_counter() - grade_start) * 1000
            logger.info(
                f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=False"
            )
            _add_stage_ms("grade_ms", grade_ms)
            _grade_op.__exit__(None, None, None)
            _grade_op_exited = True
            # PHASE 2: route=generate → keep the warm speculative generate; the
            # real generate node adopts it (single LLM call, started earlier).
            return {"intent": "generate", "route": "generate"}
        else:
            optimized = parsed.optimized_query or query
            logger.info(
                f"[RAG] [GRADE] 관련성 확인: NO → 재작성: {optimized} ({parsed.reason})"
            )
            SessionManager.add_status_log(
                "검색 결과가 부적합하여 질문 재구성을 시도합니다.",
                session_id=_get_session_id(config),
            )
            _store_grade_memo(state, docs, {"route": "transform"}, sid)
            grade_ms = (time.perf_counter() - grade_start) * 1000
            logger.info(
                f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=False"
            )
            _add_stage_ms("grade_ms", grade_ms)
            _grade_op.__exit__(None, None, None)
            _grade_op_exited = True
            # PHASE 2: route=transform → discard the speculative generate; its
            # buffered output must never reach the user.
            _cancel_speculative_generate(config)
            return {
                "intent": "transform",
                "route": "transform",
                "search_queries": [optimized],
                # 리듀서 reset_or_add는 합산 계약이므로 항상 상수 델타 1을 반환한다.
                # `retry_count + 1`을 반환하면 누적값이 다시 합산돼 예산이 이중 소진된다 (R1a-01).
                "retry_count": 1,
            }

    except (RuntimeError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"[RAG] [GRADE] 평가 실패, 기본값(NO) 적용하여 재구성 시도: {e}")
        _store_grade_memo(state, docs, {"route": "transform"}, sid)
        grade_ms = (time.perf_counter() - grade_start) * 1000
        logger.info(
            f"[RAG] [GRADE][TIMING] grade_ms={grade_ms:.1f} short_circuit=False"
        )
        _add_stage_ms("grade_ms", grade_ms)
        _grade_op.__exit__(None, None, None)
        _grade_op_exited = True
        # PHASE 2: LLM/JSON 오류로 route=transform이 되면 speculative generate를
        # 취소·폐기해야 한다 (미노출 + 레지스트리 정리 → 이후 동일 thread_id 겹침 재활성).
        _cancel_speculative_generate(config)
        return {
            "intent": "transform",
            "route": "transform",
            "retry_count": 1,
        }

    except Exception:
        logger.exception("[RAG] [GRADE] unrecoverable error — canceling speculative")
        _cancel_speculative_generate(config)
        if not _grade_op_exited:
            _grade_op.__exit__(*sys.exc_info())
            _grade_op_exited = True
        raise  # re-panic: preserve original error for graph error handling


async def rewrite_query(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """grade_documents에서 이미 재작성된 쿼리를 전달합니다. (LLM 호출 없음)"""
    search_queries = get_state_attr(state, "search_queries")

    if search_queries:
        new_query = search_queries[-1]
        logger.info(f"[RAG] [REWRITE] 전달받은 재구성 쿼리 사용: '{new_query}'")
        # retry_count는 grade_documents가 이미 +1을 적용했으므로 순수 passthrough만 한다.
        # (리듀서 reset_or_add가 합산하므로 여기서 값을 내려보내면 재시도 예산이 이중 소진됨)
        return {}

    # 폴백: grade_documents에서 쿼리 생성 실패 시 원본 유지.
    # retry_count 증가는 grade_documents 단일 지점에서만 수행한다 (R1a-01).
    # 합산 리듀서(reset_or_add)에 절대 목표값을 델타로 반환하면 예산이 이중 소진된다.
    query = get_state_attr(state, "input")
    logger.info(f"[RAG] [REWRITE] 재검색 쿼리 없음, 원본 유지: '{query}'")
    return {}


# ============================================================================
# _verify 섹션 — 답변 충실도 검증 노드 (pure move)
# ============================================================================


# verify 노드와 테스트에서 공유하는 안정-id 인용 검증 정규식.
# 형식은 format_context가 내보내는 토큰과 동일해야 한다 ([doc:<stable_id>]).
_RE_VERIFY_DOC_CITATION = re.compile(r"\[doc:([\w-]+)\]")


def _validate_cited_doc_ids(answer: str, docs: list[Document]) -> list[str]:
    """답변 내 [doc:<stable_id>] 인용이 docs에 실제로 존재하는지 멤버십 검사.

    위치 기반 int 범위 검사(int(...) > max_idx)가 아니라, docs에서 계산한
    안정-id 집합(valid_ids)에 cited id가 속하는지 확인한다. 존재하지 않는
    (알 수 없는/환각) id만 invalid_citations로 반환한다.
    """
    if not docs:
        # docs가 없는데 인용이 남아있으면 모두 무효.
        return [m.group(1) for m in _RE_VERIFY_DOC_CITATION.finditer(answer)]

    valid_ids = {_doc_stable_id(d) for d in docs}
    invalid: list[str] = []
    for m in _RE_VERIFY_DOC_CITATION.finditer(answer):
        cited_id = m.group(1)
        if cited_id not in valid_ids:
            invalid.append(cited_id)
    return invalid


def _should_verify(query: str, doc_ids: list[str]) -> bool:
    """결정적 해시 기반 검증 샘플링 결정 (D19).

    query와 정렬된 doc stable-id 집합을 sha256으로 해시해 [0, 1) 구간 bucket으로
    변환한 뒤 VERIFICATION_SAMPLE_RATE와 비교한다. 동일 입력 → 동일 결정이므로
    검증 실패 케이스를 재현할 수 있고, rate=1.0이면 항상 검증, rate=0.0이면
    절대 검증하지 않는다. 문서가 없으면 doc_ids=[""]로 처리해 빈-문서 쿼리도
    결정적 버킷을 갖는다.
    """
    stable_doc_ids = doc_ids or [""]
    stable_input = f"{query}::{','.join(sorted(stable_doc_ids))}"
    digest = hashlib.sha256(stable_input.encode()).hexdigest()
    bucket = int(digest, 16) % 10000 / 10000.0
    return bucket < VERIFICATION_SAMPLE_RATE


# Phase 1.5: Post-Generation Verification Node
async def verify_answer(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """생성된 답변의 충실도(Faithfulness)와 인용 일관성을 검증합니다."""
    # 검증 대상 데이터 (샘플링 결정에 필요 — D19 결정적 해시 입력).
    # random 샘플링과 달리 동일 query+docs는 항상 동일 결정을 내린다.
    answer = get_state_attr(state, "response", "")
    docs = get_state_attr(state, "relevant_docs") or []
    query = get_state_attr(state, "input", "")
    doc_ids = [_doc_stable_id(d) for d in docs]

    # 샘플링: 프로덕션에서는 일부만 검증
    if not VERIFICATION_ENABLED or not _should_verify(query, doc_ids):
        return {"verification_route": "end"}

    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        return {"verification_route": "end"}

    # 컨텍스트 구성
    context = format_context(docs) if docs else ""

    # 인용 일관성 검사: 모든 [doc:<stable_id>]가 실제 문서 집합에 존재하는지
    # (위치 기반 int 범위 검사가 아니라 stable id 멤버십 검사).
    invalid_citations = _validate_cited_doc_ids(answer, docs)

    if invalid_citations:
        logger.warning(f"[RAG] [VERIFY] 유효하지 않은 인용 감지: {invalid_citations}")
        return {
            "verification_route": "regenerate",
            "verification_issues": [f"유효하지 않은 인용: {invalid_citations}"],
        }
    verify_sys_text = VERIFY_PROMPT_CONFIG.get(
        "system_message",
        "답변의 충실도를 엄격하게 평가하십시오. 컨텍스트에 없는 내용이 있으면 faithful: false로 판단하십시오.",
    )
    verify_template = VERIFY_PROMPT_CONFIG.get("human_message_template", "")
    if verify_template:
        verify_prompt = verify_template.format(
            context=context, answer=answer, query=query
        )
    else:
        verify_prompt = (
            f"당신은 답변 검증 전문가입니다. 아래 [Context]와 [Answer]를 보고 "
            f"답변이 컨텍스트에 충실한지 판단하십시오.\n\n"
            f"[Context]\n{context}\n\n[Answer]\n{answer}\n\n[Question]\n{query}\n\n"
            f"다음 JSON 형식으로만 답변하십시오:\n"
            f'{{"faithful": true/false, "issues": ["문제점1"]}}\n\n'
            f"판단 기준:\n"
            f"1. 답변의 모든 핵심 주장이 컨텍스트에 근거하는가?\n"
            f"2. 컨텍스트에 없는 정보를 추측하여 답변했는가?\n"
            f"3. 인용된 내용이 실제 컨텍스트와 일치하는가?"
        )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        sys_msg = SystemMessage(content=verify_sys_text)
        human_msg = HumanMessage(content=verify_prompt)

        async with ModelManager.inference_session():
            coordinator = get_resource_manager()
            async with coordinator.use_llm(model_name=DEFAULT_OLLAMA_MODEL):
                response = await llm.ainvoke([sys_msg, human_msg], config=config)

        import json

        content = response.content if hasattr(response, "content") else str(response)
        # JSON 추출
        json_str = _strip_json_fence(content)

        result = json.loads(json_str)
        faithful = result.get("faithful", True)
        issues = result.get("issues", [])

        if not faithful or invalid_citations:
            all_issues = issues + (
                [f"유효하지 않은 인용: {invalid_citations}"]
                if invalid_citations
                else []
            )
            logger.warning(f"[RAG] [VERIFY] 검증 실패: {all_issues}")
            # 재시도 횟수 체크 (최대 1회)
            regen_count = get_state_attr(state, "regeneration_count", 0)
            if regen_count >= 1:
                logger.warning(
                    "[RAG] [VERIFY] 최대 재생성 횟수(1회) 초과, 검증 실패 상태로 종료"
                )
                return {"verification_route": "end", "verification_issues": all_issues}
            return {
                "verification_route": "regenerate",
                "verification_issues": all_issues,
                "regeneration_count": regen_count + 1,
            }

        logger.info("[RAG] [VERIFY] 검증 통과")
        return {"verification_route": "end"}

    except Exception as e:
        logger.error(f"[RAG] [VERIFY] 검증 중 오류(재생성 시도): {e}", exc_info=True)
        # fail-closed: 검증 실패=신뢰 불가 → 통과("end")로 오인하지 않고 재생성.
        # 무한루프 방지: 재생성 1회(최대) 소진 시에만 end 폴백.
        regen_count = get_state_attr(state, "regeneration_count", 0)
        if regen_count >= 1:
            logger.warning("[RAG] [VERIFY] 재생성 소진 후 검증 실패 → end 폴백")
            return {"verification_route": "end"}
        return {
            "verification_route": "regenerate",
            "verification_issues": [f"검증 실행 오류: {e}"],
        }
