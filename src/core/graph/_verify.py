"""답변 충실도 검증(Post-Generation Verification) LangGraph 노드 모듈.

``graph_builder.py`` 에서 추출됨 (graph-builder split plan Step 6, pure move);
``graph_builder.py`` 는 하위 호환을 위해 이 이름을 re-export 한다.
"""

import logging
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter

from api.schemas import GraphState
from common.config import DEFAULT_OLLAMA_MODEL, VERIFY_PROMPT_CONFIG
from core.graph._generate import format_context
from core.graph._graph_utils import _doc_stable_id, get_state_attr
from core.graph._json_utils import _strip_json_fence
from core.model_loader import ModelManager
from core.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)


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


# Phase 1.5: Post-Generation Verification Node
async def verify_answer(
    state: GraphState, config: RunnableConfig, *, writer: StreamWriter
) -> dict[str, Any]:
    """생성된 답변의 충실도(Faithfulness)와 인용 일관성을 검증합니다."""
    import random

    from common.config import VERIFICATION_ENABLED, VERIFICATION_SAMPLE_RATE

    # 샘플링: 프로덕션에서는 일부만 검증
    if not VERIFICATION_ENABLED or random.random() > VERIFICATION_SAMPLE_RATE:
        return {"verification_route": "end"}

    cfg = config.get("configurable", {})
    llm = cfg.get("llm")
    if not llm:
        return {"verification_route": "end"}

    # 검증 대상 데이터
    answer = get_state_attr(state, "response", "")
    docs = get_state_attr(state, "relevant_docs") or []
    query = get_state_attr(state, "input", "")

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
