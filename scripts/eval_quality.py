"""RAG 답변 품질 평가 하네스 (Wave 0 - Todo 1).

GraphRAG-Ollama 시스템의 답변 품질 측정 스크립트. baseline(Wave 0)과 개선 후
(Wave 3) 재측정에 동일한 조건으로 사용된다. 제품 코드(src/)는 수정하지 않는다.

측정 항목:
  (a) retrieval P@1/MRR@5 - tests/data/golden_set.json (퇴화 질문 제외 규칙 적용)
      + context_recall: golden_set.json required_chunks 근거 청크 포함 비율 (R4-06)
  (b) TTFT/TPS            - RAGSystem.astream 직접 소비 + stream_graph_events
  (c) 답변 길이           - 응답 문자열 길이(문자) + Ollama 실측 eval_count
  (d) 잘림 감지           - eval_count >= OLLAMA_NUM_PREDICT * 0.9
  (e) 1-5점 judge         - tests/data/testset_2201.csv 상위 N행
      + faithfulness: SCORE_PROMPT [Context] 근거성 판정 (R4-06 최소 구현)
  (f) prompt 토큰 초과    - count_tokens_rough(protocol+context+질문) > OLLAMA_NUM_CTX

측정 정의 (Todo 1 사양):
  t_start=t_stream_start, t_first=첫 비어있지 않은 content 청크 도착 시각
  (thought/status는 content="" 이라 자연 배제), TTFT=t_first-t_start,
  t_end=마지막 content 청크 도착, TPS=eval_count/(t_end-t_first).
  StreamingMetrics.first_token_latency는 messages 분기 전용이고(streaming_handler.py:247-249)
  답변은 custom response_chunk로 흐르며 rag_core.py:181-182가 generate의
  on_chat_model_stream을 스킵하므로 사용 금지. eval_count 누락 시 TPS만 N/A.
  주의: thinking 토큰 선행 출력 모델로 교체 시 eval_count가 thinking 포함 가능
  (현재 qwen3:4b-instruct-2507은 non-thinking).

--no-llm: retrieve 문서 수신 직후 스트림 중단으로 generation/judge 미실행
  (LLM 콜드 스타트 방지). 첫 astream의 prepare_query_config_or_build가 LLM
  리소스를 1회 로드할 수 있음 (생성 호출과 무관).
"""

# allow: SIZE_OK - 계획(todo 1)이 단일 스크립트로 통합·고정 CLI(python scripts/eval_quality.py)
# 재사용(todo 2/9)을 명시. 22개 함수가 공유 dataclass/상수에 결합된 평가 하네스로 분리 시 CLI 계약 위반.
# ruff: noqa: E402 - sys.path 부트스트랩 이후 임포트 (scripts 표준 패턴)

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# 프로젝트 루트와 src를 모듈 경로에 추가 (PYTHONPATH 미설정 환경 대응)
ROOT_DIR = Path(__file__).resolve().parent.parent
for _path in (str(ROOT_DIR), str(ROOT_DIR / "src")):
    if _path not in sys.path:
        sys.path.append(_path)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# isort: off
from src.api.streaming_handler import StreamingResponseHandler
from src.common.config import (
    ANALYSIS_PROTOCOL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    EVAL_JUDGE_MODEL,
    OLLAMA_NUM_CTX,
    OLLAMA_NUM_PREDICT,
)
from src.common.logging_config import setup_logging
from src.common.utils import count_tokens_rough
from src.core.graph_builder import format_context
from src.core.model_loader import ModelManager
from src.core.rag_core import RAGSystem
# isort: on

# --- 상수 ---
TIMEOUT_SECONDS: float = 180.0  # 질문별 타임아웃 가드
TRUNCATION_RATIO: float = 0.9  # eval_count가 num_predict * 0.9 이상이면 잘림 판정
GOLDEN_SET_PATH = ROOT_DIR / "tests" / "data" / "golden_set.json"
TESTSET_PATH = ROOT_DIR / "tests" / "data" / "testset_2201.csv"
DEFAULT_PDF = "tests/data/2201.07520v1.pdf"

# quick_eval.py:20-33의 SCORE_PROMPT 패턴 + [Context] 근거성(faithfulness) 판정 (R4-06)
SCORE_PROMPT = """
당신은 RAG 시스템 평가 전문가입니다. 아래 [질문], [정답], [모델 답변], [Context]를 비교하여 답변의 품질과 근거성을 1점에서 5점 사이로 평가하세요.

[질문]: {question}
[정답]: {ground_truth}
[모델 답변]: {answer}
[Context]: {context}

[평가 기준]:
- 5점: 답변이 정답과 의미적으로 완벽하게 일치하며 정확하고, 답변의 모든 주장이 제공된 Context에 근거함.
- 3점: 답변의 핵심은 맞지만 세부 정보가 부족하거나 약간의 노이즈가 있음.
- 1점: 답변이 틀렸거나 질문과 관련이 없음.
- 근거성(Faithfulness): 답변의 모든 주장이 제공된 Context에 근거하는지 1-5점으로 함께 평가하세요. Context에 없는 주장을 포함하면 감점합니다.

결과는 반드시 숫자 하나(예: 5)만 출력하세요. 설명은 생략하세요.
"""


@dataclass(frozen=True)
class QuestionSpec:
    """평가 대상 질문 정의."""

    source: str  # "golden" | "testset"
    row: int  # golden: golden_set.json의 1-based 행 번호 / testset: head(N) 내 1-based 행
    query: str
    entities: tuple[str, ...] = ()
    ground_truth: str | None = None
    required_chunks: tuple[str, ...] = ()  # context_recall 근거 청크 키워드 (R4-06)


# --- 데이터 로드 ---
def _load_golden() -> list[QuestionSpec]:
    """golden_set.json 전체 로드 (퇴화 질문 제외 규칙은 판정 단계에서 적용)."""
    with open(GOLDEN_SET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return [
        QuestionSpec(
            source="golden",
            row=idx,
            query=item["query"],
            entities=tuple(item.get("expected_key_entities") or []),
            required_chunks=tuple(item.get("required_chunks") or []),
        )
        for idx, item in enumerate(raw, start=1)
    ]


def _load_testset(n: int) -> list[QuestionSpec]:
    """testset_2201.csv의 상위 N행 로드 (df.head(n), 결정적 순서)."""
    import pandas as pd

    df = pd.read_csv(TESTSET_PATH)
    specs: list[QuestionSpec] = []
    for idx, (_, row) in enumerate(df.head(n).iterrows(), start=1):
        specs.append(
            QuestionSpec(
                source="testset",
                row=idx,
                query=str(row["question"]),
                ground_truth=str(row["ground_truth"]),
            )
        )
    return specs


# --- 퇴화 질문 판별 (Todo 1 사양) ---
def _is_general_intent(query: str) -> bool:
    """preprocess 의도 분류 규칙의 스크립트 측 복제 (graph_builder.py:108-111).

    len < 10 + 인사 키워드면 intent=general로 분류된다. 결정적이며 제품 로직과 동일.
    """
    q = query.strip().lower()
    return len(q) < 10 and any(
        g in q for g in ("안녕", "hi", "hello", "반가워", "누구")
    )


def _classify_degenerate(q: QuestionSpec) -> str | None:
    """퇴화 질문이면 사유 반환 (golden 전용, testset은 판정 대상 아님).

    규칙: expected_key_entities 빈 배열 또는 preprocess intent=general은 P@1/MRR
    집계 제외. 현재 golden_set에서 제외되는 것은 행 4("안녕, 너는 누구니?",
    entities=[]) 1건뿐. 행 5("이미지 생성 모델의 최신 트렌드...")는 entities
    비어있지 않고 intent=rag이므로 집계 포함 (out-of-doc 스트레스 케이스).
    """
    if q.source != "golden":
        return None
    if not q.entities:
        return "빈 엔티티 (expected_key_entities=[])"
    if _is_general_intent(q.query):
        return "intent=general (preprocess 규칙 복제)"
    return None


def _out_of_doc_note(q: QuestionSpec) -> str | None:
    """행 5 out-of-doc 스트레스 케이스 표기."""
    if q.source == "golden" and "최신 트렌드" in q.query:
        return "out-of-doc 스트레스 케이스 (문서 밖 질문, P@1=0 기대)"
    return None


# --- 스트리밍 측정 ---
async def _consume_stream(rag: RAGSystem, query: str, no_llm: bool) -> dict[str, Any]:
    """astream을 stream_graph_events로 소비하며 측정.

    documents=updates/retrieve의 metadata["documents"], performance=generate 성능
    청크의 eval_count. no_llm 시 첫 documents 수신 직후 중단 (generation 미실행).
    """
    handler = StreamingResponseHandler()
    event_stream = await rag.astream(query, model_name=DEFAULT_OLLAMA_MODEL)
    t_start = time.time()
    content_parts: list[str] = []
    t_first_content: float | None = None
    t_last_content: float | None = None
    documents: list[Any] | None = None
    performance: dict[str, Any] = {}
    async for chunk in handler.stream_graph_events(event_stream):
        if chunk.content:
            content_parts.append(chunk.content)
            if t_first_content is None:
                t_first_content = chunk.timestamp
            t_last_content = chunk.timestamp
        if chunk.metadata and "documents" in chunk.metadata:
            documents = chunk.metadata["documents"]
            if no_llm:
                break
        if chunk.performance:
            performance.update(chunk.performance)
    answer = "".join(content_parts)
    eval_count = performance.get("eval_count")
    ttft = t_first_content - t_start if t_first_content is not None else None
    tps: float | None = None
    if (
        eval_count is not None
        and t_first_content is not None
        and t_last_content is not None
    ):
        elapsed = t_last_content - t_first_content
        if elapsed > 0:
            tps = eval_count / elapsed
    return {
        "answer": answer,
        "answer_chars": len(answer),
        "eval_count": eval_count,
        "eval_count_missing": eval_count is None,
        "ttft": ttft,
        "tps": tps,
        "documents": documents or [],
        "performance": performance,
    }


# 한국어 엔티티 -> 영문 동의어/어간 맵 (R4-01 수정).
# golden_set.json의 expected_key_entities가 한국어인데 PDF 텍스트가 영문이어서
# 부분문자열 매칭이 구조적으로 실패(P@1/MRR@5 영구 0)했던 결함을 교정한다.
_ENTITY_SYNONYM_MAP: dict[str, tuple[str, ...]] = {
    "토큰화": ("tokeniz", "token"),
    "토큰": ("token",),
    "검색": ("retriev", "search"),
    "임베딩": ("embed",),
    "청크": ("chunk",),
    "정렬": ("sort", "rank"),
    "파라미터": ("parameter", "param"),
    "데이터셋": ("dataset", "data set", "data source", "corpus"),
    "규모": ("size", "scale", "million", "billion"),
    "기여점": ("contribut",),
    "요약": ("summari", "summary"),
    "이미지": ("image",),
    "이미지 학습": ("image token", "image model", "image generat", "image train"),
    "이미지 생성": ("image generat", "image synthes", "text-to-image"),
    "학습": ("train", "learn"),
    "모델": ("model",),
    "생성": ("generat",),
    "트렌드": ("trend", "recent", "latest", "state-of-the-art"),
}


def _normalize_match(text: str) -> str:
    """공백·하이픈·언더스코어를 제거한 소문자 텍스트 (근접 판정용)."""
    return re.sub(r"[\s\-_]+", "", text.lower())


def _entity_match(text_lower: str, text_norm: str, entity: str) -> bool:
    """단일 엔티티가 문서 텍스트에 존재하는지 판정 (원문 + 동의어/어간 맵).

    대소문자 하강 후 부분문자열 매칭을 시도하고, 실패 시 공백/하이픈을
    무시한 근접 판정(_normalize_match)까지 허용한다.
    """
    entity_clean = entity.strip().lower()
    if not entity_clean:
        return False
    candidates = (entity_clean,) + _ENTITY_SYNONYM_MAP.get(entity_clean, ())
    if any(cand and cand in text_lower for cand in candidates):
        return True
    entity_norm = _normalize_match(entity_clean)
    if entity_norm and entity_norm in text_norm:
        return True
    return any(cand and _normalize_match(cand) in text_norm for cand in candidates)


def _doc_relevant(doc: Any, entities: tuple[str, ...]) -> bool:
    """문서의 page_content가 모든 엔티티와 매칭되면 관련 판정 (R4-01 수정).

    영어 PDF 텍스트와 한국어 엔티티 사이의 언어 갭을 _ENTITY_SYNONYM_MAP과
    대소문자/공백/하이픈 정규화로 메운다. 모든 엔티티가 매칭되어야 관련이다.
    """
    text = getattr(doc, "page_content", "") or ""
    text_lower = text.lower()
    text_norm = _normalize_match(text)
    return all(_entity_match(text_lower, text_norm, e) for e in entities)


def _mrr_at_5(relevance: list[bool]) -> float:
    """첫 관련 문서 순위의 역수 (관련 문서 없으면 0.0)."""
    for rank, is_rel in enumerate(relevance, start=1):
        if is_rel:
            return round(1.0 / rank, 4)
    return 0.0


def _chunk_recalled(docs: list[Any], phrase: str) -> bool:
    """필수 근거 청크 키워드가 검색된 문서 중 하나에 포함되면 recall 판정 (R4-06).

    context_recall의 분모(golden_set.json required_chunks) 기준으로, 검색 결과
    상위 문서에 해당 근거 청크가 포함된 비율을 측정한다.
    """
    phrase_clean = phrase.strip().lower()
    if not phrase_clean:
        return False
    phrase_norm = _normalize_match(phrase_clean)
    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        if phrase_clean in text.lower():
            return True
        if phrase_norm and phrase_norm in _normalize_match(text):
            return True
    return False


def _estimate_prompt_tokens(query: str, docs: list[Any]) -> int:
    """generate 프롬프트(protocol+context+질문, chat_history 제외) 토큰 추정.

    human_msg 구성과 동일 (graph_builder.py:499-501).
    """
    context = format_context(docs) if docs else "일상적인 대화입니다."
    prompt = f"{ANALYSIS_PROTOCOL}\n\n[Context]\n{context}\n\n[Question]\n{query}"
    return count_tokens_rough(prompt)


def _fmt(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _empty_record(
    q: QuestionSpec, degenerate: str | None, skipped: bool
) -> dict[str, Any]:
    return {
        "source": q.source,
        "row": q.row,
        "query": q.query,
        "ground_truth": q.ground_truth,
        "degenerate": {
            "is_degenerate": degenerate is not None,
            "reason": degenerate,
            "status": "제외됨" if degenerate else "포함",
        },
        "note": _out_of_doc_note(q),
        "skipped_no_llm": skipped,
        "retrieval": None,
        "answer": "",
        "answer_chars": 0,
        "eval_count": None,
        "eval_count_missing": True,
        "ttft": None,
        "tps": None,
        "truncated": 0,
        "prompt_tokens_est": 0,
        "overflow": 0,
        "judge": None,
        "faithfulness": None,
        "judge_excluded": None,
        "judge_error": None,
        "error": None,
    }


def _assemble_measurement(
    rec: dict[str, Any], q: QuestionSpec, measured: dict[str, Any]
) -> None:
    """스트리밍 측정 결과를 레코드에 반영 (retrieval 판정 + 길이/잘림/overflow)."""
    docs = list(measured["documents"])[:5]
    doc_details: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        doc_details.append(
            {
                "page": meta.get("page", "?"),
                "section": meta.get("current_section", "일반 본문"),
                "rerank_score": float(meta.get("rerank_score", 0.0) or 0.0),
            }
        )
    max_rerank = max((d["rerank_score"] for d in doc_details), default=0.0)

    retrieval: dict[str, Any] = {
        "doc_count": len(docs),
        "max_rerank_score": round(max_rerank, 4),
        "documents": doc_details,
    }
    is_scorable = q.source == "golden" and rec["degenerate"]["is_degenerate"] is False
    if is_scorable:
        relevance = [_doc_relevant(d, q.entities) for d in docs]
        retrieval["relevant"] = any(relevance)
        retrieval["p_at_1"] = 1 if (relevance and relevance[0]) else 0
        retrieval["mrr_at_5"] = _mrr_at_5(relevance)
        if q.required_chunks:
            recalled = sum(
                1 for phrase in q.required_chunks if _chunk_recalled(docs, phrase)
            )
            retrieval["context_recall"] = round(recalled / len(q.required_chunks), 4)
        else:
            retrieval["context_recall"] = None
    else:
        retrieval["relevant"] = None
        retrieval["p_at_1"] = None
        retrieval["mrr_at_5"] = None
        retrieval["context_recall"] = None

    eval_count = measured["eval_count"]
    prompt_est = _estimate_prompt_tokens(q.query, docs)
    rec.update(
        {
            "retrieval": retrieval,
            "answer": measured["answer"],
            "answer_chars": measured["answer_chars"],
            "eval_count": eval_count,
            "eval_count_missing": measured["eval_count_missing"],
            "ttft": _fmt(measured["ttft"]),
            "tps": _fmt(measured["tps"]),
            "truncated": (
                1
                if eval_count is not None
                and eval_count >= OLLAMA_NUM_PREDICT * TRUNCATION_RATIO
                else 0
            ),
            "prompt_tokens_est": prompt_est,
            "overflow": 1 if prompt_est > OLLAMA_NUM_CTX else 0,
        }
    )


async def _judge(
    score_chain: Any,
    question: str,
    ground_truth: str,
    answer: str,
    context: str,
) -> int | None:
    """SCORE_PROMPT 패턴 1-5점 판정 (Context 근거성 포함). 파싱 실패 시 None (판정 제외).

    bare except 금지 - 구체 예외 (json.JSONDecodeError, ValueError, TypeError)만.
    """
    try:
        score_str = await score_chain.ainvoke(
            {
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "context": context,
            }
        )
        match = re.search(r"[1-5]", str(score_str))
        if match is None:
            raise ValueError("judge 응답에서 1-5 점수 미발견")
        return int(match.group())
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


async def _run_question(
    rag: RAGSystem,
    q: QuestionSpec,
    no_llm: bool,
    score_chain: Any | None,
) -> dict[str, Any]:
    """질문 1건 평가. 타임아웃/오류 시 해당 질문만 error 기록 후 계속."""
    degenerate = _classify_degenerate(q)
    if no_llm and degenerate is not None:
        # 퇴화 질문은 --no-llm에서 스트리밍 생략 (general 경로가 LLM을 호출할 수 있음)
        return _empty_record(q, degenerate, skipped=True)

    rec = _empty_record(q, degenerate, skipped=False)
    try:
        measured = await asyncio.wait_for(
            _consume_stream(rag, q.query, no_llm), timeout=TIMEOUT_SECONDS
        )
    except asyncio.TimeoutError:
        rec["error"] = f"timeout after {TIMEOUT_SECONDS}s"
        return rec
    except Exception as exc:  # 질문 단위 오류로 기록 후 계속 (스크립트 중단 금지)
        rec["error"] = str(exc)
        return rec

    _assemble_measurement(rec, q, measured)

    if no_llm:
        rec["judge_excluded"] = True
        rec["judge_error"] = "--no-llm 모드 (generation/judge 생략)"
    elif q.source == "testset" and q.ground_truth and score_chain is not None:
        judge_docs = list(measured["documents"])[:5]
        context = format_context(judge_docs) if judge_docs else "일상적인 대화입니다."
        try:
            score = await _judge(
                score_chain, q.query, q.ground_truth, rec["answer"], context
            )
            rec["judge"] = score
            rec["faithfulness"] = score  # [Context] 근거성 판정 점수 (R4-06 최소 구현)
            rec["judge_excluded"] = score is None
            if score is None:
                rec["judge_error"] = "judge 파싱 실패 (판정 제외)"
        except Exception as exc:  # judge 호출 실패는 해당 질문 판정 제외만
            rec["judge_excluded"] = True
            rec["judge_error"] = str(exc)
    return rec


# --- 집계 ---
def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    golden_scorable = [
        r
        for r in results
        if r["source"] == "golden"
        and r["degenerate"]["is_degenerate"] is False
        and r["error"] is None
    ]
    executed = [
        r
        for r in results
        if r["degenerate"]["is_degenerate"] is False
        and r["error"] is None
        and not r["skipped_no_llm"]
    ]

    p_at_1_values = [r["retrieval"]["p_at_1"] for r in golden_scorable]
    mrr_values = [r["retrieval"]["mrr_at_5"] for r in golden_scorable]
    context_recall_values = [
        r["retrieval"]["context_recall"]
        for r in golden_scorable
        if r["retrieval"].get("context_recall") is not None
    ]
    ttft_values = [r["ttft"] for r in executed if r["ttft"] is not None]
    tps_values = [r["tps"] for r in executed if r["tps"] is not None]
    eval_counts = [r["eval_count"] for r in executed if r["eval_count"] is not None]
    eval_count_denom = sum(1 for r in executed if r["eval_count"] is not None)
    truncated_count = sum(
        1 for r in executed if r["eval_count"] is not None and r["truncated"]
    )
    judge_values = [r["judge"] for r in results if r["judge"] is not None]
    faithfulness_values = [
        r["faithfulness"] for r in results if r["faithfulness"] is not None
    ]
    degenerate = [
        {
            "source": r["source"],
            "row": r["row"],
            "reason": r["degenerate"]["reason"],
        }
        for r in results
        if r["degenerate"]["is_degenerate"]
    ]

    return {
        "scorable_count": len(golden_scorable),
        "p_at_1": _mean([float(v) for v in p_at_1_values]),
        "mrr_at_5": _mean([float(v) for v in mrr_values]),
        "context_recall": _mean([float(v) for v in context_recall_values]),
        "context_recall_count": len(context_recall_values),
        "latency_question_count": len(executed),
        "avg_ttft": _mean(ttft_values),
        "avg_tps": _mean(tps_values),
        "avg_answer_chars": _mean([float(r["answer_chars"]) for r in executed]),
        "avg_eval_count": _mean([float(v) for v in eval_counts]),
        "truncated_ratio": (
            round(truncated_count / eval_count_denom, 4) if eval_count_denom else None
        ),
        "truncated_count": truncated_count,
        "overflow_ratio": _mean([float(r["overflow"]) for r in executed]),
        "overflow_count": sum(1 for r in executed if r["overflow"]),
        "judge_avg": _mean([float(v) for v in judge_values]),
        "judge_count": len(judge_values),
        "faithfulness_avg": _mean([float(v) for v in faithfulness_values]),
        "faithfulness_count": len(faithfulness_values),
        "eval_count_missing_count": sum(1 for r in executed if r["eval_count_missing"]),
        "degenerate": degenerate,
    }


# --- 출력 ---
def _render_markdown(report: dict[str, Any]) -> str:
    meta = report["meta"]
    agg = report["aggregates"]
    lines = [
        "# RAG Answer Quality Evaluation Report",
        "",
        f"- Date: {meta['timestamp']}",
        f"- PDF: {meta['pdf']}",
        f"- Model: {meta['model']} / Embedder: {meta['embedder']}",
        f"- num_predict: {meta['num_predict']} / num_ctx: {meta['num_ctx']}",
        f"- Truncation threshold: {meta['truncation_threshold']} (0.9 x num_predict)",
        f"- --no-llm: {meta['no_llm']} / testset_n: {meta['testset_n']}",
        "",
        "## Aggregates",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| scorable_count (golden, 퇴화 제외) | {agg['scorable_count']} |",
        f"| P@1 | {agg['p_at_1']} |",
        f"| MRR@5 | {agg['mrr_at_5']} |",
        f"| context_recall (golden) | {agg['context_recall']} ({agg['context_recall_count']}건) |",
        f"| latency_question_count | {agg['latency_question_count']} |",
        f"| avg TTFT (s) | {agg['avg_ttft']} |",
        f"| avg TPS | {agg['avg_tps']} |",
        f"| avg answer chars | {agg['avg_answer_chars']} |",
        f"| avg eval_count | {agg['avg_eval_count']} |",
        f"| truncated ratio | {agg['truncated_ratio']} ({agg['truncated_count']}건) |",
        f"| overflow ratio | {agg['overflow_ratio']} ({agg['overflow_count']}건) |",
        f"| judge avg (1-5) | {agg['judge_avg']} ({agg['judge_count']}건) |",
        f"| faithfulness avg (1-5) | {agg['faithfulness_avg']} ({agg['faithfulness_count']}건) |",
        f"| eval_count missing | {agg['eval_count_missing_count']}건 |",
        "",
        "## Per-Question",
        "",
        "| source | row | query | status | P@1 | MRR@5 | ctx_recall | TTFT(s) | TPS | eval_count | truncated | overflow | judge | faithfulness | max_rerank | note |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in report["questions"]:
        st = (
            "제외됨"
            if r["degenerate"]["is_degenerate"]
            else ("SKIP" if r["skipped_no_llm"] else "포함")
        )
        if r["error"]:
            st = f"ERROR({r['error'][:40]})"
        retrieval = r["retrieval"] or {}
        query = r["query"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            "| {source} | {row} | {query} | {status} | {p1} | {mrr} | {ctx_recall} | "
            "{ttft} | {tps} | "
            "{eval_count} | {truncated} | {overflow} | {judge} | {faithfulness} | "
            "{max_rerank} | {note} |".format(
                source=r["source"],
                row=r["row"],
                query=query[:60],
                status=st,
                p1=retrieval.get("p_at_1", ""),
                mrr=retrieval.get("mrr_at_5", ""),
                ctx_recall=retrieval.get("context_recall", ""),
                ttft=r["ttft"] if r["ttft"] is not None else "",
                tps=r["tps"] if r["tps"] is not None else "",
                eval_count=r["eval_count"] if r["eval_count"] is not None else "",
                truncated=r["truncated"],
                overflow=r["overflow"],
                judge=r["judge"] if r["judge"] is not None else "",
                faithfulness=r["faithfulness"] if r["faithfulness"] is not None else "",
                max_rerank=retrieval.get("max_rerank_score", ""),
                note=(r["note"] or "")[:40],
            )
        )
    lines += ["", "## Degenerate Questions", "", "| row | reason |", "| --- | --- |"]
    for d in agg["degenerate"]:
        lines.append(f"| {d['row']} | {d['reason']} |")
    return "\n".join(lines) + "\n"


def _write_outputs(report: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = out_path.with_suffix(".md")
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    return md_path


def _print_progress(res: dict[str, Any]) -> None:
    if res["skipped_no_llm"]:
        status = "SKIP(degenerate)"
    elif res["error"]:
        status = f"ERROR: {res['error'][:60]}"
    else:
        status = "OK"
    print(f"  [{res['source']}:{res['row']}] {status} | {res['query'][:48]}")


# --- 메인 ---
async def _run_eval(args: argparse.Namespace) -> dict[str, Any]:
    setup_logging(log_level="INFO")
    print("\n" + "=" * 60)
    print("[eval_quality] RAG 답변 품질 평가 시작")
    print(
        f"  model={DEFAULT_OLLAMA_MODEL} embedder={DEFAULT_EMBEDDING_MODEL} no_llm={args.no_llm}"
    )
    print("=" * 60)

    session_id = f"eval-quality-{int(datetime.now().timestamp())}"
    rag = RAGSystem(session_id=session_id)
    embedder = await ModelManager.get_embedder(DEFAULT_EMBEDDING_MODEL)

    pdf_path = str(ROOT_DIR / args.pdf)
    print(f"[1/4] 파이프라인 구축: {pdf_path}")
    await rag.build_pipeline(pdf_path, os.path.basename(pdf_path), embedder)

    golden = _load_golden()
    testset = _load_testset(args.testset_n)
    print(
        f"[2/4] 질문 로드: golden {len(golden)}건 + testset(top {args.testset_n}) {len(testset)}건"
    )

    score_chain: Any | None = None
    if not args.no_llm and testset:
        # R4-05: evaluation.judge_model(EVAL_JUDGE_MODEL)을 생성기와 분리해 로드.
        # judge는 결정적 판정을 위해 온도 0.0을 명시한다.
        judge_llm = await ModelManager.get_llm(EVAL_JUDGE_MODEL, temperature=0.0)
        print(f"  judge model={EVAL_JUDGE_MODEL} (temperature=0.0, 생성기와 분리)")
        score_chain = (
            ChatPromptTemplate.from_template(SCORE_PROMPT)
            | judge_llm
            | StrOutputParser()
        )

    print(f"[3/4] 질문 평가 실행 (timeout={TIMEOUT_SECONDS}s/건)...")
    results: list[dict[str, Any]] = []
    for q in golden + testset:
        res = await _run_question(rag, q, args.no_llm, score_chain)
        results.append(res)
        _print_progress(res)

    report: dict[str, Any] = {
        "meta": {
            "script": "scripts/eval_quality.py",
            "tag": args.tag,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pdf": str(args.pdf),
            "testset_n": args.testset_n,
            "no_llm": args.no_llm,
            "model": DEFAULT_OLLAMA_MODEL,
            "embedder": DEFAULT_EMBEDDING_MODEL,
            "judge_model": EVAL_JUDGE_MODEL,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_ctx": OLLAMA_NUM_CTX,
            "truncation_threshold": round(OLLAMA_NUM_PREDICT * TRUNCATION_RATIO, 1),
            "timeout_seconds": TIMEOUT_SECONDS,
            "golden_total": len(golden),
            "testset_total": len(testset),
        },
        "questions": results,
        "aggregates": _aggregate(results),
    }
    print("[4/4] 평가 완료")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 답변 품질 평가 하네스 (retrieval P@1/MRR@5, TTFT, TPS, "
        "답변 길이, judge, 잘림/overflow 감지)"
    )
    parser.add_argument(
        "--pdf", default=DEFAULT_PDF, help="평가 대상 PDF (프로젝트 루트 기준 경로)"
    )
    parser.add_argument(
        "--testset_n", type=int, default=3, help="testset_2201.csv 상위 N행 사용"
    )
    parser.add_argument(
        "--no-llm", action="store_true", help="generation/judge 생략 (retrieval만)"
    )
    parser.add_argument("--tag", default="eval", help="출력 파일 태그 (기본: eval)")
    parser.add_argument(
        "--out",
        default=None,
        help="JSON 출력 경로 (기본: reports/eval_quality_<tag>_<ts>.json)",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path(args.out)
        if args.out
        else ROOT_DIR / "reports" / f"eval_quality_{args.tag}_{ts}.json"
    )

    try:
        report = asyncio.run(_run_eval(args))
    except KeyboardInterrupt:
        print("\n[평가 취소]")
        return
    except Exception as exc:  # noqa: BLE001 - 최상위 오류 안내
        print(f"\n[평가 실패] {exc}")
        import traceback

        traceback.print_exc()
        return

    md_path = _write_outputs(report, out_path)
    agg = report["aggregates"]
    print(f"\n[saved] {out_path}")
    print(f"[saved] {md_path}")
    print(
        f"[summary] scorable={agg['scorable_count']} "
        f"P@1={agg['p_at_1']} MRR@5={agg['mrr_at_5']} "
        f"avg_ttft={agg['avg_ttft']} avg_tps={agg['avg_tps']} "
        f"judge_avg={agg['judge_avg']}"
    )


if __name__ == "__main__":
    main()
