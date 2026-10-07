"""투기적 참조 프리패치: 불변 레퍼런스 플랜 + 주석 prewarm.

``sources`` 이벤트 시점에 ``_build_reference_plan`` 으로 불변 플랜을 1회
계산하고, ``extract_annotations_from_docs`` 의 fitz 작업을 ``AsyncWorker``
백그라운드 루프(``asyncio.to_thread``)에 선행 제출한다. 최종 확정 시점
(``_finalize_pdf_side_effects``)에는 플랜 해시와 최종 citations 을 비교해
일치(hit)하면 prewarm 결과를 채택하고, 불일치(miss)하면 폐기 후 동기
폴백한다. 영속화(persist) → 하이드레이션(hydrate) → 동기화(sync) 순서는
절대 바꾸지 않는다.

좌표 불변식: 루프 격리(백그라운드는 ``SessionManager`` 락을 잡지 않음),
write-behind no-drop(adopt 실패 시 동기 폴백이 항상 주석을 보장),
hydration-before-read(``pdf_annotations`` set은 확정 이후에만 수행).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError as FuturesCancelledError
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from common.utils import doc_stable_id, fast_hash
from ui.components.common import get_doc_metadata

logger = logging.getLogger(__name__)

__all__ = [
    "ReferencePlan",
    "_build_reference_plan",
    "adopt_prewarmed_annotations",
    "discard_reference_plan",
    "note_stream_references",
]

#: prewarm future 대기 상한(초). 초과 시 동기 폴백한다.
PREWARM_WAIT_S = 10.0

#: 레지스트리 상한 (턴 종료 후 정리되지만 방어용으로 유지).
_MAX_ACTIVE_PLANS = 64
_MAX_PREWARM_FUTURES = 16

# 플레이스홀더 취급 섹션명 (chat_references._render_references_content와 동일).
_GENERIC_SECTIONS = frozenset(
    {"general content", "none", "?", "문서 본문", "일반 본문"}
)


@dataclass(frozen=True, slots=True)
class ReferencePlan:
    """sources 이벤트 시점에 1회 계산되는 불변 레퍼런스 플랜.

    모든 매핑은 ``MappingProxyType`` 으로 동결되어 변이를 시도하면
    ``TypeError`` 가 발생한다.     ``plan_hash`` 는 documents 서명과 citations
    서명을 함께 묶으며, 최종 확정 시점의 서명과 비교해 hit/miss를 판정한다.
    """

    plan_hash: str
    docs_signature: str
    pages: tuple[int, ...]
    page_to_refs: Mapping[int, Mapping[str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    page_to_excerpt: Mapping[int, str] = field(
        default_factory=lambda: MappingProxyType({})
    )
    labels: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))
    button_keys: Mapping[int, str] = field(default_factory=lambda: MappingProxyType({}))


_ACTIVE_PLANS: dict[tuple[str, str], ReferencePlan] = {}
_PREWARM_FUTURES: dict[str, Future[list[dict[str, Any]]]] = {}
_registry_lock = threading.Lock()

#: done 콜백 중복 등록 방지 (호출자는 ``_registry_lock`` 보유).
_PREWARM_CALLBACK_ATTACHED: set[str] = set()


def _normalize_text(text: object) -> str:
    """중복 판별용 본문 정규화: 연속 공백 축소 + 소문자화."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_section(raw: str) -> str:
    """의미 없는 플레이스홀더 섹션명을 빈 문자열로 정규화한다."""
    stripped = (raw or "").strip()
    if not stripped or stripped.lower() in _GENERIC_SECTIONS:
        return ""
    return re.sub(
        r"[\(\[]\s*p(?:age)?\.?\s*\d+\s*[\)\]]",
        "",
        stripped,
        flags=re.IGNORECASE,
    ).strip()


def _doc_page(doc: Any) -> int:
    """문서 메타데이터의 페이지 번호 (실패 시 1)."""
    try:
        return int(get_doc_metadata(doc).get("page", 1))
    except (ValueError, TypeError, AttributeError):
        return 1


def _doc_file_key(doc: Any) -> str:
    """memo 키 호환용 파일 식별자 (file_hash 우선, 없으면 경로)."""
    meta = get_doc_metadata(doc)
    file_hash = meta.get("file_hash")
    if file_hash:
        return str(file_hash)
    return str(meta.get("file_path") or meta.get("source") or "")


def _doc_content(doc: Any) -> str:
    """Document 객체/dict 양식을 모두 지원하는 본문 추출."""
    content = getattr(doc, "page_content", None)
    if content is None and isinstance(doc, dict):
        content = doc.get("page_content")
    return content if isinstance(content, str) else ""


def _docs_signature(documents: list[Any]) -> str:
    """문서 집합 서명: file key + page + 정규 본문 해시 (순서 무관)."""
    parts: list[str] = []
    for doc in documents or []:
        norm = _normalize_text(_doc_content(doc))
        parts.append(
            "|".join(
                [
                    doc_stable_id(doc),
                    _doc_file_key(doc),
                    str(_doc_page(doc)),
                    fast_hash(norm),
                ]
            )
        )
    return fast_hash(json.dumps(sorted(parts), sort_keys=True))


def _citations_signature(citations: list[dict[str, Any]]) -> str:
    """인용 집합 서명: doc_id + page + section + span 해시 (순서 무관)."""
    parts: list[str] = []
    for cit in citations or []:
        if not isinstance(cit, dict) or cit.get("doc_id") is None:
            continue
        parts.append(
            "|".join(
                [
                    str(cit.get("doc_id")),
                    str(cit.get("page") or ""),
                    str(cit.get("section") or ""),
                    fast_hash(_normalize_text(cit.get("text_span"))),
                ]
            )
        )
    return fast_hash(json.dumps(sorted(parts), sort_keys=True))


def _build_reference_plan(
    documents: list[Any],
    citations: list[dict[str, Any]],
    msg_id: str = "",
) -> ReferencePlan:
    """불변 레퍼런스 플랜을 계산한다 (순수 함수, 락 없음).

    ``chat_references._render_references_content`` 의 그룹화(페이지 기준
    1버튼, Branch-D 텍스트 dedup, 구체적 섹션명 보존)와 동일한 규칙으로
    ``page_to_refs`` / ``page_to_excerpt`` / ``labels`` / ``button_keys``
    를 만든다. Streamlit 호출 없음 — 렌더는 호출자가 담당한다.
    """
    docs = list(documents or [])
    cits = [c for c in (citations or []) if isinstance(c, dict)]

    doc_citations = [c for c in cits if c.get("doc_id") is not None]
    if not doc_citations and docs:
        seen_keys: set[tuple[str, int]] = set()
        fallback: list[dict[str, Any]] = []
        for d in docs:
            meta = get_doc_metadata(d)
            sec = str(meta.get("current_section", "문서 본문"))
            page = _doc_page(d)
            key = (sec, page)
            if key not in seen_keys:
                seen_keys.add(key)
                fallback.append(
                    {"doc_id": doc_stable_id(d), "section": sec, "page": page}
                )
        doc_citations = fallback

    doc_page_map: dict[str, int] = {}
    for d in docs:
        page_num = _doc_page(d)
        doc_page_map.setdefault(doc_stable_id(d), page_num)

    doc_text_map: dict[str, str] = {}
    for d in docs:
        sid = doc_stable_id(d)
        if sid not in doc_text_map:
            doc_text_map[sid] = _normalize_text(_doc_content(d))

    doc_raw_map: dict[str, str] = {}
    for d in docs:
        sid = doc_stable_id(d)
        if sid not in doc_raw_map:
            doc_raw_map[sid] = _doc_content(d)

    targets: list[tuple[int, dict[str, Any]]] = []
    for cit in doc_citations:
        sid = str(cit.get("doc_id"))
        target_p = cit.get("page") or doc_page_map.get(sid, 1)
        try:
            target_p = int(target_p)  # type: ignore[assignment]
        except (ValueError, TypeError):
            target_p = 1
        targets.append((target_p, cit))
    targets.sort(key=lambda item: item[0])

    page_to_refs: dict[int, dict[str, str]] = {}
    page_to_excerpt: dict[int, str] = {}
    seen_text_hashes: dict[str, int] = {}
    for target_p, cit in targets:
        sid = str(cit.get("doc_id"))
        clean_sec = _clean_section(str(cit.get("section") or ""))
        raw_sec = str(cit.get("section") or "").strip()

        span = cit.get("text_span")
        norm = (
            _normalize_text(span)
            if isinstance(span, str)
            else doc_text_map.get(sid, "")
        )
        span_raw = span if isinstance(span, str) and span else doc_raw_map.get(sid, "")
        if target_p not in page_to_excerpt and span_raw:
            page_to_excerpt[target_p] = span_raw
        if norm:
            text_hash = fast_hash(norm)
            if text_hash in seen_text_hashes:
                kept = seen_text_hashes[text_hash]
                prev_sec = page_to_refs[kept]["section"]
                cand_sec = clean_sec or raw_sec
                if cand_sec and len(cand_sec) > len(prev_sec):
                    page_to_refs[kept]["section"] = cand_sec
                continue
            seen_text_hashes[text_hash] = target_p

        if target_p not in page_to_refs:
            page_to_refs[target_p] = {"sid": sid, "section": clean_sec}
        elif not page_to_refs[target_p]["section"] and clean_sec:
            page_to_refs[target_p]["section"] = clean_sec

    sorted_pages = sorted(page_to_refs.keys())
    labels: dict[int, str] = {}
    button_keys: dict[int, str] = {}
    for idx, page_num in enumerate(sorted_pages):
        sec_title = page_to_refs[page_num]["section"]
        labels[page_num] = (
            f"{sec_title} · p.{page_num}" if sec_title else f"p.{page_num}"
        )
        button_keys[page_num] = f"pop_doc_{msg_id}_{page_num}_{idx}"

    frozen_refs: dict[int, Mapping[str, str]] = {
        page: MappingProxyType(dict(info)) for page, info in page_to_refs.items()
    }
    docs_sig = _docs_signature(docs)
    cits_sig = _citations_signature(cits)
    plan_hash = fast_hash(
        json.dumps({"docs": docs_sig, "cits": cits_sig}, sort_keys=True)
    )
    return ReferencePlan(
        plan_hash=plan_hash,
        docs_signature=docs_sig,
        pages=tuple(sorted_pages),
        page_to_refs=MappingProxyType(frozen_refs),
        page_to_excerpt=MappingProxyType(dict(page_to_excerpt)),
        labels=MappingProxyType(dict(labels)),
        button_keys=MappingProxyType(dict(button_keys)),
    )


def _snapshot_documents(documents: list[Any]) -> list[dict[str, Any]]:
    """prewarm용 불변 스냅샷: 세션 객체와 공유하지 않는 순수 dict 복사.

    ``extract_annotations_from_docs`` 는 dict 양식(``page_content`` /
    ``metadata``)을 지원하므로, 원본 Document 객체를 백그라운드 스레드와
    공유하지 않고 값만 복사해 락-프리로 전달한다.
    """
    snapshot: list[dict[str, Any]] = []
    for doc in documents or []:
        meta = get_doc_metadata(doc)
        snapshot.append({"page_content": _doc_content(doc), "metadata": dict(meta)})
    return snapshot


def _evict_if_needed_locked() -> None:
    """레지스트리 상한 방어 (호출자는 ``_registry_lock`` 보유)."""
    while len(_ACTIVE_PLANS) > _MAX_ACTIVE_PLANS:
        _ACTIVE_PLANS.pop(next(iter(_ACTIVE_PLANS)))
    while len(_PREWARM_FUTURES) > _MAX_PREWARM_FUTURES:
        _PREWARM_FUTURES.pop(next(iter(_PREWARM_FUTURES)))


def note_stream_references(
    sid: str,
    msg_id: str,
    documents: list[Any],
    citations: list[dict[str, Any]],
    *,
    extractor: Callable[[list[Any]], list[dict[str, Any]]] | None = None,
) -> ReferencePlan | None:
    """sources/citations 이벤트에서 플랜 계산 + prewarm 1회 제출.

    동일 ``docs_signature`` 에는 fitz 작업을 단 1회만 제출하고(중복 제출
    방지), 플랜은 최신 citations 기준으로 갱신한다. PDF 미로드 세션에는
    아무 것도 하지 않는다. 세션 락을 잡지 않으며, 투기적 최적화이므로 어떤
    실패에도 스트리밍을 깨지 않고 ``None`` 을 반환한다.
    """
    from common.async_worker import AsyncWorker
    from core.session import SessionManager

    try:
        if not SessionManager.get("pdf_file_path", "", sid):
            return None
        plan = _build_reference_plan(documents, citations, msg_id)
    except Exception as exc:
        logger.debug("[PREFETCH] 플랜 계산 생략: %s", exc)
        return None
    with _registry_lock:
        _ACTIVE_PLANS[(sid, msg_id)] = plan
        if plan.docs_signature in _PREWARM_FUTURES:
            return plan
        snapshot = _snapshot_documents(documents)
        if extractor is None:
            from common.utils import extract_annotations_from_docs

            extractor = extract_annotations_from_docs

        async def _prewarm() -> list[dict[str, Any]]:
            return await asyncio.to_thread(extractor, snapshot)

        try:
            future = AsyncWorker().submit(_prewarm())
        except RuntimeError as exc:
            logger.warning("[PREFETCH] prewarm 제출 실패: %s", exc)
            return plan
        _PREWARM_FUTURES[plan.docs_signature] = future
        _evict_if_needed_locked()
    return plan


def _drop_prewarm_future_locked(
    fut: Future[list[dict[str, Any]]], docs_signature: str
) -> None:
    """레지스트리에서 prewarm future 제거 (호출자는 ``_registry_lock`` 보유)."""
    if _PREWARM_FUTURES.get(docs_signature) is fut:
        _PREWARM_FUTURES.pop(docs_signature, None)
    _PREWARM_CALLBACK_ATTACHED.discard(docs_signature)


def _make_prewarm_done_callback(
    sid: str, docs_signature: str
) -> Callable[[Future[list[dict[str, Any]]]], None]:
    """prewarm 완료 시 ``SessionManager.pdf_annotations`` 를 채운다.

    워커 스레드에서 실행되므로 Streamlit 호출(``st.rerun`` 포함) 금지:
    뷰어 fragment(``render_pdf_area`` → ``render_pdf_viewer``)가 다음
    fragment 실행(참조 클릭 경로 ``_page_scoped_annotations``)에서 값을
    읽어 rect를 표시한다. 판정 후 future는 레지스트리에서 제거한다.
    """

    def _on_done(fut: Future[list[dict[str, Any]]]) -> None:
        from core.session import SessionManager

        try:
            annotations = fut.result(timeout=0)
        except (FutureTimeoutError, TimeoutError) as exc:
            logger.info("[PREFETCH] prewarm 미완료로 폐기: %s", exc)
            with _registry_lock:
                _drop_prewarm_future_locked(fut, docs_signature)
            return
        except (asyncio.CancelledError, FuturesCancelledError) as exc:
            logger.info("[PREFETCH] prewarm 취소로 폐기: %s", exc)
            with _registry_lock:
                _drop_prewarm_future_locked(fut, docs_signature)
            return
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning("[PREFETCH] prewarm 백그라운드 실패로 폐기: %s", exc)
            with _registry_lock:
                _drop_prewarm_future_locked(fut, docs_signature)
            return
        with _registry_lock:
            _drop_prewarm_future_locked(fut, docs_signature)
        try:
            file_hash = SessionManager.get("file_hash", None, sid)
            SessionManager.set(
                "pdf_annotations",
                {"file_hash": file_hash, "annotations": annotations},
                sid,
            )
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            logger.warning("[PREFETCH] prewarm 하이드레이션 실패: %s", exc)

    return _on_done


def adopt_prewarmed_annotations(
    sid: str,
    msg_id: str,
    documents: list[Any],
    citations: list[dict[str, Any]],
    *,
    wait_s: float = PREWARM_WAIT_S,
) -> list[dict[str, Any]] | None:
    """최종 확정 시점의 hit/miss 판정 (렌더 스레드 논블로킹).

    활성 플랜의 ``plan_hash`` 가 최종 서명과 일치하고 prewarm future 가
    확보되면 그 결과를 반환(hit)한다. 불일치(miss)하면 플랜을 폐기하고
    ``None`` 을 반환 — 호출자는 반드시 동기 폴백을 수행한다.

    렌더 스레드는 절대 대기하지 않는다: ``future.result(timeout=0)`` 으로
    즉시 판정하고, 미완료 시 done 콜백을 등록한 뒤 ``None`` 을 반환한다
    (rect 없이 렌더). 콜백이 완료 시 ``SessionManager.pdf_annotations``
    를 채우면 뷰어 fragment가 클릭 경로에서 rect를 표시한다. future는
    완료/취소 판정 시점에만 레지스트리에서 제거한다 (pop-then-wait 유실
    방지). ``wait_s`` 는 하위 호환용으로 유지되며 무시된다.
    """
    _ = wait_s  # 하위 호환 시그니처: 렌더 스레드 대기에 사용하지 않음
    _t_join = time.perf_counter()
    expected = _build_reference_plan(documents, citations, msg_id)
    with _registry_lock:
        active = _ACTIVE_PLANS.get((sid, msg_id))
        if active is None or active.plan_hash != expected.plan_hash:
            _ACTIVE_PLANS.pop((sid, msg_id), None)
            return None
        future = _PREWARM_FUTURES.get(active.docs_signature)
        docs_signature = active.docs_signature
        _ACTIVE_PLANS.pop((sid, msg_id), None)
    if future is None:
        return None
    try:
        annotations = future.result(timeout=0)
    except (FutureTimeoutError, TimeoutError):
        with _registry_lock:
            if docs_signature not in _PREWARM_CALLBACK_ATTACHED:
                _PREWARM_CALLBACK_ATTACHED.add(docs_signature)
                future.add_done_callback(
                    _make_prewarm_done_callback(sid, docs_signature)
                )
        logger.info(
            "[PERF] join-block msg_id=%s took %.3fs prewarmed=pending",
            msg_id,
            time.perf_counter() - _t_join,
        )
        return None
    except (asyncio.CancelledError, FuturesCancelledError) as exc:
        logger.info("[PREFETCH] prewarm 취소로 동기 폴백: %s", exc)
        with _registry_lock:
            _drop_prewarm_future_locked(future, docs_signature)
        return None
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        logger.warning("[PREFETCH] prewarm 실패로 동기 폴백: %s", exc)
        with _registry_lock:
            _drop_prewarm_future_locked(future, docs_signature)
        return None
    with _registry_lock:
        _drop_prewarm_future_locked(future, docs_signature)
    logger.info(
        "[PERF] join-block msg_id=%s took %.3fs prewarmed=hit",
        msg_id,
        time.perf_counter() - _t_join,
    )
    return annotations


def discard_reference_plan(sid: str, msg_id: str) -> None:
    """해당 턴의 활성 플랜을 폐기한다 (취소/오류 턴 정리용)."""
    with _registry_lock:
        _ACTIVE_PLANS.pop((sid, msg_id), None)
