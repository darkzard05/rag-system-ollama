# src/_bg_indexing.py
# 백그라운드 인덱싱 / 모델 교체 로직 전용 모듈 (main.py 로부터 순수 이동).
# 동작 보존(behavior-preserving) — 함수 본문은 이동 전과 동일하다.
"""
백그라운드 RAG 재구축(_bg_rebuild_task) 및 QA Chain 교체(_swap_llm_core,
_bg_update_qa_chain, _update_qa_chain), 업로드 검증 실패 전달(_post_upload_error)
로직을 담당한다. main.py(Streamlit 진입점) 의 이벤트 디스패치/렌더링 코드로부터
분리된 전용 모듈이며, 이 함수들은 common.utils.run_in_background_worker 가 전용
AsyncWorker 이벤트 루프에서 실행한다.
"""

# [Path Bootstrap] 저장소 루트의 src/ 를 sys.path 에 추가하여
# `from common`/`from core` 식의 bare import 가 해석되도록 한다.
# (main.py 와 동일한 관례 — 이 모듈을 standalone 으로 import 하는 경우 대비)
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import asyncio
import logging

from common.config import DEFAULT_OLLAMA_MODEL

# main.py 의 모듈 전역 logger 와 결합을 피하기 위한 이 모듈 자체 로거
logger = logging.getLogger(__name__)


async def _bg_rebuild_task(
    session_id: str, file_path: str, file_name: str, embedder_name: str
):
    """
    [Background Task] 업로드된 파일과 선택된 모델을 사용하여 RAG 파이프라인을 비동기로 재구축합니다.
    """
    from core.model_loader import ModelManager
    from core.rag_core import RAGSystem
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    SessionManager.set("rebuild_done", False, session_id=session_id)
    SessionManager.set("rebuild_error", None, session_id=session_id)
    SessionManager.set(
        "rebuild_status", f"Analyzing '{file_name}'...", session_id=session_id
    )

    # 타임라인에 분석 시작 메시지 추가 (하나의 메시지를 업데이트)
    # on_file_upload가 만든 자리표시자(msg_id=build_{session_id})를 재사용하여
    # 빈 대화 깜빡임 없이 진행 상황이 그 위에 업데이트된다.
    build_msg_id = f"build_{session_id}"
    SessionManager.set("build_msg_id", build_msg_id, session_id=session_id)
    SessionManager.add_message(
        "system",
        f"Analysis started for '{file_name}'",
        msg_type="build_progress",
        msg_id=build_msg_id,
        progress=0,
        status="Preparing analysis...",
        cancelable=True,
        logs=[],
        session_id=session_id,
    )

    try:
        embedder = await ModelManager.get_embedder(embedder_name)

        # Check cancellation before starting build
        if SessionManager.get("rebuild_cancelled", False, session_id=session_id):
            logger.info(f"[MAIN] Rebuild cancelled by user for session {session_id}")
            SessionManager.set("rebuild_cancelled", False, session_id=session_id)
            SessionManager.set("rebuild_progress", 0, session_id=session_id)
            SessionManager.add_status_log(
                "Document analysis cancelled.", session_id=session_id
            )
            SessionManager.add_message(
                "system",
                "Document analysis cancelled",
                msg_type="build_error",
                msg_id=build_msg_id,
                error="cancelled by the user",
                session_id=session_id,
            )
            return

        rag_sys = RAGSystem(session_id=session_id)

        def _report_progress(pct: int, msg: str = ""):
            SessionManager.set("rebuild_progress", pct, session_id=session_id)
            if msg:
                SessionManager.set("rebuild_status", msg, session_id=session_id)
            # 타임라인 진행 메시지 업데이트 (동일 msg_id)
            SessionManager.add_message(
                "system",
                f"Analyzing '{file_name}'...",
                msg_type="build_progress",
                msg_id=build_msg_id,
                progress=pct,
                status=msg or f"Progress {pct}%",
                cancelable=True,
                logs=SessionManager.get("status_logs", [], session_id) or [],
                session_id=session_id,
            )

        SessionManager.set("rebuild_progress", 0, session_id=session_id)

        # Check cancellation again before expensive build_pipeline call
        if SessionManager.get("rebuild_cancelled", False, session_id=session_id):
            logger.info(f"[MAIN] Rebuild cancelled by user for session {session_id}")
            SessionManager.set("rebuild_cancelled", False, session_id=session_id)
            SessionManager.set("rebuild_progress", 0, session_id=session_id)
            SessionManager.add_status_log(
                "Document analysis cancelled.", session_id=session_id
            )
            SessionManager.add_message(
                "system",
                "Document analysis cancelled",
                msg_type="build_error",
                msg_id=build_msg_id,
                error="cancelled by the user",
                session_id=session_id,
            )
            return

        def _is_cancelled() -> bool:
            return bool(
                SessionManager.get("rebuild_cancelled", False, session_id=session_id)
            )

        success_message, cache_used = await rag_sys.build_pipeline(
            file_path=file_path,
            file_name=file_name,
            embedder=embedder,
            on_progress=_report_progress,
            check_cancelled=_is_cancelled,
        )

        SessionManager.set("rebuild_progress", 100, session_id=session_id)
        SessionManager.set("pdf_processed", True, session_id=session_id)
        SessionManager.set("pdf_processing_error", None, session_id=session_id)
        SessionManager.set(
            "doc_stats",
            {
                "file_name": file_name,
                "cache_used": bool(cache_used),
                "embedder": embedder_name,
            },
            session_id=session_id,
        )
        SessionManager.add_status_log(success_message, session_id=session_id)
        # 타임라인 진행 메시지 완료 처리 (동일 msg_id)
        SessionManager.add_message(
            "system",
            success_message,
            msg_type="build_progress",
            msg_id=build_msg_id,
            progress=100,
            status=success_message,
            cancelable=False,
            done=True,
            logs=[],
            session_id=session_id,
        )
        SessionManager.add_message("system", success_message, session_id=session_id)
    except asyncio.CancelledError:
        logger.info(
            f"[MAIN] Rebuild pipeline cancelled mid-build for session {session_id}"
        )
        SessionManager.set("rebuild_cancelled", False, session_id=session_id)
        SessionManager.set("rebuild_progress", 0, session_id=session_id)
        SessionManager.add_message(
            "system",
            "Document analysis cancelled",
            msg_type="build_error",
            msg_id=build_msg_id,
            error="async task cancelled",
            session_id=session_id,
        )
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        error_msg = f"An error occurred while processing the document: {e}"
        SessionManager.set("rebuild_error", error_msg, session_id=session_id)
        SessionManager.set("pdf_processing_error", error_msg, session_id=session_id)
        SessionManager.set("rebuild_progress", 0, session_id=session_id)
        SessionManager.set("pdf_processed", False, session_id=session_id)
        SessionManager.add_message(
            "system",
            "Error during document analysis",
            msg_type="build_error",
            msg_id=build_msg_id,
            error=error_msg,
            session_id=session_id,
        )
        SessionManager.add_message(
            "assistant",
            error_msg,
            msg_type="build_error",
            session_id=session_id,
        )
    finally:
        SessionManager.set("rebuild_done", True, session_id=session_id)
        SessionManager.set("is_building_rag", False, session_id=session_id)


def _swap_llm_core(session_id: str) -> None:
    """LLM(QA Chain)만 교체하는 공통 코어. (R: Group 10 중복 통합)

    ``_bg_update_qa_chain``(async) 와 ``_update_qa_chain``(sync) 가 중복하던
    모델 로드 → 세션 상태 갱신 → 상태 로그 → 오류 처리 → 완료 플래그 로직을
    단일 동기 함수로 통합한다.

    - ``is_swapping_model`` 플래그 관리는 호출자(각 함수)가 담당한다
      (async 는 True/False 를, sync 는 전혀 쓰지 않음 — 동작 보존).
    - ``load_llm`` 은 동기 호출이다. async 진입점은 ``asyncio.to_thread`` 로
      이 함수를 오프로드한다. 전역 세션 설정은 변경하지 않는다(명시적 session_id).
    """
    from core.model_loader import load_llm
    from core.session import SessionManager

    selected_model = SessionManager.get("last_selected_model", session_id=session_id)
    try:
        SessionManager.add_status_log(
            "Switching inference model...", session_id=session_id
        )
        model_name = str(selected_model or DEFAULT_OLLAMA_MODEL)
        llm = load_llm(model_name)
        SessionManager.set("llm", llm, session_id=session_id)
        SessionManager.add_status_log("Inference model switched", session_id=session_id)
    except Exception as e:
        error_msg = f"Failed to update the QA chain: {e}"
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        SessionManager.add_status_log(error_msg, session_id=session_id)
        SessionManager.add_message(
            "assistant",
            error_msg,
            msg_type="build_error",
            session_id=session_id,
        )
    finally:
        SessionManager.set("rag_build_complete_flag", True, session_id=session_id)


async def _bg_update_qa_chain(session_id: str) -> None:
    """
    [Background Task] 문서 인덱싱은 유지한 채 LLM(QA Chain)만 백그라운드로 교체합니다.

    `_bg_rebuild_task`와 동일하게 run_in_background_worker(common.utils)가 전용
    AsyncWorker 이벤트 루프에서 실행한다. load_llm은 동기 호출이므로
    `asyncio.to_thread`로 스레드 풀에 오프로드한다 — AsyncWorker는 단일 루프이며
    스트림 소비자(ui/components/streaming.py)와 공유되므로 루프에서 직접 실행하면
    UI 스트리밍까지 함께 블로킹된다. 실제 스왑은 공용 ``_swap_llm_core`` 위임 (R: Group 10).
    """
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    SessionManager.set("is_swapping_model", True, session_id=session_id)
    try:
        await asyncio.to_thread(_swap_llm_core, session_id)
    finally:
        SessionManager.set("is_swapping_model", False, session_id=session_id)


def _update_qa_chain(session_id: str | None = None) -> None:
    """
    문서 인덱싱은 유지한 채 LLM(QA Chain)만 교체합니다.

    공용 ``_swap_llm_core`` 에 위임한다 (R: Group 10).
    """
    from core.session import SessionManager

    sid = session_id or SessionManager.get_session_id()
    _swap_llm_core(sid)


def _post_upload_error(error_msg: str, session_id: str | None = None) -> None:
    """업로드 검증 실패를 타임라인 메시지로 전달합니다 (메인 레벨 요소 삽입 금지).

    on_file_upload는 _render_app_layout보다 먼저 실행되므로 st.error(...)는
    메인 블록 상단(두 컬럼 위)에 렌더되어 채팅 스크롤러를 아래로 밀어낸다.
    실패 메시지는 세션 상태에 기록하고 chat 컬럼 내부의 타임라인 fragment가
    렌더링하도록 해 레이아웃을 그대로 유지한다. (QA-실패 패턴과 동일)
    """
    from core.session import SessionManager

    sid = session_id or SessionManager.get_session_id()
    # error_msg는 이미 "❌ " 접두사를 포함하므로 중복해서 붙이지 않는다.
    SessionManager.add_status_log(error_msg, session_id=sid)
    SessionManager.add_message(
        "assistant",
        error_msg,
        msg_type="build_error",
        session_id=sid,
    )
