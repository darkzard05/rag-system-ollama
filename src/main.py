# Streamlit 기반 RAG 챗봇의 메인 진입점 및 전체 오케스트레이션을 담당하는 파일
"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

# [Lazy Import용] 런타임에 필요한 모듈들
import atexit
import contextlib
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Literal, cast

import nest_asyncio
import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from common.constants import FilePathConstants, StringConstants
from common.logging_config import setup_logging
from common.utils import safe_cache_resource, sync_run

# 1. Streamlit 페이지 설정 (최우선 실행 - 가이드라인 준수)
st.set_page_config(
    page_title=StringConstants.PAGE_TITLE,
    layout=cast(Literal["centered", "wide"], StringConstants.LAYOUT),
)

# 2. 로깅 설정 (최상단)
logger = setup_logging(log_level="DEBUG", log_file=Path("logs/app.log"))

# 상수 정의
PAGE_TITLE = StringConstants.PAGE_TITLE
LAYOUT = StringConstants.LAYOUT
MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# 비동기 패치 적용
nest_asyncio.apply()

# --- [추가] 필수 세션 상태 초기화 ---
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "pdf_window_start" not in st.session_state:
    st.session_state.pdf_window_start = 1
if "pdf_target_page" not in st.session_state:
    st.session_state.pdf_target_page = None
if "last_valid_height" not in st.session_state:
    st.session_state.last_valid_height = 800
if "is_generating_answer" not in st.session_state:
    st.session_state.is_generating_answer = False


def _check_windows_integrity():
    """
    [Background] Windows 환경의 라이브러리 충돌을 체크하고 주기적으로 세션을 정리합니다.
    """
    # [최적화] 세션 정리 추가 (메모리 누수 방지)
    try:
        from core.session import SessionManager

        # 1시간 이상 활동 없는 세션 정리 (물리적 파일 삭제 포함)
        SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
    except Exception as e:
        logger.error(f"[SYSTEM] [CLEANUP] 세션 정리 중 오류: {e}")

    # [최적화] CI 환경에서는 무거운 라이브러리 체크 생략 (충돌 위험 방지)
    import platform

    if platform.system() != "Windows" or os.getenv("GITHUB_ACTIONS") == "true":
        return

    try:
        # 무거운 라이브러리 로드 테스트 (핵심 RAG용)
        import sys

        if "torch" not in sys.modules:
            import torch

            _ = torch.tensor([1.0])

        with contextlib.suppress(ValueError, RuntimeError):
            logger.info("[SYSTEM] [INTEGRITY] Windows 라이브러리 무결성 점검 완료 (OK)")

    except Exception as e:
        logger.warning(f"[SYSTEM] [INTEGRITY] 점검 중 예외 발생: {e}")


@st.cache_resource
def _start_global_background_worker():
    """
    [Singleton] 서버 인스턴스당 단 하나만 실행되는 백그라운드 워커입니다.
    세션 정리 및 시스템 무결성 점검을 주기적으로 수행합니다.
    """

    def maintenance_loop():
        logger.info("[SYSTEM] 전역 백그라운드 워커 시작됨")

        while True:
            try:
                _check_windows_integrity()
            except Exception as e:
                logger.error(f"[SYSTEM] 백그라운드 워커 루프 중 오류: {e}")

            # 1시간(3600초) 대기 후 반복
            time.sleep(3600)

    thread = threading.Thread(
        target=maintenance_loop, name="GlobalMaintenanceWorker", daemon=True
    )
    thread.start()
    return thread


def _run_background_checks():
    """백그라운드 점검 작업을 시작합니다 (싱글톤 보장)."""
    _start_global_background_worker()


@st.cache_data(ttl=300)  # 5분간 캐싱
def _get_available_models_cached():
    """Ollama 모델 목록을 캐싱하여 UI 블로킹을 최소화합니다."""
    from core.model_loader import get_available_models

    return get_available_models()


@safe_cache_resource(show_spinner=False)
def _init_temp_directory():
    """임시 디렉토리를 초기화합니다."""
    from common.constants import FilePathConstants

    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    temp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SYSTEM] [INIT] 임시 디렉토리 준비 완료: {temp_path}")
    return str(temp_path)


def _cleanup_current_file():
    """현재 세션에서 사용 중인 임시 파일을 삭제합니다. (종료 핸들러용)"""
    from core.session import SessionManager

    # Streamlit 세션 상태를 직접 접근하기 어려우므로 SessionManager는 thread-safe하게 설계됨
    try:
        # [수정] create=False로 설정하여 종료 시 불필요한 세션 생성 및 로깅 오류 방지
        path = SessionManager.get("pdf_file_path", create=False)
        if path:
            SessionManager.safe_remove_file(path)
    except Exception:
        pass


@st.cache_resource
def _register_cleanup_handlers():
    """[Singleton] 프로세스 종료 핸들러를 단 한 번만 등록합니다."""
    atexit.register(_cleanup_current_file)
    logger.info("[SYSTEM] 프로세스 종료 핸들러 등록 완료")
    return True


# 앱 시작 시 초기화 수행 (캐싱으로 인해 최초 1회만 작동)
_init_temp_directory()
_run_background_checks()
_register_cleanup_handlers()


def _ensure_models_are_loaded() -> bool:
    """
    선택된 LLM 및 임베딩 모델을 중앙 관리자를 통해 안전하게 로드합니다.
    """
    from core.model_loader import ModelManager
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

    selected_model = SessionManager.get("last_selected_model")
    selected_embedding = SessionManager.get("last_selected_embedding_model")

    if not selected_model:
        selected_model = DEFAULT_OLLAMA_MODEL
        SessionManager.set("last_selected_model", selected_model)

    if not selected_embedding:
        selected_embedding = DEFAULT_EMBEDDING_MODEL
        SessionManager.set("last_selected_embedding_model", selected_embedding)

    try:
        # 1. 임베딩 모델 로드 (ModelManager 사용)
        SystemNotifier.loading("임베딩 모델 준비 중...")
        embedder = sync_run(ModelManager.get_embedder(selected_embedding))
        SessionManager.set("embedder", embedder)

        # [수정] 모델 타입에 따른 디바이스 정보 추출 안전성 강화
        if hasattr(embedder, "model_kwargs"):
            actual_device = embedder.model_kwargs.get("device", "UNKNOWN").upper()
        else:
            # OllamaEmbeddings 등 원격/추상화된 백엔드인 경우
            actual_device = "OLLAMA"

        display_device = "GPU" if actual_device == "CUDA" else actual_device
        SystemNotifier.success(f"임베딩 모델 준비 완료 ({display_device})")

        # 2. LLM 로드 (ModelManager 사용)
        SystemNotifier.loading(f"추론 모델({selected_model}) 준비 중...")
        llm = sync_run(ModelManager.get_llm(selected_model))
        SessionManager.set("llm", llm)
        SystemNotifier.success("추론 모델 준비 완료")

        return True

    except Exception as e:
        logger.error(f"모델 로드 중 치명적 오류 발생: {e}", exc_info=True)
        # [수정] st.error 대신 스레드 안전한 알림 시스템 사용 (Thread-Safety 보장)
        SystemNotifier.error("모델 로드 실패", details=str(e))
        return False


def _rebuild_rag_system(session_id: str | None = None) -> None:
    """
    업로드된 파일과 선택된 모델을 사용하여 RAG 파이프라인을 백그라운드에서 재구축합니다.
    """
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

    if session_id:
        SessionManager.set_session_id(session_id)
    else:
        session_id = SessionManager.get_session_id()
    file_name = SessionManager.get("last_uploaded_file_name")
    file_path = SessionManager.get("pdf_file_path")

    if not file_name or not file_path:
        return

    # [수정] 호출부(_handle_pending_tasks)에서 이미 플래그를 관리하므로 내부 중복 체크는 로그로 대체
    logger.debug(f"[SYSTEM] RAG 파이프라인 구축 시작: {file_name}")

    if (
        SessionManager.get("pdf_processed")
        and not SessionManager.get("pdf_processing_error")
        and SessionManager.get("file_hash") is not None
    ):
        return

    SessionManager.set("is_building_rag", True)
    try:
        if not _ensure_models_are_loaded():
            return

        embedder = SessionManager.get("embedder")
        SystemNotifier.loading(f"'{file_name}' 분석 중...")

        # [Lazy Import]
        from core.rag_core import RAGSystem

        rag_sys = RAGSystem(session_id=SessionManager.get_session_id())

        # RAG 파이프라인 빌드 (내부에서 상세 로그 기록)
        success_message, cache_used = sync_run(
            rag_sys.build_pipeline(
                file_path=file_path, file_name=file_name, embedder=embedder
            )
        )

        # 상태 명시적 업데이트
        SessionManager.set("pdf_processed", True)
        SessionManager.add_status_log(f"✅ {success_message}")
        SessionManager.add_message("system", success_message)
        SessionManager.add_message("system", "READY_FOR_QUERY")

        logger.info(f"[SYSTEM] RAG 빌드 완료: {file_name}")

    except Exception as e:
        logger.error(f"RAG 빌드 실패: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.set("pdf_processed", True)
        SessionManager.add_message("system", f"❌ {error_msg}")
    finally:
        SessionManager.set("is_building_rag", False)
        SessionManager.set("rag_build_complete_flag", True)


def _update_qa_chain(session_id: str | None = None) -> None:
    """
    문서 인덱싱은 유지한 채 LLM(QA Chain)만 교체합니다.
    """
    from core.session import SessionManager

    if session_id:
        SessionManager.set_session_id(session_id)

    selected_model = SessionManager.get("last_selected_model")
    try:
        SessionManager.add_status_log("🔄 추론 모델 교체 중")

        # [Lazy Import]
        from core.model_loader import load_llm

        model_name = str(selected_model or DEFAULT_OLLAMA_MODEL)
        llm = load_llm(model_name)
        SessionManager.set("llm", llm)
        SessionManager.add_status_log("✅ 추론 모델 교체 완료")

        logger.info(f"LLM updated to: {selected_model}")
        msg = "✅ 추론 모델이 업데이트되었습니다."
        SessionManager.add_message("system", msg)

    except Exception as e:
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        SessionManager.add_message("assistant", f"❌ 업데이트 실패: {e}")
    finally:
        # UI 리런 유도를 위한 플래그 설정 (global-status-bar에서 감지)
        SessionManager.set("rag_build_complete_flag", True)


# --- Callbacks ---
def on_file_upload() -> None:
    """파일 업로드 이벤트 콜백"""
    from core.session import SessionManager
    from infra.notification_system import SystemNotifier

    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is None or not hasattr(uploaded_file, "type"):
        return

    # [개선] 파일 타입 검사 (MIME 타입 확인)
    if uploaded_file.type != "application/pdf":
        st.error("❌ 올바른 PDF 파일이 아닙니다. PDF 형식의 파일을 업로드해주세요.")
        return

    # [개선] 파일 크기 검사
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"❌ 파일 크기가 너무 큽니다 ({file_size_mb:.2f} MB). {MAX_FILE_SIZE_MB}MB 이하의 파일을 업로드해주세요."
        )
        return

    # 파일이 변경된 경우에만 처리
    if uploaded_file.name != SessionManager.get("last_uploaded_file_name"):
        # [최적화] 이전 문서 상태 강제 초기화 (에러 방지)
        st.session_state.pdf_page_index = 1
        st.session_state.pdf_annotations = []
        if "active_ref_id" in st.session_state:
            st.session_state.active_ref_id = None
        SessionManager.set("current_page", 1)

        # [관리강화] 이전 임시 파일 즉시 삭제
        old_path = SessionManager.get("pdf_file_path")
        if old_path:
            SessionManager.safe_remove_file(old_path)

        SessionManager.set("last_uploaded_file_name", uploaded_file.name)

        # [전용 폴더 사용] 안정적인 임시 파일 생성
        try:
            # 절대 경로로 변환
            temp_dir = os.path.abspath(FilePathConstants.TEMP_DIR)
            os.makedirs(temp_dir, exist_ok=True)

            # [개선] 파일명에 세션 ID와 타임스탬프를 넣어 충돌 방지
            sid = SessionManager.get_session_id()
            safe_name = f"upload_{sid}_{int(time.time())}.pdf"
            tmp_path = os.path.join(temp_dir, safe_name)

            with open(tmp_path, "wb") as f:
                # [개선] 메모리 버퍼 대신 스트림 복사 사용
                shutil.copyfileobj(uploaded_file, f)

            SessionManager.set("pdf_file_path", tmp_path)
            SessionManager.set("new_file_uploaded", True)
            SystemNotifier.success(f"문서 업로드 완료: {uploaded_file.name}", icon="📄")
            SystemNotifier.info("문서 내용 분석 및 인덱싱 시작")
            logger.info(f"[SYSTEM] [UPLOAD] 파일 저장 완료: {tmp_path}")
        except Exception as e:
            SystemNotifier.error("파일 저장 중 오류 발생", details=str(e))


def on_model_change() -> None:
    """LLM 모델 변경 이벤트 콜백"""
    from core.session import SessionManager

    selected = st.session_state.get("model_selector")
    last = SessionManager.get("last_selected_model")

    if not selected or "---" in selected or selected == last:
        return

    if not SessionManager.get("is_first_run"):
        SessionManager.add_message("system", "🔄 추론 모델 변경 요청")

    SessionManager.set("last_selected_model", selected)
    # 이미 문서가 처리된 상태라면 QA 체인만 업데이트하면 됨
    if SessionManager.get("pdf_processed"):
        SessionManager.set("needs_qa_chain_update", True)


def on_embedding_change() -> None:
    """임베딩 모델 변경 이벤트 콜백"""
    from core.session import SessionManager

    selected = st.session_state.get("embedding_model_selector")
    last = SessionManager.get("last_selected_embedding_model")

    if not selected or selected == last:
        return

    if not SessionManager.get("is_first_run"):
        SessionManager.add_message("system", "🔄 임베딩 모델 변경 요청")

    SessionManager.set("last_selected_embedding_model", selected)
    # 임베딩 모델이 바뀌면 문서를 다시 인덱싱해야 함
    if SessionManager.get("pdf_file_path"):
        SessionManager.set("needs_rag_rebuild", True)


def _render_app_layout(available_models: list[str] | None = None) -> None:
    """앱의 전체 레이아웃을 렌더링합니다. (사이드바 설정 + 메인 2열: PDF + 채팅)"""
    from core.session import SessionManager
    from ui.ui import (
        render_global_status_bar,
        render_left_column,
    )

    # [추가] 1초 주기 상태 업데이트 및 리런 트리거 활성화
    render_global_status_bar()

    # [개선] 상단 툴바 레이아웃 (상태바 + 설정 팝오버)
    from ui.components.sidebar import render_settings_content

    with st.container():
        col_status, col_settings = st.columns([0.85, 0.15])
        with col_status:
            render_global_status_bar()
        with col_settings, st.popover("⚙️ 설정", use_container_width=True):
            render_settings_content(
                file_uploader_callback=on_file_upload,
                model_selector_callback=on_model_change,
                embedding_selector_callback=on_embedding_change,
                is_generating=bool(SessionManager.get("is_generating_answer", False)),
                current_file_name=SessionManager.get("last_uploaded_file_name"),
                available_models=available_models,
            )

    # 2. 메인 영역을 2열로 구성 (동일 비율 1:1)
    col_pdf, col_chat = st.columns([1, 1], gap="medium")

    with col_pdf:
        from ui.components.viewer import render_pdf_column

        render_pdf_column()

    with col_chat:
        render_left_column()


def _handle_pending_tasks() -> None:
    """지연된 무거운 작업(RAG 빌드, 모델 교체 등)을 순차적으로 처리합니다."""
    from core.session import SessionManager

    # 중복 실행 방지를 위한 사전 체크
    is_building = bool(SessionManager.get("is_building_rag", False))

    # 1. 새 파일 업로드 처리
    if SessionManager.get("new_file_uploaded") and not is_building:
        logger.info("[SYSTEM] 새 파일 업로드 감지 -> 처리 시작")
        # 즉시 플래그 해제 (중복 실행 방지)
        SessionManager.set("new_file_uploaded", False)
        # 스레드 기동 전 미리 플래그 설정 (Race Condition 방지)
        SessionManager.set("is_building_rag", True)

        current_file_path = SessionManager.get("pdf_file_path")
        current_file_name = SessionManager.get("last_uploaded_file_name")

        # 기본 상태 초기화 (필요한 경로 정보는 유지)
        SessionManager.reset_for_new_file()
        SessionManager.set("pdf_file_path", current_file_path)
        SessionManager.set("last_uploaded_file_name", current_file_name)
        # reset_for_new_file에서 is_building_rag가 꺼졌을 수 있으므로 다시 켬
        SessionManager.set("is_building_rag", True)

        # RAG 구축 실행 (백그라운드 스레드)
        sid = SessionManager.get_session_id()
        thread = threading.Thread(target=_rebuild_rag_system, args=(sid,), daemon=True)
        add_script_run_ctx(thread)
        thread.start()
        logger.info("[SYSTEM] RAG 구축 백그라운드 스레드 시작됨")

    # 2. 모델 재빌드 요청 처리
    elif SessionManager.get("needs_rag_rebuild") and not is_building:
        logger.info("[SYSTEM] RAG 재빌드 요청 수락")
        SessionManager.set("needs_rag_rebuild", False)
        SessionManager.set("is_building_rag", True)

        sid = SessionManager.get_session_id()
        thread = threading.Thread(target=_rebuild_rag_system, args=(sid,), daemon=True)
        add_script_run_ctx(thread)
        thread.start()

    # 3. QA 체인 업데이트 처리
    elif SessionManager.get("needs_qa_chain_update"):
        logger.info("[SYSTEM] QA 체인 업데이트 요청 수락")
        SessionManager.set("needs_qa_chain_update", False)

        # [수정] UI 블로킹 방지를 위해 백그라운드 스레드에서 실행
        sid = SessionManager.get_session_id()
        thread = threading.Thread(target=_update_qa_chain, args=(sid,), daemon=True)
        add_script_run_ctx(thread)
        thread.start()
        logger.info("[SYSTEM] QA 체인 업데이트 백그라운드 스레드 시작됨")


def main() -> None:
    """메인 애플리케이션 오케스트레이터"""
    from core.session import SessionManager
    from ui.ui import inject_custom_css

    # 0. 실행 지표 초기화
    if "full_run_count" not in st.session_state:
        st.session_state.full_run_count = 0
    st.session_state.full_run_count += 1

    # 1. 모델 목록 로딩 (캐싱 적용)
    if "available_models_list" not in st.session_state:
        with st.spinner("시스템 초기화 중..."):
            fetched_models = _get_available_models_cached()
            from common.config import DEFAULT_OLLAMA_MODEL

            if not fetched_models or (
                len(fetched_models) == 1 and "서버" in fetched_models[0]
            ):
                st.session_state.available_models_list = [DEFAULT_OLLAMA_MODEL]
            else:
                st.session_state.available_models_list = fetched_models
        # 목록 확보 후 즉시 리런하여 전체 UI 구성
        st.rerun()

    # 2. 세션 즉시 준비
    SessionManager.init_session()

    # [수정] PDF 업로드 상태에 따른 사이드바 확장 상태 결정 후 CSS 주입 (단 1회 수행)
    is_expanded = bool(SessionManager.get("pdf_file_path"))
    inject_custom_css(is_expanded=is_expanded)

    # [추가] 세션 ID 불일치로 인한 '영구 분석 중' 상태 방지
    if SessionManager.get("pdf_file_path") and not SessionManager.get("pdf_processed"):
        # 분석이 중단된 것으로 간주하고 입력창 열기
        SessionManager.set("is_generating_answer", False)

    # [추가] 리런 성능 지표 출력 (디버그 모드 시)
    if os.getenv("DEBUG_UI") == "true":
        st.sidebar.caption(
            f"📊 Full Reruns: {st.session_state.full_run_count} | Frag: {st.session_state.get('fragment_run_count', 0)}"
        )

    # 3. 실제 UI 렌더링 (모델이 준비된 상태)
    available_models = st.session_state.available_models_list
    _render_app_layout(available_models=available_models)

    # 4. 백그라운드 태스크 처리 (RAG 빌드, 모델 교체 등)
    _handle_pending_tasks()

    # 5. 첫 실행 플래그 해제
    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)
        logger.info("[SYSTEM] 시스템 초기화 완료")


if __name__ == "__main__":
    main()
