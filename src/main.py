"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

import os
from pathlib import Path
from typing import Any

import nest_asyncio
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

# 로깅 설정 (최상단)
from common.logging_config import setup_logging

logger = setup_logging(log_level="INFO", log_file=Path("logs/app.log"))

from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)  # noqa: E402
from common.constants import FilePathConstants, StringConstants  # noqa: E402

# [Lazy Import] 무거운 코어 모듈 임포트 제거 (함수 내부로 이동)
from core.session import SessionManager  # noqa: E402
from infra.notification_system import SystemNotifier  # noqa: E402
from ui.ui import (  # noqa: E402
    inject_custom_css,
    render_left_column,
    render_pdf_viewer,
    render_sidebar,
    update_window_height,
)

# 상수 정의
PAGE_TITLE = StringConstants.PAGE_TITLE
LAYOUT = StringConstants.LAYOUT
MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# 비동기 패치 적용 (최상단 실행)

nest_asyncio.apply()

# Streamlit 페이지 설정 (최우선 실행 - UI 즉시 표시용)
st.set_page_config(page_title=StringConstants.PAGE_TITLE, layout=StringConstants.LAYOUT)

# [보안] 세션 ID 강제 초기화 및 격리 보장
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    ctx = get_script_run_ctx()
    if ctx:
        SessionManager.init_session(session_id=ctx.session_id)
        logger.debug(f"[SYSTEM] [SESSION] 세션 초기화 완료 | ID: {ctx.session_id}")
except Exception as e:
    logger.warning(f"세션 초기화 실패: {e}")

# --- [추가] 필수 세션 상태 초기화 ---
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "last_valid_height" not in st.session_state:
    st.session_state.last_valid_height = 800
if "is_generating_answer" not in st.session_state:
    st.session_state.is_generating_answer = False

import atexit  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402


# [Extreme Lazy Import] 로깅 설정조차 필요한 시점으로 미룸
def get_logger():
    from common.logging_config import setup_logging

    return setup_logging(log_level="INFO", log_file=Path("logs/app.log"))


def _check_windows_integrity():
    """
    [Background] Windows 환경의 라이브러리 충돌을 체크하고 주기적으로 세션을 정리합니다.
    """
    import os
    import platform
    import time

    # [최적화] 세션 정리 추가 (메모리 누수 방지)
    try:
        from core.session import SessionManager

        SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
    except Exception:
        pass

    # [최적화] CI 환경에서는 무거운 라이브러리 체크 생략 (충돌 위험 방지)
    if platform.system() != "Windows" or os.getenv("GITHUB_ACTIONS") == "true":
        return

    try:
        # UI 렌더링을 위해 잠시 양보
        time.sleep(1.5)

        # 무거운 라이브러리 로드 테스트
        import torch
        import torchvision

        # 간단한 연산 테스트로 DLL 로드 확인
        _ = torch.tensor([1.0])
        logger.info("[SYSTEM] [INTEGRITY] Windows 라이브러리 무결성 점검 완료 (OK)")

    except ImportError as e:
        error_msg = str(e)
        if "0xc0000139" in error_msg or "DLL load failed" in error_msg:
            logger.critical(f"[SYSTEM] [INTEGRITY] 치명적 오류 감지: {error_msg}")
            # 사용자가 인지할 수 있도록 세션에 경고 기록
            from core.session import SessionManager

            SessionManager.add_message(
                "system",
                f"⚠️ 시스템 무결성 경고: Windows DLL 호환성 문제가 감지되었습니다. \n({error_msg})",
            )
    except Exception as e:
        logger.warning(f"[SYSTEM] [INTEGRITY] 점검 중 예외 발생: {e}")


def _run_background_checks():
    """백그라운드 점검 작업을 시작합니다."""
    # [최적화] 중복 실행 방지 (이미 시작되었거나 완료된 경우 스킵)
    if st.session_state.get("integrity_check_triggered"):
        return

    st.session_state.integrity_check_triggered = True
    threading.Thread(target=_check_windows_integrity, daemon=True).start()
    # 임시 디렉토리 정리도 여기서 호출하거나 기존처럼 유지
    # _init_temp_directory()는 이미 별도 스레드를 쓰고 있음


@st.cache_resource(show_spinner=False)
def _init_temp_directory():
    """임시 디렉토리를 초기화하고 잔해를 백그라운드에서 제거합니다."""
    from common.constants import FilePathConstants

    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    temp_path.mkdir(parents=True, exist_ok=True)

    def cleanup_task():
        try:
            # UI가 먼저 뜨도록 잠시 대기
            time.sleep(1)
            from infra.deployment_manager import get_deployment_manager

            manager = get_deployment_manager()
            # 실제 임시 디렉토리(temp_path)와 배포 디렉토리를 모두 정리
            # [수정] 임시 파일은 1시간만 지나도 정리 (테스트 반복 시 쌓임 방지)
            manager.cleanup_orphaned_artifacts(max_age_hours=1, target_dir=temp_path)
            manager.cleanup_orphaned_artifacts(max_age_hours=24)  # 기본 배포 폴더 정리
            logger.info(
                f"[SYSTEM] [JANITOR] 백그라운드 자원 정리 완료 | 대상: {temp_path} 및 deployments/"
            )
        except Exception as e:
            logger.error(f"[SYSTEM] [JANITOR] 리소스 정리 실패 | {e}")

    # 백그라운드 스레드 시작
    threading.Thread(target=cleanup_task, daemon=True).start()
    return str(temp_path)
    return str(temp_path)


# 앱 시작 시 초기화 수행 (캐싱으로 인해 최초 1회만 작동)
_init_temp_directory()
_run_background_checks()


def _cleanup_current_file():
    """현재 세션에서 사용 중인 임시 파일을 삭제합니다. (종료 핸들러용)"""
    # Streamlit 세션 상태를 직접 접근하기 어려우므로 SessionManager는 thread-safe하게 설계됨
    try:
        path = SessionManager.get("pdf_file_path")
        if path and os.path.exists(path):
            # [Windows] 파일 잠금 해제를 위한 재시도 로직
            for attempt in range(3):
                try:
                    os.remove(path)
                    print(f"[System] Cleanup: Deleted temp file {path}")
                    return
                except PermissionError:
                    if attempt < 2:  # 마지막 시도가 아니면 대기
                        time.sleep(0.5)
                except Exception:
                    pass
    except Exception:
        pass


# 프로세스 종료 시 핸들러 등록
atexit.register(_cleanup_current_file)


def _ensure_models_are_loaded() -> bool:
    """
    선택된 LLM 및 임베딩 모델을 중앙 관리자를 통해 안전하게 로드합니다.
    """
    from core.model_loader import ModelManager

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
        embedder = ModelManager.get_embedder(selected_embedding)
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
        llm = ModelManager.get_llm(selected_model)
        SessionManager.set("llm", llm)
        SystemNotifier.success("추론 모델 준비 완료")

        return True

    except Exception as e:
        logger.error(f"모델 로드 중 치명적 오류 발생: {e}", exc_info=True)
        st.error(f"❌ 모델 로드 실패: {e}")
        return False


def _rebuild_rag_system() -> None:
    """
    업로드된 파일과 선택된 모델을 사용하여 RAG 파이프라인을 재구축합니다.
    """
    file_name = SessionManager.get("last_uploaded_file_name")
    file_path = SessionManager.get("pdf_file_path")

    if not file_name or not file_path:
        return

    # [중복 실행 방지 강화]
    if st.session_state.get("is_building_rag"):
        return

    if (
        SessionManager.get("pdf_processed")
        and not SessionManager.get("pdf_processing_error")
        and SessionManager.get("vector_store") is not None
    ):
        return

    st.session_state.is_building_rag = True
    try:
        if not _ensure_models_are_loaded():
            return

        embedder = SessionManager.get("embedder")

        # [추가] 분석 시작 알림
        SystemNotifier.loading(f"'{file_name}' 분석 시작")

        # 실시간 상태 박스 업데이트를 위한 콜백 정의 (이제 내부 로그만 사용)
        def sync_ui():
            pass

        # [Lazy Import]
        from core.rag_core import build_rag_pipeline

        # RAG 파이프라인 빌드 (내부에서 상세 로그 기록 및 UI 동기화)
        success_message, cache_used = build_rag_pipeline(
            uploaded_file_name=file_name,
            file_path=file_path,
            embedder=embedder,
            on_progress=sync_ui,
        )

        SessionManager.add_message("system", success_message)
        SessionManager.add_message("system", "READY_FOR_QUERY")

    except Exception as e:
        logger.error(f"RAG 빌드 실패: {e}", exc_info=True)
        error_msg = f"문서 처리 중 오류가 발생했습니다: {str(e)}"
        SessionManager.set("pdf_processing_error", error_msg)
        SessionManager.set("pdf_processed", True)  # 분석 프로세스 종료(실패) 표시
        SessionManager.add_message("system", f"❌ {error_msg}")
    finally:
        st.session_state.is_building_rag = False


def _update_qa_chain() -> None:
    """
    문서 인덱싱은 유지한 채 LLM(QA Chain)만 교체합니다.
    """
    selected_model = SessionManager.get("last_selected_model")
    try:
        SessionManager.add_status_log("🔄 추론 모델 교체 중")

        # [Lazy Import]
        from core.model_loader import load_llm

        llm = load_llm(selected_model)
        SessionManager.set("llm", llm)
        SessionManager.replace_last_status_log("✅ 추론 모델 교체 완료")

        logger.info(f"LLM updated to: {selected_model}")
        msg = "✅ 추론 모델이 업데이트되었습니다."
        SessionManager.add_message("system", msg)

    except Exception as e:
        logger.error(f"QA 업데이트 실패: {e}", exc_info=True)
        SessionManager.add_message("assistant", f"❌ 업데이트 실패: {e}")


# --- Callbacks ---
def on_file_upload() -> None:
    """파일 업로드 이벤트 콜백"""
    uploaded_file = st.session_state.get("pdf_uploader")
    if not uploaded_file:
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
        if "pdf_page_index_input" in st.session_state:
            st.session_state.pdf_page_index_input = 1
        if "active_ref_id" in st.session_state:
            st.session_state.active_ref_id = None
        SessionManager.set("current_page", 1)

        # [관리강화] 이전 임시 파일 즉시 삭제
        old_path = SessionManager.get("pdf_file_path")
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
                logger.info(f"[System] [Cleanup] 이전 파일 삭제: {old_path}")
            except Exception as e:
                logger.warning(f"이전 파일 삭제 실패: {e}")

        SessionManager.set("last_uploaded_file_name", uploaded_file.name)

        # [전용 폴더 사용] 안정적인 임시 파일 생성
        try:
            # 절대 경로로 변환
            temp_dir = os.path.abspath(FilePathConstants.TEMP_DIR)
            os.makedirs(temp_dir, exist_ok=True)

            # 파일명에 타임스탬프를 넣어 중복 방지 (안전성)
            import time

            safe_name = f"upload_{int(time.time())}.pdf"
            tmp_path = os.path.join(temp_dir, safe_name)

            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            SessionManager.set("pdf_file_path", tmp_path)
            SessionManager.set("new_file_uploaded", True)
            SystemNotifier.success(f"문서 업로드 완료: {uploaded_file.name}", icon="📄")
            SystemNotifier.info("문서 내용 분석 및 인덱싱 시작")
            logger.info(f"[System] [Upload] 파일 저장 완료: {tmp_path}")
        except Exception as e:
            SystemNotifier.error("파일 저장 중 오류 발생", details=str(e))


def on_model_change() -> None:
    """LLM 모델 변경 이벤트 콜백"""
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


def _render_app_layout(
    is_skeleton_pass: bool, available_models: list[str] | None = None
) -> None:
    """앱의 전체 레이아웃을 렌더링하고 주요 플레이스홀더를 반환합니다."""
    # 0. 최우선 CSS 주입 (깜박임 방지)
    inject_custom_css()

    # 1. 사이드바 렌더링
    render_sidebar(
        file_uploader_callback=on_file_upload,
        model_selector_callback=on_model_change,
        embedding_selector_callback=on_embedding_change,
        is_generating=bool(SessionManager.get("is_generating_answer", False)),
        current_file_name=SessionManager.get("last_uploaded_file_name"),
        current_embedding_model=SessionManager.get("last_selected_embedding_model"),
        available_models=available_models,
    )

    # 2. 메인 영역 레이아웃
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader(
            StringConstants.MSG_CHAT_TITLE
            if hasattr(StringConstants, "MSG_CHAT_TITLE")
            else "💬 채팅"
        )
        render_left_column()

    with col_right:
        st.subheader(
            StringConstants.MSG_PDF_VIEWER_TITLE
            if hasattr(StringConstants, "MSG_PDF_VIEWER_TITLE")
            else "📄 PDF 미리보기"
        )
        render_pdf_viewer()


def _handle_pending_tasks() -> None:
    """지연된 무거운 작업(RAG 빌드, 모델 교체 등)을 순차적으로 처리합니다."""
    if SessionManager.get("new_file_uploaded"):
        current_file_path = SessionManager.get("pdf_file_path")
        current_file_name = SessionManager.get("last_uploaded_file_name")
        SessionManager.reset_for_new_file()
        SessionManager.set("pdf_file_path", current_file_path)
        SessionManager.set("last_uploaded_file_name", current_file_name)
        SessionManager.set("new_file_uploaded", False)
        _rebuild_rag_system()
        st.rerun()

    elif SessionManager.get("needs_rag_rebuild"):
        SessionManager.set("needs_rag_rebuild", False)
        _rebuild_rag_system()
        st.rerun()

    elif SessionManager.get("needs_qa_chain_update"):
        SessionManager.set("needs_qa_chain_update", False)
        _update_qa_chain()
        st.rerun()


def main() -> None:
    """메인 애플리케이션 오케스트레이터"""
    # 1. 초기 레이아웃 및 세션 즉시 준비
    inject_custom_css()
    SessionManager.init_session()

    # [추가] 세션 ID 불일치로 인한 '영구 분석 중' 상태 방지
    if SessionManager.get("pdf_file_path") and not SessionManager.get("pdf_processed"):
        # 분석이 중단된 것으로 간주하고 입력창 열기
        SessionManager.set("is_generating_answer", False)

    # 2. UI 즉시 렌더링 (Optimistic UI)
    # 모델 목록 로딩 상태 확인
    if "available_models_list" not in st.session_state:
        st.session_state.available_models_list = None  # 아직 시도 안 함

    available_models = st.session_state.available_models_list

    _render_app_layout(
        is_skeleton_pass=(available_models is None), available_models=available_models
    )

    # 3. 데이터 로딩 (UI가 그려진 후 딱 한 번만 실행되도록 유도)
    if st.session_state.available_models_list is None:
        from core.model_loader import get_available_models

        # 실제 Ollama 모델 목록 가져오기
        fetched_models = get_available_models()

        # 만약 에러 메시지가 포함되어 있거나 비어있다면 최소한 기본 모델은 포함시킴
        from common.config import DEFAULT_OLLAMA_MODEL

        if not fetched_models or (
            len(fetched_models) == 1 and "서버" in fetched_models[0]
        ):
            st.session_state.available_models_list = [DEFAULT_OLLAMA_MODEL]
        else:
            st.session_state.available_models_list = fetched_models

        st.rerun()

    # 4. 백그라운드 태스크 처리 (RAG 빌드, 모델 교체 등)
    _handle_pending_tasks()

    # 5. 첫 실행 플래그 해제 (예열 과정 삭제 - 지연 로딩 정책 채택)
    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)
        logger.info("[SYSTEM] 시스템 초기화 완료")

    # 6. 창 높이 측정 (가장 마지막에 실행하여 레이아웃 영향 최소화)
    update_window_height()


if __name__ == "__main__":
    main()
