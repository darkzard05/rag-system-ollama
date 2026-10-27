# src/main.py
# Streamlit 기반 RAG 챗봇의 메인 진입점 및 전체 오케스트레이션을 담당하는 파일
"""
RAG Chatbot 애플리케이션의 메인 진입점 파일입니다.
Streamlit 프레임워크를 기반으로 UI를 구성하고 세션 상태를 관리합니다.
"""

# P4 (조사: .omo/evidence/log-issue-fixes/p4_warning_source.md): langgraph 1.0.1의
# LangChainPendingDeprecationWarning("allowed_objects 기본값 변경 예정")은 모듈 임포트
# 시점(langgraph/checkpoint/serde/jsonplus.py 모듈 최상위 LC_REVIVER = Reviver())에
# 발생하며 앱의 JsonPlusSerializer 파라미터로는 제거할 수 없다. 경고 필터는
# list-index-0 우선(last-registered wins)이므로, langchain_core._api.deprecation이
# 임포트 시 자체 등록하는 필터보다 우리의 ignore가 나중에 등록(먼저 검사)되어야 한다.
# 따라서 (1) 해당 모듈을 먼저 임포트한 뒤 (2) 카테고리+메시지 범위로만 억제를
# 등록한다 — 다른 경고는 모두 그대로 유지된다.
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The default value of .allowed_objects. will change",
    category=LangChainPendingDeprecationWarning,
)

# [Path Bootstrap] 저장소 루트의 src/ 를 sys.path 에 추가하여
# `from common`/`from core` 식의 bare import 가 해석되도록 한다.
# (프로젝트 스크립트들이 사용하는 관례: scripts/test_app.py 등)
# 이 부트스트랩이 없으면 streamlit run src/main.py 가 저장소 루트에서
# 실행될 때 `No module named 'src.common'` 으로 실패한다.
import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# [패치] transformers 5.x auto_docstring stdout noise 원천 차단
# (importlib 기반 — 해당 속성은 서브모듈이 아닌 데코레이터 함수로 노출된다)
from common.logging_config import silence_transformers_auto_docstring

silence_transformers_auto_docstring()

# [Lazy Import용] 런타임에 필요한 모듈들
import asyncio
import atexit
import contextlib
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Literal, cast

import streamlit as st

from common.async_worker import AsyncWorker
from common.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
    MSG_ERROR_OLLAMA_NOT_RUNNING,
)
from common.constants import FilePathConstants, StringConstants
from common.logging_config import setup_logging
from common.utils import compute_file_hash, safe_cache_resource
from ui.strings import t

# 1. Streamlit 페이지 설정 (최우선 실행 - 가이드라인 준수)
st.set_page_config(
    page_title=StringConstants.PAGE_TITLE,
    layout=cast(Literal["centered", "wide"], StringConstants.LAYOUT),
    initial_sidebar_state="expanded",
)


# 2. 로깅 설정 (최상단, 서버당 1회만 초기화)
# LOG_LEVEL 환경변수로 DEBUG 전환 가능 (기본 INFO). JSON 파싱 실패 등 원시
# 출력 추적이 필요할 때 LOG_LEVEL=DEBUG 로 서버를 기동한다.
@st.cache_resource
def _init_logging() -> logging.Logger:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    return setup_logging(log_level=log_level, log_file=FilePathConstants.LOG_FILE)


logger = _init_logging()

MAX_FILE_SIZE_MB = StringConstants.MAX_FILE_SIZE_MB

# NumExpr 최대 스레드 설정 (CPU 코어 수 기반)
_numexpr_threads = os.cpu_count() or 4
os.environ.setdefault("NUMEXPR_MAX_THREADS", str(_numexpr_threads))

# 비동기 워커 초기화 (서버 인스턴스당 단일 이벤트 루프, nest_asyncio 대체)
_async_worker = AsyncWorker()

# 부팅 시각: VRAM 언로드 유예 판단에 사용한다 (프리웜과의 쓰래싱 방지).
_APP_START_TIME = time.monotonic()
_BOOT_VRAM_UNLOAD_GRACE_SECONDS = 120.0

# current_page / is_generating_answer are initialized via DEFAULT_SESSION_STATE
# and synced by UIBridge.sync_session(). pdf_target_page는 일회성 점프 토큰(dict
# 또는 None)으로 DEFAULT_SESSION_STATE에 포함되지 않으므로 여기서 초기화한다.
if "pdf_target_page" not in st.session_state:
    st.session_state.pdf_target_page = None


def _check_windows_integrity(uptime_seconds: float | None = None):
    """
    [Background] Windows 환경의 라이브러리 충돌을 체크하고 주기적으로 세션을 정리합니다.

    ``uptime_seconds`` 기동 후 경과 시간을 주입해 부팅 유예 기간을 검증할 수 있다.
    """
    current_uptime = (
        uptime_seconds
        if uptime_seconds is not None
        else (time.monotonic() - _APP_START_TIME)
    )
    try:
        from core.session import SessionManager

        # 1시간 이상 활동 없는 세션 정리 (물리적 파일 삭제 및 참조 해제 포함)
        SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
        # 세션이 하나도 없으면 Ollama 상주 모델을 명시 언로드한다.
        # 시간당 sweep에서만 발화하므로 활성 사용자에게는 영향이 없다.
        # (기본 keep_alive=5m가 1차 방어, 여기는 확정 정리.)
        # [부팅 유예] 기동 직후에는 백그라운드 프리웜이 모델을 VRAM에 올리는
        # 중이므로, 언로드를 보내면 로드/언로드 쓰래싱이 발생한다.
        try:
            if (
                current_uptime >= _BOOT_VRAM_UNLOAD_GRACE_SECONDS
                and SessionManager.get_stats().get("active_sessions", 0) == 0
            ):
                from common.async_worker import AsyncWorker
                from core.model_loader import ModelManager

                _unload_fut = AsyncWorker().submit(ModelManager.clear_vram())

                def _log_unload_done(fut):
                    try:
                        fut.result()
                    except Exception as ue:
                        logger.warning(
                            f"[SYSTEM] [CLEANUP] 유휴 VRAM 언로드 실패: {ue}"
                        )
                    else:
                        logger.info("[SYSTEM] [CLEANUP] 유휴 VRAM 언로드 완료")

                _unload_fut.add_done_callback(_log_unload_done)
        except Exception as ue:
            logger.warning(f"[SYSTEM] [CLEANUP] 유휴 VRAM 언로드 스킵: {ue}")
    except Exception as e:
        logger.error(f"[SYSTEM] [CLEANUP] 세션 정리 중 오류: {e}")

    import platform

    if platform.system() != "Windows" or os.getenv("GITHUB_ACTIONS") == "true":
        return

    try:
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            logger.warning("[SYSTEM] [INTEGRITY] torch 모듈을 찾을 수 없습니다.")
            return

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
        # [부팅 유예] 첫 sweep 전에 프리웜이 끝날 시간을 준다. Worker 시작
        # 시각을 고정해 Streamlit rerun으로 모듈 전역이 재할당돼도 유예가
        # 리셋되지 않게 한다.
        started_at = time.monotonic()
        time.sleep(60)
        while True:
            with contextlib.suppress(Exception):
                _check_windows_integrity(uptime_seconds=time.monotonic() - started_at)
            # 1시간(3600초) 대기 후 반복
            time.sleep(3600)

    thread = threading.Thread(
        target=maintenance_loop, name="GlobalMaintenanceWorker", daemon=True
    )
    thread.start()
    return thread


def _get_available_models_cached(non_blocking_fallback: bool = False):
    """Ollama 모델 목록을 프로세스 전역 TTL 캐시로 조회합니다.

    캐시의 단일 진원은 ``core._loaders._fetch_available_models_cached`` 의
    프로세스 전역 TTL 캐시(성공 300s / 실패 15s)다. Streamlit 세션과 무관하게
    모든 세션이 공유하므로 새 탭/새로고침마다 ``list()`` HTTP 호출(최대 5s)이
    다시 발생하지 않는다. 실패는 짧게만 캐시해 Ollama가 늦게 기동해도 UI가
    오류 상태로 오래 고정되지 않는다.

    ``non_blocking_fallback=True`` 는 UI 메인 스레드용으로, 캐시 미스 시
    동기 HTTP 호출 없이 백그라운드 갱신만 예약하고 기본 모델을 즉시 반환한다.

    Caveat: Ollama CLI에서 신규 모델을 pull한 경우 사이드바의 "모델 새로고침"
    버튼(``on_refresh_models`` → ``clear_available_models_cache``)으로
    명시적으로 갱신해야 한다.
    """
    from core.model_loader import get_available_models

    return get_available_models(non_blocking_fallback=non_blocking_fallback)


def _load_available_models(non_blocking_fallback: bool = True) -> list[str]:
    """Ollama 모델 목록을 동적으로 조회하여 반환합니다.

    기본값은 UI 메인 스레드를 블로킹하지 않는 fast path다. 명시적 새로고침처럼
    최신 값이 즉시 필요한 경로만 ``non_blocking_fallback=False`` 로 호출한다.
    get_available_models는 Ollama 미연결 시 기본 모델과 오류 문자열을 함께
    반환할 수 있으므로, UI에 노출되면 안 되는 오류 문자열은 걸러냅니다.
    조회가 실패(타임아웃/비정상 종료 등)하면 안전하게 기본 모델로 폴백합니다.
    """
    try:
        models = _get_available_models_cached(
            non_blocking_fallback=non_blocking_fallback
        )
    except Exception:
        logger.exception("[SYSTEM] Ollama 모델 목록 조회 실패, 기본 모델로 대체")
        return [DEFAULT_OLLAMA_MODEL]

    filtered = [m for m in models if m != MSG_ERROR_OLLAMA_NOT_RUNNING]
    return filtered or [DEFAULT_OLLAMA_MODEL]


def _is_ollama_reachable() -> bool:
    """Return True if the cached model list indicates Ollama is reachable.

    Reads the cached result through the non-blocking path (no extra blocking
    call) and checks for the sentinel error string or an empty list.
    """
    try:
        models = _get_available_models_cached(non_blocking_fallback=True)
    except Exception:
        return False
    return bool(models) and MSG_ERROR_OLLAMA_NOT_RUNNING not in models


def _warm_available_models() -> None:
    """RC-C: 프로세스 부트 시 Ollama 모델 목록을 비동기 프리워밍합니다.

    ``_get_available_models_cached`` 는 프로세스 전역 TTL 캐시(성공 300s)이므로
    부팅 시 데몬 스레드에서 최대 1회 조회해 캐시를 데우고, 첫 사용자 렌더가
    ~5s 블로킹(list 호출)되는 것을 막는다. Ollama 미연결 시 예외는 무시 —
    첫 렌더의 기존 폴백/스피너가 보완.
    """
    try:
        _get_available_models_cached(non_blocking_fallback=True)
    except Exception:  # noqa: BLE001 - best-effort warmup; first render handles failure
        logger.debug("[SYSTEM] model-list warmup skipped (Ollama not ready)")


def _warm_heavy_imports() -> None:
    """ScriptRunContext 없이 안전하게 선행 로드할 수 있는 무거운 모듈만 예열한다.

    ``streamlit_pdf_viewer`` 는 모듈 임포트 시점에 ``declare_component`` 를
    호출한다. Streamlit 은 ScriptRunContext 가 없는 스레드에서는 컴포넌트
    등록을 건너뛰므로, 백그라운드 스레드에서 먼저 임포트하면 모듈만
    ``sys.modules`` 에 캐시되고 컴포넌트는 미등록 상태로 남는다. 이후 script
    스레드의 lazy import 는 캐시를 재사용해 등록을 다시 시도하지 않으므로
    뷰어 iframe 이 404 가 된다. 따라서 이 함수에서 임포트하면 안 된다.
    """
    try:
        import pymupdf  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - best-effort warmup
        logger.debug("[WARMUP] pymupdf import skipped: %s", exc)


# 프로세스 시작 직후(import 직후) 백그라운드 웜업 — UI 첫 렌더 블로킹 제거.
# st.cache_resource 가드로 rerun마다 스레드가 재생성되는 것을 막는다.
@st.cache_resource(show_spinner=False)
def _start_model_list_warmup() -> bool:
    threading.Thread(
        target=_warm_available_models, name="model-list-warmup", daemon=True
    ).start()
    threading.Thread(
        target=_warm_heavy_imports, name="heavy-import-warmup", daemon=True
    ).start()
    return True


@safe_cache_resource(show_spinner=False)
def _init_temp_directory():
    """임시 디렉토리를 초기화합니다."""
    temp_path = Path(FilePathConstants.TEMP_DIR).absolute()
    temp_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"[SYSTEM] [INIT] 임시 디렉토리 준비 완료: {temp_path}")
    return str(temp_path)


def _cleanup_current_file():
    """모든 세션에서 사용 중인 임시 파일을 삭제합니다. (종료 핸들러용)

    "default" 세션만 정리하면 다른 세션의 임시 PDF가 남으므로,
    전체 세션의 경로를 수집하여 삭제합니다.
    """
    from core.session import SessionManager

    try:
        paths = SessionManager.get_all_pdf_paths()
        for path in paths:
            SessionManager.safe_remove_file(path)
    except Exception:
        pass


def _cleanup_coord_cache() -> None:
    """프로세스 종료 시 좌표 캐시를 정리합니다 (지연 임포트로 부팅 지연 방지)."""
    with contextlib.suppress(Exception):
        from cache.coord_cache import coord_cache

        asyncio.run(coord_cache.close())


@st.cache_resource
def _register_cleanup_handlers():
    """[Singleton] 프로세스 종료 핸들러를 단 한 번만 등록합니다."""
    atexit.register(_cleanup_current_file)
    atexit.register(_cleanup_coord_cache)
    logger.info("[SYSTEM] 프로세스 종료 핸들러 등록 완료")
    return True


# 앱 시작 시 초기화 수행 (캐싱으로 인해 최초 1회만 작동)
_init_temp_directory()
_start_global_background_worker()
_register_cleanup_handlers()

# F1: attach the Streamlit UI-sync adapter so SessionManager can mirror state
# into st.session_state without core importing streamlit.
from core.session import SessionManager
from ui.session_sync import StreamlitSessionSync

SessionManager.set_ui_sync(StreamlitSessionSync())


async def _warmup_models() -> None:
    """사용자 상호작용 발생 시 즉각 양보하는 안전한 백그라운드 프리워밍."""
    from core.session import SessionManager

    # 2.5초 대기를 100ms 단위 폴링으로 분할하여 사용자 상호작용 즉시 감지
    for _ in range(25):
        await asyncio.sleep(0.1)
        if (
            SessionManager.get("new_file_uploaded")
            or SessionManager.get("is_building_rag")
            or SessionManager.get("pdf_file_path")
            or SessionManager.get("is_generating_answer")
        ):
            logger.info("[WARMUP] 사용자 상호작용 감지 — 프리워밍 즉시 중단")
            return

    from core.model_loader import _warmup_models as _core_warmup

    try:
        await _core_warmup()
    except Exception as e:
        logger.debug(f"[WARMUP] warmup interrupted or failed: {e}")


# [WARMUP] 프로세스(서버) 단위 1회 프리웜.
# Streamlit은 스크립트 rerun 시 모듈 최상위 코드를 새 모듈 네임스페이스에서
# 재실행하고, 새 탭/새로고침은 별도 세션(st.session_state)을 만든다. 따라서
# 모듈 전역 변수나 세션 플래그는 프로세스 단위 보장이 안 된다(세션마다 프리웜이
# 재제출되어 MAX_CONCURRENT_INFERENCE=1 락을 점유, 첫 질의가 대기). 프로세스
# 전역인 st.cache_resource 싱글턴을 1회 제출 가드로 사용한다.
@st.cache_resource(show_spinner=False)
def _trigger_global_warmup_once() -> bool:
    """서버 라이프사이클당 정확히 1회만 프리웜을 제출합니다."""
    coro = _warmup_models()
    try:
        _async_worker.submit(coro)
    except Exception as e:
        coro.close()  # 미제출 코루틴의 "never awaited" 경고 방지
        logger.warning(f"[WARMUP] 모델 프리웜 실패 — 첫 쿼리에서 로드됨: {e}")
    return True


def _active_build_state(session_id: str | None = None):
    """Return the non-terminal BuildState when a build is genuinely active.

    ``is_building_rag`` alone can outlive a failed dispatch or a stale test
    fixture. Requiring a non-terminal state prevents a permanently stuck flag
    from blocking future uploads.
    """
    from core.build_state import is_terminal
    from core.session import SessionManager

    if not SessionManager.get("is_building_rag", False, session_id):
        return None
    build = SessionManager.get_build_state(session_id=session_id)
    if build is None or is_terminal(build.phase):
        return None
    return build


def _persist_upload_atomically(target: str, content: bytes) -> None:
    """Write ``content`` to ``target`` via unique temp file + os.replace.

    In-place ``open(..., "wb")`` truncates the destination, so a concurrent
    reader can observe a torn PDF. Windows also rejects replaces while a
    reader holds the file, hence the bounded retry loop.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target_path.name}.", suffix=".tmp", dir=str(target_path.parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        last_error: OSError | None = None
        for _ in range(200):
            try:
                os.replace(tmp_path, target_path)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                time.sleep(0.005)
        if last_error is not None:
            raise last_error
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def on_file_upload() -> None:
    from core.session import SessionManager

    uploaded_file = st.session_state.get("pdf_uploader")
    if uploaded_file is None or not hasattr(uploaded_file, "type"):
        return

    # A build is already running. The sidebar uploader is disabled in that
    # state, but programmatic/race uploads still reach this callback; reject
    # them before reset_for_new_file can clobber the in-flight build state.
    if _active_build_state() is not None and not SessionManager.get(
        "new_file_uploaded", False
    ):
        _post_upload_error(
            "A document is already being analyzed. Please wait for it to finish."
        )
        return

    if uploaded_file.type != "application/pdf":
        _post_upload_error("Invalid PDF file. Please upload a file in PDF format.")
        return

    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        _post_upload_error(
            f"File is too large ({file_size_mb:.2f} MB). Please upload a file no larger than {MAX_FILE_SIZE_MB} MB."
        )
        return

    # [멱등 가드] Streamlit rerun/더블 트리거로 on_change 콜백이 동일 파일에
    # 대해 중복 발동되는 것을 차단한다. 해시 계산 전에 세션에 처리 중 플래그를
    # 선점 세트하여, 재진입 시점에 이미 처리 중이면 즉시 종료한다.
    upload_guard_key = "file_upload_in_progress"
    if SessionManager.get(upload_guard_key, False):
        return
    SessionManager.set(upload_guard_key, True)
    try:
        _process_uploaded_file(uploaded_file)
    finally:
        SessionManager.set(upload_guard_key, False)


def _find_cached_upload(
    temp_dir: str, file_bytes: bytes, uploaded_hash: str
) -> str | None:
    """해시 기반 결정론적 파일명(Content-Addressed)으로 O(1) 파일 존재 여부 확인."""
    if not uploaded_hash:
        return None
    # 1. Content-Addressed 타깃 파일 O(1) 우선 검사
    target_file = Path(temp_dir) / f"upload_{uploaded_hash}.pdf"
    try:
        if target_file.is_file() and target_file.stat().st_size == len(file_bytes):
            return str(target_file)
    except OSError:
        pass

    # 2. 레거시 파일 스캔은 크기가 완전히 일치하는 파일만 최대 3건으로 엄격히 제한
    try:
        expected_size = len(file_bytes)
        candidates = [
            p
            for p in Path(temp_dir).glob("upload_*.pdf")
            if p.name != target_file.name
            and p.is_file()
            and p.stat().st_size == expected_size
        ]
    except OSError as e:
        logger.warning(f"[UPLOAD] 재사용 스캔 실패: {e}")
        return None

    for entry in candidates[:3]:
        try:
            if compute_file_hash(str(entry)) == uploaded_hash:
                return str(entry)
        except OSError:
            continue
    return None


def _process_uploaded_file(uploaded_file) -> None:
    """실제 업로드 처리 로직. on_file_upload 가 멱등 가드 후 호출한다."""
    from core.session import SessionManager
    from services.monitoring.notification_system import SystemNotifier
    from ui.components.common import navigate_to_page

    sid = SessionManager.get_session_id()
    last_file_name = SessionManager.get("last_uploaded_file_name", session_id=sid)
    last_file_hash = SessionManager.get("file_hash", session_id=sid)
    # reset_for_new_file clears pdf_file_path; capture the old path first so
    # cleanup can still run after the new file is committed.
    old_path = SessionManager.get("pdf_file_path", session_id=sid)

    file_bytes = b""
    try:
        # uploaded_file는 BytesIO와 유사한 객체이므로 포인터를 리셋하여 전체 바이트를 계산합니다.
        uploaded_file.seek(0)
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        uploaded_hash = compute_file_hash("", data=file_bytes)
    except Exception:
        uploaded_hash = ""

    # [확장자 위조 방지 및 1페이지 선행 추출]
    # 업로드 검증 시점에 이미 열려 있는 문서 객체로부터 1페이지 바이트를 즉시
    # 추출해 두면, 뷰어 첫 프레임이 디스크 재오픈/재파싱 없이 렌더링된다.
    first_page_bytes = b""
    try:
        import pymupdf as fitz  # lazy import: 검증이 필요한 시점에만 로드

        if not file_bytes:
            raise ValueError("PDF file is empty.")
        with fitz.open(stream=file_bytes, filetype="pdf") as upload_doc:
            page_count = len(upload_doc)
            if page_count < 1:
                raise ValueError("PDF file has no pages.")
            first_doc = fitz.open()
            try:
                first_doc.insert_pdf(upload_doc, from_page=0, to_page=0)
                # garbage=0 및 deflate=False로 불필요한 객체 테이블 스캔과 압축 CPU 소모를 모두 제거하여 즉각 생성
                first_page_bytes = first_doc.tobytes(garbage=0, deflate=False)
            finally:
                first_doc.close()
    except Exception as e:
        logger.warning(f"손상되었거나 읽을 수 없는 PDF 업로드 차단: {e}")
        _post_upload_error(
            "The PDF file is corrupted or unreadable. Please upload a valid PDF file."
        )
        return

    if uploaded_file.name != last_file_name or uploaded_hash != last_file_hash:
        temp_dir = os.path.abspath(FilePathConstants.TEMP_DIR)
        cached_path = _find_cached_upload(temp_dir, file_bytes, uploaded_hash)
        # [UX] 새 파일 전환: 대화/플래그를 렌더 전(콜백 시점)에 초기화하여
        # 업로드 직후 불필요한 전체 rerun 없이도 빈 화면 깜빡임이 사라진다.
        # (reset_for_new_file이 current_page/pdf_annotations/pdf_target_page 초기화 포함)
        SessionManager.reset_for_new_file(session_id=sid)

        # Persist before publishing viewer/build metadata. If this fails, the
        # session stays in a clean "no document" state instead of a permanent
        # PREPARING panel pointing at a half-written file.
        try:
            if cached_path is not None:
                tmp_path = cached_path
                logger.info(f"[UPLOAD] 동일 파일 재사용 (dedup): {tmp_path}")
            else:
                tmp_path = os.path.join(temp_dir, f"upload_{uploaded_hash}.pdf")
                _persist_upload_atomically(tmp_path, file_bytes)
        except OSError as e:
            logger.error("[UPLOAD] 파일 저장 실패: %s", e, exc_info=True)
            _post_upload_error(
                "Error while saving the uploaded file. Please try again."
            )
            return

        SessionManager.set("total_pages", page_count, session_id=sid)
        # [FAST-PATH 동기화] file_hash와 last_uploaded_file_name을 미리보기 등록과 동시에
        # 즉시 설정하여, 뷰어가 first_page_preview의 해시 일치 검사를 즉시 통과할 수 있도록 보장한다.
        SessionManager.set("file_hash", uploaded_hash, session_id=sid)
        SessionManager.set(
            "last_uploaded_file_name", uploaded_file.name, session_id=sid
        )
        if first_page_bytes:
            SessionManager.set(
                "first_page_preview",
                {"file_hash": uploaded_hash, "bytes": first_page_bytes},
                session_id=sid,
            )

        # === [추가 시작] Frame 1에서 우측 분석 패널이 1페이지 뷰어와 동시 출력되도록 즉시 초기화 ===
        SessionManager.init_build_state(
            file_name=uploaded_file.name,
            file_hash=uploaded_hash,
            session_id=sid,
        )
        SessionManager.set("is_building_rag", True, session_id=sid)
        # === [추가 끝] ===

        # 새 문서는 항상 1페이지부터: 단일 navigate_to_page(1) 경로로
        # SessionManager.current_page를 리셋한다. 위젯 키(INTERACTIVE_KEYS)는
        # 직접 쓰지 않으며, 네비 입력 위젯은 렌더 시 매니저에서 default를 받는다.
        # reset_for_new_file이 pdf_target_page 토큰(저장소)을 이미 삭제했으므로
        # UI 미러만 정리한다.
        navigate_to_page(1)
        st.session_state.pop("pdf_target_page", None)

        SessionManager.set("pdf_file_path", tmp_path, session_id=sid)
        SessionManager.set("new_file_uploaded", True, session_id=sid)
        SystemNotifier.success(f"Document uploaded: {uploaded_file.name}")

        if old_path and old_path != tmp_path:
            # reset_for_new_file already removed this session's reference, so
            # any remaining reference belongs to another session.
            still_referenced = SessionManager.get_all_pdf_paths().count(old_path) > 0
            if not still_referenced and not SessionManager.safe_remove_file(old_path):
                logger.warning(f"이전 업로드 파일 삭제 실패 (잠금 가능): {old_path}")
                SystemNotifier.warning(f"이전 파일 정리에 실패했습니다: {old_path}")
    else:
        SystemNotifier.success(f"'{uploaded_file.name}' was already uploaded.")


def on_new_chat() -> None:
    """새 대화 시작: 문서/파이프라인은 유지하고 대화만 초기화합니다."""
    from core.session import SessionManager

    current_sid = SessionManager.get_session_id()
    SessionManager.reset_conversation(current_sid)
    # 버튼 위젯 interaction 자체가 리런을 유발하므로 명시 rerun 금지 (중복 리런 방지).


def on_model_change() -> None:
    from core.session import SessionManager

    selected = st.session_state.get("model_selector")
    last = SessionManager.get("last_selected_model")

    if not selected or "---" in selected or selected == last:
        return

    SessionManager.set("last_selected_model", selected)
    if SessionManager.get("pdf_processed"):
        SessionManager.set("needs_qa_chain_update", True)


def on_embedding_change() -> None:
    from core.session import SessionManager

    selected = st.session_state.get("embedding_model_selector")
    last = SessionManager.get("last_selected_embedding_model")

    if not selected or selected == last:
        return

    SessionManager.set("last_selected_embedding_model", selected)
    if SessionManager.get("pdf_file_path"):
        SessionManager.set("needs_rag_rebuild", True)


def on_refresh_models() -> None:
    """Ollama 모델 목록을 재조회하여 사이드바 셀렉터를 갱신합니다."""
    from core.model_loader import clear_available_models_cache

    # 코어 프로세스 전역 TTL 캐시를 무효화해야 신규 pull 모델이 보인다.
    clear_available_models_cache()
    # 명시적 새로고침은 최신 값이 즉시 필요하므로 동기 조회를 허용한다.
    st.session_state.available_models_list = _load_available_models(
        non_blocking_fallback=False
    )
    # 버튼 위젯 interaction 자체가 리런을 유발하므로 명시 rerun 금지 (중복 리런 방지).


def _render_app_layout(
    available_models: list[str] | None = None, ollama_reachable: bool = True
) -> None:
    from core.session import SessionManager
    from ui.components.sidebar import render_settings_content

    with st.sidebar:
        render_settings_content(
            file_uploader_callback=on_file_upload,
            model_selector_callback=on_model_change,
            embedding_selector_callback=on_embedding_change,
            new_chat_callback=on_new_chat,
            refresh_models_callback=on_refresh_models,
            is_generating=bool(SessionManager.get("is_generating_answer", False)),
            is_swapping_model=bool(SessionManager.get("is_swapping_model", False)),
            current_file_name=SessionManager.get("last_uploaded_file_name"),
            available_models=available_models,
            ollama_reachable=ollama_reachable,
        )

    from ui.ui import render_main_content

    render_main_content()


def _handle_pending_tasks() -> None:
    from core.session import SessionManager

    current_sid = SessionManager.get_session_id()
    is_building = bool(SessionManager.get("is_building_rag", False, current_sid))
    is_generating = bool(SessionManager.get("is_generating_answer", False, current_sid))
    needs_rerun = False

    if not is_building and not is_generating:
        _BUILD_RERUN_LAST.pop(current_sid, None)
        has_pending = (
            SessionManager.get("new_file_uploaded", False, current_sid)
            or SessionManager.get("needs_rag_rebuild", False, current_sid)
            or SessionManager.get("needs_qa_chain_update", False, current_sid)
        )
        if not has_pending:
            return

    def _safe_dispatch(kind: str, coro: Any) -> None:
        """백그라운드 디스패치를 감싸 실패 시 플래그가 영구 잔류되지 않게 한다.

        ``run_in_background_worker`` 가 submit 에서 실패하면 ``_on_complete``
        가 영원히 호출되지 않아 ``is_building_rag``/``needs_*`` 플래그가
        굳어 입력창이 비활성화된다(INT-입력동결). 디스패치 직후 죽으면
        여기서 플래그를 롤백한다.
        """
        from common.utils import run_in_background_worker

        try:
            run_in_background_worker(coro, current_sid)
        except Exception as exc:  # noqa: BLE001 - 디스패치 실패는 복구해야 함
            logger.error("[MAIN] 백그라운드 디스패치 실패 (%s): %s", kind, exc)
            if kind == "rebuild":
                SessionManager.set("needs_rag_rebuild", True, current_sid)
                from core.build_state import fail_build, is_terminal

                failed = SessionManager.get_build_state(session_id=current_sid)
                if failed is not None and not is_terminal(failed.phase):
                    message = f"Background task dispatch failed: {exc}"
                    SessionManager.finalize_build_state(
                        fail_build(failed, message), session_id=current_sid
                    )
            elif kind == "qa_update":
                SessionManager.set("needs_qa_chain_update", True, current_sid)
            SessionManager.set("is_building_rag", False, current_sid)
            SessionManager.set(
                "pdf_processing_error",
                f"Background task dispatch failed: {exc}",
                current_sid,
            )
            _BUILD_RERUN_LAST.pop(current_sid, None)

    if SessionManager.get(
        "new_file_uploaded", False, current_sid
    ) and not SessionManager.get("is_generating_answer", False, current_sid):
        SessionManager.set("new_file_uploaded", False, current_sid)

        current_file_path = SessionManager.get("pdf_file_path", None, current_sid)
        current_file_name = SessionManager.get(
            "last_uploaded_file_name", None, current_sid
        )
        current_file_hash = str(SessionManager.get("file_hash", "", current_sid) or "")
        current_embedding_model = (
            SessionManager.get("last_selected_embedding_model", None, current_sid)
            or DEFAULT_EMBEDDING_MODEL
        )

        # 이미 업로드 핸들러에서 생성되었는지 확인 후 없을 때만 폴백 생성
        if SessionManager.get_build_state(session_id=current_sid) is None:
            SessionManager.init_build_state(
                str(current_file_name or "document.pdf"),
                current_file_hash,
                session_id=current_sid,
            )
        SessionManager.set("is_building_rag", True, current_sid)
        # [UX-FIRST-PAGE] 백그라운드 태스크의 초기 preparing -> extracting 위상 전이가
        # 첫 페이지 페인트 보호 윈도우(0.8s) 동안 즉각 리런을 유발하지 않도록 기준 시각을 등록한다.
        _BUILD_RERUN_LAST[current_sid] = (time.monotonic(), "preparing")

        _safe_dispatch(
            "rebuild",
            _bg_rebuild_task(
                current_sid,
                current_file_path,
                current_file_name,
                current_embedding_model,
                expected_file_hash=current_file_hash,
            ),
        )
        # [UX-FIRST-PAGE] 신규 파일 업로드 직후 st.rerun()을 호출하지 않는다.
        # 첫 렌더링 프레임에서 브라우저가 좌측 PDF 1페이지 뷰어를 즉시 화면에
        # 온전히 페인트하도록 보장한다. 분석 패널은 백그라운드 워커의 첫 위상
        # 이벤트가 자연스러운 Rerun을 트리거할 때 렌더링된다.
        needs_rerun = False

    elif (
        SessionManager.get("needs_rag_rebuild", False, current_sid)
        and not is_building
        and not SessionManager.get("is_generating_answer", False, current_sid)
    ):
        SessionManager.set("needs_rag_rebuild", False, current_sid)
        SessionManager.set("is_building_rag", True, current_sid)
        SessionManager.set("active_faiss_retriever", None, current_sid)
        SessionManager.set("active_bm25_retriever", None, current_sid)
        SessionManager.delete("active_retriever_hash", current_sid)

        current_file_path = SessionManager.get("pdf_file_path", None, current_sid)
        current_file_name = SessionManager.get(
            "last_uploaded_file_name", None, current_sid
        )
        current_file_hash = str(SessionManager.get("file_hash", "", current_sid) or "")
        current_embedding_model = SessionManager.get(
            "last_selected_embedding_model", None, current_sid
        )

        # [FIX] 임베딩 변경 재빌드도 동기 init으로 즉시 running 카드를 노출한다.
        if current_file_name:
            SessionManager.init_build_state(
                str(current_file_name),
                current_file_hash,
                session_id=current_sid,
            )

        _safe_dispatch(
            "rebuild",
            _bg_rebuild_task(
                current_sid,
                current_file_path,
                current_file_name,
                current_embedding_model,
                expected_file_hash=current_file_hash,
            ),
        )
        needs_rerun = True

    elif SessionManager.get("needs_qa_chain_update", False, current_sid):
        SessionManager.set("needs_qa_chain_update", False, current_sid)
        # [INT-3] 이미 교체 진행 중이면 중복 디스패치 금지 (is_swapping_model 가드)
        if not SessionManager.get("is_swapping_model", False, current_sid):
            SessionManager.set("is_swapping_model", True, current_sid)

            _safe_dispatch("qa_update", _bg_update_qa_chain(current_sid))
        needs_rerun = True

    if needs_rerun:
        st.rerun()


def _ensure_ui_globals() -> None:
    """Pass-1-safe UI bootstrap: session init + bridge sync."""
    from core.session import SessionManager
    from ui.bridge import UIBridge

    SessionManager.init_session()
    UIBridge.sync_session()


def main() -> None:
    from core.session import SessionManager
    from ui.ui import inject_custom_css

    _t_main = time.perf_counter()
    inject_custom_css()  # light import (ui.ui only); CSS lands in the first frame
    _ensure_ui_globals()  # baseline state 원자적 동기화
    logger.debug("[PERF] main(): css+globals took %.3fs", time.perf_counter() - _t_main)

    # [최적화] 모델 목록 1회만 조회하여 available_models와 reachable 여부를 동시에 도출
    current_models = _load_available_models(non_blocking_fallback=True)
    st.session_state.available_models_list = current_models
    ollama_reachable = bool(current_models) and (
        MSG_ERROR_OLLAMA_NOT_RUNNING not in current_models
    )

    if SessionManager.get("pdf_file_path") and not SessionManager.get("pdf_processed"):
        SessionManager.set("is_generating_answer", False)

    if not ollama_reachable:
        st.error(t("error_ollama_not_running"))

    _t_layout = time.perf_counter()
    _render_app_layout(
        available_models=current_models, ollama_reachable=ollama_reachable
    )
    logger.debug(
        "[PERF] main(): _render_app_layout took %.3fs",
        time.perf_counter() - _t_layout,
    )

    _t_pending = time.perf_counter()
    _handle_pending_tasks()
    logger.debug(
        "[PERF] main(): _handle_pending_tasks took %.3fs",
        time.perf_counter() - _t_pending,
    )

    if SessionManager.get("is_first_run"):
        SessionManager.set("is_first_run", False)

    # [FIX-DEFER-WARMUP] 첫 프레임 UI(온보딩 가이드 패널 포함)가 완벽히 그려진 후 유휴 시간에 프리웜 실행
    _trigger_global_warmup_once()
    logger.debug(
        "[PERF] main(): TOTAL first paint complete in %.3fs",
        time.perf_counter() - _t_main,
    )

    # [FIX-GIL-CONTENTION] 모델 목록/pymupdf 예열 스레드는 첫 프레임이
    # 완전히 그려진 뒤에 시작한다. 모듈 임포트 시점에 띄우면 메인 렌더
    # 스레드와 GIL을 경합해 첫 페인트가 지연된다.
    _start_model_list_warmup()


# 빌드 진행 Rerun 제어: 메이저 위상(Phase) 전이 시에만 전역 Rerun을 허용한다.
# 동일 위상 내의 세부 진행률(%) 변경은 전역 main() 리런을 유발하지 않는다.
_INITIAL_PAINT_WINDOW_S = 0.8
_BUILD_RERUN_LAST: dict[str, tuple[float, str]] = {}


def _request_build_progress_rerun(session_id: str, phase_name: str) -> None:
    """메이저 위상 전이 시에만 전역 Rerun을 트리거합니다.

    동일 위상 내의 세부 진행률(%) 갱신은 무시하여 CSS 주입·사이드바·메인
    레이아웃 전체를 다시 평가하는 Full Rerun 폭사를 차단합니다. 단, 업로드
    직후 첫 페인트 보호 윈도우(0.8s) 동안에는 모든 비-terminal 위상 전이를
    억제하여 브라우저가 좌측 1페이지를 먼저 그릴 시간을 보장합니다.
    """
    from core.build_state import BuildPhase, is_terminal
    from core.session import SessionManager

    if not SessionManager.get("is_building_rag", False, session_id):
        _BUILD_RERUN_LAST.pop(session_id, None)
        return
    build = SessionManager.get_build_state(session_id=session_id)
    if build is not None and (
        is_terminal(build.phase) or build.phase == BuildPhase.DONE
    ):
        _BUILD_RERUN_LAST.pop(session_id, None)
        return

    now = time.monotonic()
    last = _BUILD_RERUN_LAST.get(session_id)

    # [UX-FIRST-PAGE] 업로드 직후 첫 페인트 보호 윈도우(0.8s) 내에서는 어떤
    # 비-terminal 위상 전이도 rerun을 유발하지 않는다. 마커는 갱신하지 않아
    # 윈도우 시작 시각이 고정되며, 윈도우가 끝난 뒤 첫 이벤트가 패널을 갱신한다.
    if (
        last is not None
        and last[1] == "preparing"
        and (now - last[0] < _INITIAL_PAINT_WINDOW_S)
    ):
        return

    # 동일 위상 내 세부 진행률(%) 갱신은 전역 Rerun을 유발하지 않는다.
    if last is not None and last[1] == phase_name:
        return

    _BUILD_RERUN_LAST[session_id] = (now, phase_name)
    try:
        from common.async_worker import request_session_rerun

        request_session_rerun(session_id)
    except ImportError:
        return


async def _bg_rebuild_task(
    session_id: str,
    file_path: str,
    file_name: str,
    embedder_name: str,
    *,
    expected_file_hash: str | None = None,
):
    """
    [Background Task] 업로드된 파일과 선택된 모델을 사용하여 RAG 파이프라인을 비동기로 재구축합니다.
    """
    from core.build_state import (
        BuildEvent,
        BuildPhase,
        cancel_build,
        complete_build,
        fail_build,
        is_terminal,
    )
    from core.model_loader import ModelManager
    from core.rag_core import RAGSystem
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    worker_file_hash = str(
        expected_file_hash
        if expected_file_hash is not None
        else SessionManager.get("file_hash", "", session_id) or ""
    )

    # A newer upload may have reset/replaced the build state before this
    # worker body runs. Never touch flags/state in that case.
    _initial_state = SessionManager.get_build_state(session_id=session_id)
    if expected_file_hash is not None and (
        _initial_state is None or _initial_state.file_hash != expected_file_hash
    ):
        logger.info(
            "[MAIN] stale build worker ignored (session=%s expected=%s current=%s)",
            session_id,
            expected_file_hash,
            None if _initial_state is None else _initial_state.file_hash,
        )
        return

    SessionManager.set("rebuild_error", None, session_id=session_id)
    SessionManager.set("rebuild_start_time", time.time(), session_id=session_id)
    # 단일 상태 객체 초기화 (3B: 평탄 레거시 이중기록 폐지).
    # 업로드 콜백에서 이미 동기 init한 동일 파일의 비터미널 상태가 있으면
    # 재사용한다 (started_at 보존, 첫 페인트의 running 카드와 연속성 유지).
    _existing = SessionManager.get_build_state(session_id=session_id)
    if (
        _existing is None
        or _existing.file_name != file_name
        or _existing.file_hash != worker_file_hash
        or is_terminal(_existing.phase)
    ):
        SessionManager.init_build_state(
            file_name,
            worker_file_hash,
            session_id=session_id,
        )
    else:
        SessionManager.set("is_building_rag", True, session_id=session_id)
        SessionManager.set("rebuild_cancelled", False, session_id=session_id)

    _started_state = SessionManager.get_build_state(session_id=session_id)
    build_token = _started_state.started_at if _started_state is not None else None

    def _is_current_build() -> bool:
        """True only while this worker's BuildState is still the session's own."""
        current = SessionManager.get_build_state(session_id=session_id)
        if current is None or build_token is None:
            return False
        return (
            current.started_at == build_token and current.file_hash == worker_file_hash
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

    def _report_progress(pct: int, msg: str = "", detail: str = ""):
        # 3B: 평탄 레거시 키 갱신 폐지. 진행률은 BuildEvent 경로(_report_event)
        # 단일 소스로만 전파된다. 시그니처는 build_pipeline 콜백 호환용으로 유지.
        _ = (pct, msg, detail)

    def _report_event(event: BuildEvent) -> None:
        if not _is_current_build():
            logger.debug(
                "[MAIN] stale build event ignored (session=%s phase=%s)",
                session_id,
                event.phase.value,
            )
            return
        SessionManager.update_build_state(
            event, session_id=session_id, status_text=event.status or None
        )
        if is_terminal(event.phase) or event.phase == BuildPhase.DONE:
            _BUILD_RERUN_LAST.pop(session_id, None)
        else:
            # [FIX] 진행 중 화면 전파: 메이저 위상 전이일 때만 rerun을 유발한다.
            # 동일 위상 내 세부 % 갱신은 전역 main() 리런을 만들지 않는다.
            # Core(SessionManager)는 UI를 모르므로 호출은 여기서 수행한다.
            _request_build_progress_rerun(session_id, event.phase.value)

    def _finalize_cancelled(reason: str) -> None:
        if not _is_current_build():
            return
        build = SessionManager.get_build_state(session_id=session_id)
        if build is not None:
            SessionManager.finalize_build_state(
                cancel_build(build), session_id=session_id
            )
        _BUILD_RERUN_LAST.pop(session_id, None)
        SessionManager.set("rebuild_cancelled", False, session_id=session_id)
        SessionManager.add_status_log(
            "Document analysis cancelled.", session_id=session_id
        )
        SessionManager.add_message(
            "system",
            "Document analysis cancelled",
            msg_type="build_error",
            msg_id=build_msg_id,
            error=reason,
            session_id=session_id,
        )

    try:
        embedder = await ModelManager.get_embedder(embedder_name)

        # Check cancellation before starting build
        if SessionManager.get("rebuild_cancelled", False, session_id=session_id):
            logger.info(f"[MAIN] Rebuild cancelled by user for session {session_id}")
            _finalize_cancelled("Document analysis cancelled.")
            return

        rag_sys = RAGSystem(session_id=session_id)

        # Check cancellation again before expensive build_pipeline call
        if SessionManager.get("rebuild_cancelled", False, session_id=session_id):
            logger.info(f"[MAIN] Rebuild cancelled by user for session {session_id}")
            _finalize_cancelled("cancelled by the user")
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
            on_event=_report_event,
        )

        if not _is_current_build():
            logger.info(
                "[MAIN] stale build completion ignored (session=%s)", session_id
            )
            return

        done_build = SessionManager.get_build_state(session_id=session_id)
        if done_build is not None:
            SessionManager.finalize_build_state(
                complete_build(done_build, cache_used=bool(cache_used)),
                session_id=session_id,
            )
        _BUILD_RERUN_LAST.pop(session_id, None)
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
    except asyncio.CancelledError:
        logger.info(
            f"[MAIN] Rebuild pipeline cancelled mid-build for session {session_id}"
        )
        _finalize_cancelled("async task cancelled")
    except Exception as e:
        logger.error(f"Background RAG rebuild error: {e}", exc_info=True)
        if not _is_current_build():
            logger.info("[MAIN] stale build error ignored (session=%s)", session_id)
            return
        error_msg = "문서 분석 중 오류가 발생했습니다. 파일 상태를 확인하신 후 다시 업로드해 주세요."
        failed = SessionManager.get_build_state(session_id=session_id)
        if failed is not None:
            SessionManager.finalize_build_state(
                fail_build(failed, error_msg), session_id=session_id
            )
        _BUILD_RERUN_LAST.pop(session_id, None)
        SessionManager.set("rebuild_error", error_msg, session_id=session_id)
        SessionManager.set("pdf_processing_error", error_msg, session_id=session_id)
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
        # Only the worker that still owns the current BuildState may clear the
        # shared build flag / throttle marker. A stale worker must not unfreeze
        # or re-render the newer build.
        if _is_current_build():
            SessionManager.set("is_building_rag", False, session_id=session_id)
            _BUILD_RERUN_LAST.pop(session_id, None)


async def _bg_update_qa_chain(session_id: str) -> None:
    """[Background Task] 문서 인덱싱은 유지한 채 LLM(QA Chain)만 비동기 작업 스레드에서 교체합니다."""
    from core.model_loader import load_llm
    from core.session import SessionManager

    SessionManager.set_session_id(session_id)
    SessionManager.set("is_swapping_model", True, session_id=session_id)

    def _swap_sync() -> None:
        selected_model = SessionManager.get(
            "last_selected_model", session_id=session_id
        )
        try:
            SessionManager.add_status_log(
                "Switching inference model...", session_id=session_id
            )
            model_name = str(selected_model or DEFAULT_OLLAMA_MODEL)
            llm = load_llm(model_name)
            SessionManager.set("llm", llm, session_id=session_id)
            SessionManager.add_status_log(
                "Inference model switched", session_id=session_id
            )
        except Exception as e:
            error_msg = "답변 생성 모델을 변경하는 중 오류가 발생했습니다."
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

    try:
        await asyncio.to_thread(_swap_sync)
    finally:
        SessionManager.set("is_swapping_model", False, session_id=session_id)


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


if __name__ == "__main__":
    main()
