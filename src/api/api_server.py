"""
FastAPI 기반 RAG 시스템 백엔드 서버
UI와 독립적으로 RAG 기능을 외부 API로 제공합니다.

라우트 분리 구성 (동작 보존 리팩토링):
- routes_chat.py: /api/v1/query, /api/v1/stream_query, DELETE /api/v1/session/{id}
- routes_docs.py: /api/v1/upload, GET /api/v1/pdf/{file_hash}
- _deps.py: 라우트들이 공유하는 인증/소유권 헬퍼와 상태

본 모듈은 app 인스턴스, lifespan, 인증 계열 라우트(login/logout/health/admin)
및 공통 정적 자원을 유지하고 두 라우터를 include_router 로 등록합니다.
"""

import asyncio
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials

from api._deps import (
    _bind_file_owner,
    _bind_session_owner,
    _doc_to_source,
    _file_owners,
    _owners_lock,
    _require_file_owner,
    _require_session_owner,
    _resolve_pdf_path,
    _session_owners,
    _sweep_expired_library_files,
    _sweep_stale_owners,
    _validate_session_id,
    auth_manager,
    auth_scheme,
    get_session_context,
    run_bootstrap_once,
    verify_token,
)
from api.routes_chat import router as chat_router
from api.routes_docs import router as docs_router
from api.schemas import LoginRequest, LogoutRequest, TokenResponse
from common.config import (
    CORS_ALLOW_ORIGINS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from common.constants import FilePathConstants
from core.rag_core import RAGSystem
from core.session import SessionManager
from security.auth_system import AuthenticationManager

# 외부 소비처/테스트가 api_server 네임스페이스에서 접근하는 재수출 심볼
__all__ = [
    "RAGSystem",
    "SessionManager",
    "auth_manager",
    "verify_token",
    "_bind_session_owner",
    "_bind_file_owner",
    "_require_session_owner",
    "_require_file_owner",
    "_validate_session_id",
    "_sweep_stale_owners",
    "_sweep_expired_library_files",
    "_resolve_pdf_path",
    "_doc_to_source",
    "_session_owners",
    "_file_owners",
    "_owners_lock",
    "get_session_context",
]

# 로깅 설정
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 생애주기 관리 (Startup/Shutdown)"""

    # Startup: 세션 자동 정리 태스크 시작
    async def session_cleaner():
        try:
            while True:
                await asyncio.sleep(600)  # 10분마다 실행
                # 세션 정리 및 보안 감사 수행
                await asyncio.to_thread(SessionManager.cleanup_expired_sessions, 3600)
                await asyncio.to_thread(SessionManager.perform_security_audit)
                await asyncio.to_thread(_sweep_stale_owners)
                await asyncio.to_thread(_sweep_expired_library_files)
        except asyncio.CancelledError:
            logger.info("[API] 세션 정리 태스크 종료 중...")

    cleaner_task = asyncio.create_task(session_cleaner())
    logger.info("[API] 세션 자동 정리 태스크 시작됨 (주기: 10분)")

    # [WARMUP] 시작 시 LLM+임베더 1회 프리웜 — 첫 쿼리 TTFT 제거.
    # Ollama 미연결 등 실패 시에도 서버 시작은 비차단/비치명적으로 계속된다.
    try:
        from core.model_loader import _warmup_models

        await _warmup_models()
    except Exception as e:
        logger.warning(f"[WARMUP] 모델 프리웜 실패 — 첫 쿼리에서 로드됨: {e}")

    yield

    # Shutdown: 태스크 정리
    cleaner_task.cancel()
    import contextlib

    with contextlib.suppress(asyncio.CancelledError):
        await cleaner_task

    # [추가] 서버 종료 시 VRAM 명시적 해제
    from core.model_loader import ModelManager

    await ModelManager.clear_vram()

    logger.info("[API] 서버 종료 및 리소스 정리 완료")


app = FastAPI(
    title="RAG System API",
    description="Ollama와 LangGraph 기반의 고도화된 RAG 시스템 API",
    version="2.0.0",
    lifespan=lifespan,
)

# D20: CORS는 기본 차단(allow_origins=[]) — 브라우저 교차-오리진 요청은
# config.yml의 cors.allow_origins에 명시적으로 등록된 오리진만 허용한다.
# allow_credentials=True와 "*"는 함께 사용할 수 없으므로 와일드카드를 금지한다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

# --- 보안 및 인증 설정 ---
# auth_scheme/auth_manager 는 api/_deps 에 단일 인스턴스로 정의되어
# 라우트 모듈과 공유된다 (dependency_overrides 키 동일성 유지).
TEST_USER = "admin"

# AUTO_BOOTSTRAP_ADMIN: 기본 활성(true). 명시적으로 비활성화(0/no/false/"")하면
# 기본 admin 계정 생성을 건너뛴다. 자동 생성된 비밀번호/API 키 값은 로그에 출력되지 않는다.
_BOOTSTRAP_DEFAULT = os.getenv("AUTO_BOOTSTRAP_ADMIN", "true").lower() not in (
    "0",
    "no",
    "false",
    "",
)

TEST_PASSWORD: str = ""
TEST_API_KEY: str = ""


def _bootstrap_credentials(auth_manager: AuthenticationManager) -> tuple[str, str]:
    """관리자 크리덴셜을 준비한다. AUTO_BOOTSTRAP_ADMIN 가 명시적으로 비활성화된 경우 부트스트랩을 건너뛴다."""
    if not _BOOTSTRAP_DEFAULT:
        logger.warning(
            "AUTO_BOOTSTRAP_ADMIN 이 비활성화되어 기본 admin 계정을 생성하지 않습니다."
        )
        return "", ""
    env_password = os.getenv("TEST_ADMIN_PASSWORD")
    if env_password:
        password = env_password
    else:
        password = secrets.token_urlsafe(12)
        logger.warning(
            "관리자 비밀번호가 자동 생성되었습니다. TEST_ADMIN_PASSWORD 로 고정하거나 "
            "환경 변수로 주입하세요 (값은 출력되지 않습니다)."
        )
    auth_manager.upsert_admin_credentials(TEST_USER, "admin_user", password)

    env_api_key = os.getenv("TEST_API_KEY")
    if env_api_key:
        auth_manager.register_fixed_api_key(TEST_USER, env_api_key, expires_in=2592000)
        api_key = env_api_key
    else:
        api_key = auth_manager.create_api_key(TEST_USER, expires_in=86400)
        logger.warning(
            "관리자 API Key 가 자동 생성되었습니다 (값은 출력되지 않습니다)."
        )
    return password, api_key


TEST_PASSWORD, TEST_API_KEY = run_bootstrap_once(
    lambda: _bootstrap_credentials(auth_manager)
)


@app.post("/api/v1/login")
async def login(request: LoginRequest, client_request: Request) -> TokenResponse:
    """사용자 이름/비밀번호로 접근 토큰을 발급합니다."""
    client_ip = client_request.client.host if client_request.client else None
    result = auth_manager.authenticate_by_username(
        request.username, request.password, client_ip
    )
    if result is None:
        raise HTTPException(
            status_code=401, detail="사용자 이름 또는 비밀번호가 올바르지 않습니다."
        )
    access_token, session_id = result
    owner = auth_manager.verify_token(access_token)
    if owner:
        _bind_session_owner(session_id, owner)
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=3600,
        session_id=session_id,
    )


@app.post("/api/v1/logout")
async def logout(
    request: LogoutRequest | None = None,
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """현재 접근 토큰을 무효화하고, 제공된 세션을 비활성화합니다."""
    if not auth_manager.revoke_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="유효하지 않은 토큰입니다.")
    if request and request.session_id:
        auth_manager.logout(request.session_id)
    return {"message": "로그아웃되었습니다."}


# --- PDF 스토리지 루트 및 입력 경계 ---
# 테스트가 api_server.PDF_STORAGE_DIR / MAX_PDF_SIZE_BYTES 를 패치하므로
# 본 모듈에 정의한다. 라우트/헬퍼는 호출 시점에 _app_server_module() 로 읽는다.
PDF_STORAGE_DIR = str(FilePathConstants.TEMP_DIR / "pdf_library")
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


@app.get("/api/v1/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model": DEFAULT_OLLAMA_MODEL,
    }


@app.get("/api/v1/admin/stats")
async def get_system_stats(user_id: str = Depends(verify_token)):
    """시스템 전체 통계 및 세션 정보를 반환합니다. (관리자 전용)"""
    if not auth_manager.is_admin(user_id):
        raise HTTPException(status_code=403, detail="관리자만 접근할 수 있습니다.")
    return {
        "session_stats": SessionManager.get_stats(),
        "auth_stats": auth_manager.get_statistics(),
        "active_models": {
            "llm": DEFAULT_OLLAMA_MODEL,
            "embedding": DEFAULT_EMBEDDING_MODEL,
        },
    }


# --- 라우터 등록 (경로/스키마 불변 — routes_chat/routes_docs 로 이동된 엔드포인트) ---
app.include_router(chat_router)
app.include_router(docs_router)
