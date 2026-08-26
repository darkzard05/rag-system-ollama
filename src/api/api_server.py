"""
FastAPI 기반 RAG 시스템 백엔드 서버
UI와 독립적으로 RAG 기능을 외부 API로 제공합니다.
"""

import asyncio
import logging
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.schemas import (
    LoginRequest,
    LogoutRequest,
    QueryRequest,
    QueryResponse,
    TokenResponse,
)
from api.streaming_handler import (
    ServerSentEventsHandler,
    get_adaptive_controller,
    get_streaming_handler,
)
from common.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_OLLAMA_MODEL
from common.constants import FilePathConstants
from common.exceptions import PDFProcessingError
from core.document_processor import compute_file_hash
from core.rag_core import RAGSystem
from core.resource_manager import get_resource_manager
from core.session import SessionManager
from security.auth_system import AuthenticationManager

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

# --- 보안 및 인증 설정 ---
auth_scheme = HTTPBearer()
auth_manager = AuthenticationManager()

# [임시] 테스트용 유저 및 API 키 등록 (CI 환경 호환성 위해 환경 변수 지원)
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


TEST_PASSWORD, TEST_API_KEY = _bootstrap_credentials(auth_manager)


# --- 세션/파일 소유권 레지스트리 ---
# session_id/file_hash -> (소유 user_id, 바인딩 시각 epoch).
# 값 타입: 바인딩 시각을 포함한 tuple로 확장 (stale 항목 스윕용).
#
# 소유권 정책 (명시적 결정):
# - FILES: fail-closed. 미등록(unbound) 파일이거나 다른 사용자가 소유한 파일이면 403.
#   UI 업로드 PDF(src/main.py)는 동일 temp 디렉터리에 저장되지만 절대 바인딩되지 않으며,
#   UI/API는 별도 프로세스라 메모리 레지스트리를 공유할 수 없다. 따라서 fail-closed로
#   미등록 파일을 API 서빙 불가(403) 처리하여 크로스 유저 /pdf/{hash} 구멍을 차단한다.
#   UI는 PDF를 로컬 렌더링(streamlit-pdf-viewer)하며 /api/v1/pdf 를 호출하지 않으므로 UX 손실이 없다.
# - SESSIONS: first-use claim (fail-closed 기반). 미등록 세션은 최초 인증 사용자가
#   소유권을 점유(claim)하며, 이후 다른 사용자의 접근은 403으로 차단된다(크로스 유저 구멍 폐쇄).
#   단일 사용자 로컬 플로우(업로드 전 질의 400 등)는 유지되며, /login 시점에도 세션이 바인딩된다.
#   문서 접근은 파일 소유권(_require_file_owner)이 별도로 관장한다.
_OWNER_TTL_SECONDS = 7 * 24 * 3600
# PDF 라이브러리 보존 기간: 세션 정리와 분리되어 오래된 업로드 파일을 수거
_PDF_RETENTION_DAYS = 30

# 입력 경계
MAX_PDF_SIZE_BYTES = 50 * 1024 * 1024  # 50MB
MAX_QUERY_LENGTH = 4000
MAX_SESSION_ID_LENGTH = 64

_session_owners: dict[str, tuple[str, float]] = {}
_file_owners: dict[str, tuple[str, float]] = {}
_owners_lock = threading.RLock()


def _bind_session_owner(session_id: str, user_id: str) -> None:
    with _owners_lock:
        if session_id not in _session_owners:
            _session_owners[session_id] = (user_id, time.time())


def _require_session_owner(session_id: str, user_id: str) -> None:
    with _owners_lock:
        entry = _session_owners.get(session_id)
        if entry is None:
            # 미등록 세션: 최초 인증 사용자가 소유권을 점유(claim)한다.
            # 이로써 다른 사용자가 해당 세션에 접근하는 구멍을 차단하며,
            # 단일 사용자 로컬 플로우(업로드 전 질의 400 등)는 유지된다.
            _session_owners[session_id] = (user_id, time.time())
            return
        owner = entry[0]
    if owner != user_id:
        raise HTTPException(
            status_code=403, detail="다른 사용자의 세션에 접근할 수 없습니다."
        )


def _validate_session_id(session_id: str) -> None:
    """세션 ID 입력 경계 검증 (과도한 길이의 입력 거부)."""
    if len(session_id) > MAX_SESSION_ID_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"세션 ID는 {MAX_SESSION_ID_LENGTH}자 이하여야 합니다.",
        )


def _bind_file_owner(file_hash: str, user_id: str) -> None:
    with _owners_lock:
        if file_hash not in _file_owners:
            _file_owners[file_hash] = (user_id, time.time())


def _require_file_owner(file_hash: str, user_id: str) -> None:
    with _owners_lock:
        entry = _file_owners.get(file_hash)
    owner = entry[0] if entry else None
    # fail-closed: 미등록(unbound) 또는 타인 소유 파일은 403.
    if owner is None or owner != user_id:
        raise HTTPException(status_code=403, detail="문서에 접근할 권한이 없습니다.")


def _sweep_stale_owners() -> None:
    """만료되었거나 디스크에서 삭제된 소유권 항목을 제거합니다.

    - 세션: TTL(_OWNER_TTL_SECONDS)을 초과한 바인딩 제거
    - 파일: TTL 초과 또는 스토리지에 PDF 파일이 존재하지 않는 항목 제거
    """
    now = time.time()
    with _owners_lock:
        for session_id, entry in list(_session_owners.items()):
            if now - entry[1] > _OWNER_TTL_SECONDS:
                del _session_owners[session_id]
        storage = Path(PDF_STORAGE_DIR)
        for file_hash, entry in list(_file_owners.items()):
            stale = now - entry[1] > _OWNER_TTL_SECONDS
            if stale or not (storage / f"{file_hash}.pdf").exists():
                del _file_owners[file_hash]


def _sweep_expired_library_files() -> None:
    """보존 기간(_PDF_RETENTION_DAYS)을 초과한 PDF 라이브러리 파일과 그 소유권 항목을 제거합니다."""
    storage = Path(PDF_STORAGE_DIR)
    if not storage.is_dir():
        return
    cutoff = time.time() - _PDF_RETENTION_DAYS * 24 * 3600
    for path in storage.glob("*.pdf"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                with _owners_lock:
                    _file_owners.pop(path.stem, None)
        except OSError:
            continue


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    """토큰 유효성을 검증하는 공통 의존성"""
    token = credentials.credentials
    # API Key 또는 JWT 토큰 모두 지원
    user_id = auth_manager.verify_api_key(token) or auth_manager.verify_token(token)

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="유효하지 않거나 만료된 인증 토큰입니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id


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


# --- 세션 격리 의존성 ---
async def get_session_context(x_session_id: str | None = Header(None)) -> str:
    """헤더에서 세션 ID를 추출하고 컨텍스트를 고정합니다."""
    sid = x_session_id or "default"
    # [핵심] API 요청 스레드의 컨텍스트 변수에 세션 ID 주입
    from core.session import SessionManager

    SessionManager.set_session_id(sid)

    # [최적화] 가벼운 세션 초기화는 직접 호출
    SessionManager.init_session(session_id=sid)
    return sid


# --- PDF 서빙 및 참조 좌표 노출 ---
# 업로드된 PDF를 file_hash 기반으로 보존하는 스토리지 루트
PDF_STORAGE_DIR = str(FilePathConstants.TEMP_DIR / "pdf_library")


def _resolve_pdf_path(file_hash: str) -> Path | None:
    """스토리지 루트 내에서 file_hash와 일치하는 PDF 경로를 반환합니다.

    경로 주입을 방지하기 위해 file_hash를 파일명 안전 문자열로 정규화하고,
    최종 후보 경로가 항상 스토리지 루트 내부에 위치하는지 검증합니다.
    """
    storage = Path(PDF_STORAGE_DIR)
    if not storage.is_dir():
        return None

    # 경로 구분자/상대 경로 제거 (path traversal 방지)
    safe_hash = (file_hash or "").replace("/", "").replace("\\", "").replace("..", "")
    if not safe_hash:
        return None

    # 1) 해시 기반 직접 경로 (upload 엔드포인트가 {hash}.pdf 로 저장한 경우)
    candidate = (storage / safe_hash).with_suffix(".pdf")
    try:
        if candidate.is_file() and candidate.resolve().is_relative_to(
            storage.resolve()
        ):
            return candidate
    except OSError:
        pass

    # 2) 스캔 폴백: 다른 이름으로 저장된 동일 콘텐츠 파일 검색
    try:
        for path in storage.glob("*.pdf"):
            if compute_file_hash(str(path)) == safe_hash:
                return path
    except OSError:
        return None
    return None


def _doc_to_source(doc: Any, max_chars: int = 200, suffix: str = "") -> dict[str, Any]:
    """검색 결과 Document를 API 소스 딕셔너리로 직렬화합니다.

    기존 page/content 필드를 유지하면서 좌표 및 해시 메타데이터를
    존재할 때만 Optional로 노출합니다 (구버전 클라이언트 호환).
    """
    metadata = doc.metadata
    content = doc.page_content[:max_chars]
    source: dict[str, Any] = {
        "page": metadata.get("page"),
        "content": content + suffix,
    }
    for key in ("pages", "page_coords", "word_coords", "file_hash"):
        if metadata.get(key) is not None:
            source[key] = metadata[key]
    return source


# --- Endpoints ---


@app.get("/api/v1/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model": DEFAULT_OLLAMA_MODEL,
    }


@app.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    embedding_model: str | None = Form(None),
    user_id: str = Depends(verify_token),
):
    """
    인증된 사용자의 PDF 문서를 업로드하고 해당 세션에 인덱싱합니다.
    """
    _validate_session_id(session_id)

    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 다른 사용자가 소유한 세션으로의 업로드를 차단 (교차 사용자 내용 주입 방지)
    _require_session_owner(session_id, user_id)
    # [TOCTOU 방지] 소유권 바인딩을 핸들러 최상단으로 이동: 첫 업로드가 진행되는 동안
    # 다른 사용자의 동일 세션 업로드가 교차 주입되는 경합을 차단한다.
    _bind_session_owner(session_id, user_id)

    # 명시적으로 세션 초기화 (최적화: 직접 호출)
    SessionManager.init_session(session_id=session_id)

    try:
        # PDF를 file_hash 기반 스토리지에 영구 보존 (PDF 서빙 + 좌표 하이드레이션용)
        # [입력 경계] 파일 크기 제한: 메모리 고갈 방지를 위해 청크 단위로 읽으며 상한을 초과하면 거부
        content = b""
        total = 0
        while chunk := await file.read(1024 * 1024):  # 1MB 청크
            total += len(chunk)
            if total > MAX_PDF_SIZE_BYTES:
                raise HTTPException(
                    status_code=413, detail="파일 크기가 50MB를 초과합니다."
                )
            content += chunk
        file_hash = compute_file_hash("", data=content)
        # [TOCTOU 방지] 파일 소유권 바인딩을 파일 저장 직전으로 이동.
        _bind_file_owner(file_hash, user_id)
        storage = Path(PDF_STORAGE_DIR)
        storage.mkdir(parents=True, exist_ok=True)
        persisted_path = storage / f"{file_hash}.pdf"
        persisted_path.write_bytes(content)

        # [수정] 동적으로 결정된 임베딩 모델 사용 (모델 로딩은 무거우므로 스레드 유지)
        embedder = await get_resource_manager().get_embedder_for_session(
            session_id, embedding_model
        )

        # [중요] RAGSystem 클래스를 통해 세션 격리 보장
        rag_sys = RAGSystem(session_id=session_id)
        msg, cache_used = await rag_sys.build_pipeline(
            file_path=str(persisted_path),
            file_name=file.filename,
            embedder=embedder,
        )

        SessionManager.set(
            "last_uploaded_file_name", file.filename, session_id=session_id
        )
        SessionManager.set("file_hash", file_hash, session_id=session_id)
        SessionManager.set("pdf_file_path", str(persisted_path), session_id=session_id)
        SessionManager.set(
            "pdf_library_path", str(persisted_path), session_id=session_id
        )
        logger.info(
            f"[API] 문서 인덱싱 완료: {file.filename} (Session: {session_id}, Cache: {cache_used})"
        )

        return {
            "message": msg,
            "filename": file.filename,
            "session_id": session_id,
            "cache_used": cache_used,
            "file_hash": file_hash,
        }

    except (OSError, ValueError) as e:
        logger.error(f"업로드 오류 (Session: {session_id}): {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="문서 업로드 처리 중 오류가 발생했습니다."
        ) from e


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user_id: str = Depends(verify_token),
    _session_ctx: str = Depends(get_session_context),
):
    """
    인증된 세션 컨텍스트에서 질의를 수행합니다.
    """
    sid = request.session_id or "default"
    _validate_session_id(sid)
    if len(request.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"질문은 {MAX_QUERY_LENGTH}자 이하여야 합니다."
        )
    SessionManager.init_session(session_id=sid)
    _require_session_owner(sid, user_id)

    if SessionManager.get("last_uploaded_file_name", session_id=sid) is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    start_time = time.time()
    try:
        # [개선] RAGSystem 클래스를 통해 통합된 인터페이스 호출 (설정 및 리소스 관리 자동화)
        rag_sys = RAGSystem(session_id=sid)
        result = await rag_sys.aquery(request.query, model_name=request.model_name)

        execution_time = (time.time() - start_time) * 1000

        sources = []
        for doc in result.get("relevant_docs", result.get("documents", [])):
            sources.append(_doc_to_source(doc, max_chars=200, suffix="..."))

        return QueryResponse(
            answer=result["response"], sources=sources, execution_time_ms=execution_time
        )

    except (RuntimeError, ValueError, KeyError) as e:
        logger.error(f"질의 오류 (Session: {sid}): {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="질의 처리 중 오류가 발생했습니다."
        ) from e


@app.post("/api/v1/stream_query")
async def stream_query_rag(
    request: QueryRequest,
    fastapi_request: Request,
    user_id: str = Depends(verify_token),
    _session_ctx: str = Depends(get_session_context),
):
    """
    인증된 세션에 대해 실시간 스트리밍(SSE) 응답을 제공합니다.
    """
    sid = request.session_id or "default"
    _validate_session_id(sid)
    if len(request.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"질문은 {MAX_QUERY_LENGTH}자 이하여야 합니다."
        )
    SessionManager.init_session(session_id=sid)
    _require_session_owner(sid, user_id)

    file_name = SessionManager.get("last_uploaded_file_name", session_id=sid)
    logger.debug(f"[TEST] Session ID: {sid}, last_uploaded_file_name: {file_name}")

    if file_name is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    rag_app = SessionManager.get("rag_engine", session_id=sid)
    if rag_app is None:
        raise HTTPException(
            status_code=500, detail="QA 시스템이 초기화되지 않았습니다."
        )

    async def event_generator():
        logger.debug(f"[API] Streaming started for session: {sid}")

        # [개선] RAGSystem 클래스를 통해 통합된 인터페이스 호출 (설정 및 리소스 관리 자동화)
        rag_sys = RAGSystem(session_id=sid)

        handler = get_streaming_handler()
        controller = get_adaptive_controller(client_profile="api")
        sse_handler = ServerSentEventsHandler()

        # 배치 버퍼
        batch_buffer: list[tuple[str | None, dict[str, Any], int | None]] = []
        batch_size = 10  # 10개 이벤트마다 배치 전송
        event_counter = 0

        try:
            # RAGSystem이 직접 생성한 스트림 이벤트를 핸들러에 전달
            async for chunk in handler.stream_graph_events(
                await rag_sys.astream(request.query, model_name=request.model_name),
                adaptive_controller=controller,
            ):
                # 클라이언트 연결 끊김 확인 (자원 보호)
                if await fastapi_request.is_disconnected():
                    logger.info(f"[API] Client disconnected, stopping stream: {sid}")
                    break

                # 1. 상태 업데이트 처리
                if chunk.status:
                    batch_buffer.append(
                        (
                            "status",
                            {"message": chunk.status, "node": chunk.node_name},
                            event_counter,
                        )
                    )
                    event_counter += 1
                    # 비동기 제너레이터 내에서도 명시적 세션 ID 사용 (최적화: 직접 호출)
                    SessionManager.add_status_log(chunk.status, session_id=sid)

                # 2. 메시지(답변 및 사고 과정) 처리
                if chunk.content:
                    batch_buffer.append(
                        (
                            "message",
                            {"content": chunk.content},
                            event_counter,
                        )
                    )
                    event_counter += 1

                if chunk.thought:
                    batch_buffer.append(
                        (
                            "thought",
                            {"content": chunk.thought},
                            event_counter,
                        )
                    )
                    event_counter += 1

                # 3. 메타데이터(문서) 처리
                if chunk.metadata and "documents" in chunk.metadata:
                    docs = [
                        _doc_to_source(d, max_chars=100)
                        for d in chunk.metadata["documents"]
                    ]
                    batch_buffer.append(
                        (
                            "sources",
                            {"documents": docs},
                            event_counter,
                        )
                    )
                    event_counter += 1

                # 배치 크기 도달 시 전송
                if len(batch_buffer) >= batch_size:
                    yield sse_handler.format_sse_batch(batch_buffer)
                    batch_buffer.clear()

            # 남은 버퍼 전송
            if batch_buffer:
                yield sse_handler.format_sse_batch(batch_buffer)
                batch_buffer.clear()

            yield sse_handler.format_sse_event("end", {"status": "done"}, event_counter)
        except (RuntimeError, ValueError, ConnectionError, PDFProcessingError) as e:
            logger.error(f"Streaming error (Session: {sid}): {e}")
            yield sse_handler.format_sse_error(str(e))

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 버퍼링 방지
        },
    )


@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(verify_token)):
    """특정 세션의 데이터를 삭제하고 메모리를 해제합니다. (소유자만 가능)"""
    _validate_session_id(session_id)
    _require_session_owner(session_id, user_id)
    success = SessionManager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    with _owners_lock:
        _session_owners.pop(session_id, None)
    return {"message": f"Session {session_id} deleted successfully"}


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


@app.get("/api/v1/pdf/{file_hash}")
async def serve_pdf(
    file_hash: str,
    user_id: str = Depends(verify_token),
) -> FileResponse:
    """file_hash로 저장된 PDF 문서를 반환합니다.

    브라우저 PDF 뷰어에서 #page=N 으로 페이지 이동을 지원합니다.
    인증이 필요하며, 경로 주입 공격을 방지합니다.
    """
    # resolve-first: 파일이 존재하지 않으면 소유권 검사 이전에 404 (unbound 여부와 무관).
    pdf_path = _resolve_pdf_path(file_hash)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF 문서를 찾을 수 없습니다.")
    # fail-closed: 미등록(unbound)/타인 소유 파일은 403.
    _require_file_owner(file_hash, user_id)
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
