"""
FastAPI 기반 RAG 시스템 백엔드 서버
UI와 독립적으로 RAG 기능을 외부 API로 제공합니다.
"""

import asyncio
import logging
import os
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

from api.schemas import QueryRequest, QueryResponse
from api.streaming_handler import (
    ServerSentEventsHandler,
    get_adaptive_controller,
    get_streaming_handler,
)
from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_OLLAMA_MODEL
from common.constants import FilePathConstants
from core.document_processor import compute_file_hash
from core.rag_core import RAGSystem
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
        except asyncio.CancelledError:
            logger.info("[API] 세션 정리 태스크 종료 중...")

    cleaner_task = asyncio.create_task(session_cleaner())
    logger.info("[API] 세션 자동 정리 태스크 시작됨 (주기: 10분)")

    yield

    # Shutdown: 태스크 정리
    cleaner_task.cancel()
    import contextlib

    with contextlib.suppress(asyncio.CancelledError):
        await cleaner_task

    # [추가] 서버 종료 시 VRAM 명시적 해제
    from core.model_loader import ModelManager

    ModelManager.clear_vram()

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
TEST_API_KEY = os.getenv("TEST_API_KEY")

_env_password = os.getenv("TEST_ADMIN_PASSWORD")
if _env_password:
    TEST_PASSWORD = _env_password
    logger.info("[Security] 환경변수에서 관리자 비밀번호 로드 완료")
else:
    TEST_PASSWORD = "admin"
    logger.warning(
        "[Security] TEST_ADMIN_PASSWORD 미설정 — 개발용 기본 비밀번호('admin') 사용. "
        "프로덕션 환경에서는 반드시 환경변수를 설정하세요."
    )

auth_manager.register_user(TEST_USER, "admin_user", TEST_PASSWORD)

if TEST_API_KEY:
    # 지정된 키로 등록 (CI용, 30일 만료)
    auth_manager.register_fixed_api_key(TEST_USER, TEST_API_KEY, expires_in=2592000)
    logger.info("[Security] 고정 API 키 활성화 (CI/Test 모드, 30일 유효)")
else:
    # 무작위 키 생성 (일반 실행 모드, 24시간 만료)
    TEST_API_KEY = auth_manager.create_api_key(TEST_USER, expires_in=86400)
    logger.info(
        "[Security] 시스템 보호 활성화. 24시간 유효한 API Key가 생성되었습니다."
    )


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


# --- 싱글톤 리소스 관리 (중앙 ModelManager 위임) ---
class RAGResourceManager:
    @classmethod
    async def get_llm(cls, model_name: str | None = None, session_id: str = "default"):
        current_model = SessionManager.get("last_selected_model", session_id=session_id)
        target_model = model_name or current_model or DEFAULT_OLLAMA_MODEL

        # [최적화] 세션별 모델 전환 로그
        if current_model and current_model != target_model:
            logger.info(
                f"[MODEL] [SWITCH] LLM 전환 (Session: {session_id}) | {current_model} -> {target_model}"
            )

        # [최적화] ModelManager의 비동기 메서드 직접 호출 (스레드 전환 오버헤드 제거)
        from core.model_loader import ModelManager

        llm = await ModelManager.get_llm(target_model)
        SessionManager.set("last_selected_model", target_model, session_id=session_id)
        return llm

    @classmethod
    async def get_embedder(
        cls, model_name: str | None = None, session_id: str = "default"
    ):
        current_embedder = SessionManager.get(
            "last_selected_embedding_model", session_id=session_id
        )
        target_model = model_name or current_embedder or AVAILABLE_EMBEDDING_MODELS[0]

        if current_embedder and current_embedder != target_model:
            logger.info(
                f"[MODEL] [SWITCH] 임베딩 모델 전환 (Session: {session_id}) | {current_embedder} -> {target_model}"
            )

        from core.model_loader import ModelManager

        embedder = await ModelManager.get_embedder(target_model)

        SessionManager.set(
            "last_selected_embedding_model", target_model, session_id=session_id
        )
        return embedder


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
PDF_STORAGE_DIR = str(FilePathConstants.TEMP_DIR)


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
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 명시적으로 세션 초기화 (최적화: 직접 호출)
    SessionManager.init_session(session_id=session_id)

    try:
        # PDF를 file_hash 기반 스토리지에 영구 보존 (PDF 서빙 + 좌표 하이드레이션용)
        content = await file.read()
        file_hash = compute_file_hash("", data=content)
        storage = Path(PDF_STORAGE_DIR)
        storage.mkdir(parents=True, exist_ok=True)
        persisted_path = storage / f"{file_hash}.pdf"
        persisted_path.write_bytes(content)

        # [수정] 동적으로 결정된 임베딩 모델 사용 (모델 로딩은 무거우므로 스레드 유지)
        embedder = await RAGResourceManager.get_embedder(
            embedding_model, session_id=session_id
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
    SessionManager.init_session(session_id=sid)

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
    SessionManager.init_session(session_id=sid)

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
        controller = get_adaptive_controller()
        sse_handler = ServerSentEventsHandler()

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
                    yield sse_handler.format_sse_event(
                        "status", {"message": chunk.status, "node": chunk.node_name}
                    )
                    # 비동기 제너레이터 내에서도 명시적 세션 ID 사용 (최적화: 직접 호출)
                    SessionManager.add_status_log(chunk.status, session_id=sid)

                # 2. 메시지(답변 및 사고 과정) 처리
                if chunk.content:
                    yield sse_handler.format_sse_event(
                        "message", {"content": chunk.content}
                    )

                if chunk.thought:
                    yield sse_handler.format_sse_event(
                        "thought", {"content": chunk.thought}
                    )

                # 3. 메타데이터(문서) 처리
                if chunk.metadata and "documents" in chunk.metadata:
                    docs = [
                        _doc_to_source(d, max_chars=100)
                        for d in chunk.metadata["documents"]
                    ]
                    yield sse_handler.format_sse_event("sources", {"documents": docs})

            yield sse_handler.format_sse_event("end", {"status": "done"})
        except (RuntimeError, ValueError, ConnectionError) as e:
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
    """특정 세션의 데이터를 삭제하고 메모리를 해제합니다."""
    success = SessionManager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/api/v1/admin/stats")
async def get_system_stats(user_id: str = Depends(verify_token)):
    """시스템 전체 통계 및 세션 정보를 반환합니다."""
    # admin 유저만 접근 가능하도록 추가 검증 가능
    return {
        "session_stats": SessionManager.get_stats(),
        "auth_stats": auth_manager.get_statistics(),
        "active_models": {
            "llm": DEFAULT_OLLAMA_MODEL,
            "embedding": AVAILABLE_EMBEDDING_MODELS[0],
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
    pdf_path = _resolve_pdf_path(file_hash)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF 문서를 찾을 수 없습니다.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
