"""
FastAPI 기반 RAG 시스템 백엔드 서버
UI와 독립적으로 RAG 기능을 외부 API로 제공합니다.
"""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from api.schemas import QueryRequest, QueryResponse
from api.streaming_handler import (
    ServerSentEventsHandler,
    get_adaptive_controller,
    get_streaming_handler,
)
from common.config import AVAILABLE_EMBEDDING_MODELS, DEFAULT_OLLAMA_MODEL
from core.model_loader import load_embedding_model, load_llm
from core.rag_core import build_rag_pipeline
from core.session import SessionManager
from security.auth_system import AuthenticationManager

# 로깅 설정
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Ollama와 LangGraph 기반의 고도화된 RAG 시스템 API",
    version="2.0.0",
)

# --- 보안 및 인증 설정 ---
auth_scheme = HTTPBearer()
auth_manager = AuthenticationManager()

# [임시] 테스트용 유저 및 API 키 등록 (CI 환경 호환성 위해 환경 변수 지원)
TEST_USER = "admin"
TEST_API_KEY = os.getenv("TEST_API_KEY")

auth_manager.register_user(TEST_USER, "admin_user", "admin123")
if TEST_API_KEY:
    # 지정된 키로 등록 (CI용)
    auth_manager._api_keys[TEST_API_KEY] = TEST_USER
    logger.info("[Security] 고정 API 키 활성화 (CI/Test 모드)")
else:
    # 무작위 키 생성 (일반 실행 모드)
    TEST_API_KEY = auth_manager.create_api_key(TEST_USER)
    logger.info(f"[Security] 시스템 보호 활성화. 생성된 API Key: {TEST_API_KEY}")


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


# --- 싱글톤 리소스 관리 (동시성 제어 강화) ---
class RAGResourceManager:
    _llm = None
    _embedder = None
    _lock = asyncio.Lock()  # 모델 로드 시 레이스 컨디션 방지

    @classmethod
    async def get_llm(cls):
        if cls._llm is None:
            async with cls._lock:
                if cls._llm is None:  # Double-check pattern
                    cls._llm = load_llm(DEFAULT_OLLAMA_MODEL)
        return cls._llm

    @classmethod
    async def get_embedder(cls):
        if cls._embedder is None:
            async with cls._lock:
                if cls._embedder is None:
                    cls._embedder = load_embedding_model(AVAILABLE_EMBEDDING_MODELS[0])
        return cls._embedder


# --- 세션 격리 의존성 ---
async def get_session_context(x_session_id: str | None = Header(None)) -> str:
    """헤더에서 세션 ID를 추출하고 컨텍스트를 고정합니다."""
    sid = x_session_id or "default"
    SessionManager.set_session_id(sid)
    SessionManager.init_session()
    return sid


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
    user_id: str = Depends(verify_token),
):
    """
    인증된 사용자의 PDF 문서를 업로드하고 해당 세션에 인덱싱합니다.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # Form 데이터로부터 세션 설정
    SessionManager.set_session_id(session_id)
    SessionManager.init_session()

    try:
        # 임시 파일 저장
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            embedder = await RAGResourceManager.get_embedder()

            # [수정] 무거운 동기 파이프라인 구축 작업을 별도 스레드에서 실행하여 이벤트 루프 차단 방지
            msg, cache_used = await asyncio.to_thread(
                build_rag_pipeline,
                uploaded_file_name=file.filename,
                file_path=tmp_path,
                embedder=embedder,
            )

            SessionManager.set("last_uploaded_file_name", file.filename)
            logger.info(
                f"[API] 문서 인덱싱 완료: {file.filename} (Session: {session_id}, Cache: {cache_used})"
            )

            return {
                "message": msg,
                "filename": file.filename,
                "session_id": session_id,
                "cache_used": cache_used,
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        logger.error(f"업로드 오류 (Session: {session_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest, user_id: str = Depends(verify_token)):
    """
    인증된 세션 컨텍스트에서 질의를 수행합니다.
    """
    sid = request.session_id or "default"
    SessionManager.set_session_id(sid)
    SessionManager.init_session()

    if SessionManager.get("last_uploaded_file_name") is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    start_time = time.time()
    try:
        llm = await RAGResourceManager.get_llm()
        rag_app = SessionManager.get("rag_engine")

        if rag_app is None:
            raise HTTPException(
                status_code=500, detail="QA 시스템이 초기화되지 않았습니다."
            )

        # LangGraph 실행 설정에 세션 ID 명시적 바인딩
        config = {"configurable": {"llm": llm, "session_id": sid, "thread_id": sid}}

        result = await rag_app.ainvoke({"input": request.query}, config=config)
        execution_time = (time.time() - start_time) * 1000

        sources = []
        for doc in result.get("documents", []):
            sources.append(
                {
                    "page": doc.metadata.get("page"),
                    "content": doc.page_content[:200] + "...",
                }
            )

        return QueryResponse(
            answer=result["response"], sources=sources, execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"질의 오류 (Session: {sid}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/stream_query")
async def stream_query_rag(request: QueryRequest, user_id: str = Depends(verify_token)):
    """
    인증된 세션에 대해 실시간 스트리밍(SSE) 응답을 제공합니다.
    """
    sid = request.session_id or "default"
    SessionManager.set_session_id(sid)
    SessionManager.init_session()

    if SessionManager.get("last_uploaded_file_name") is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    rag_app = SessionManager.get("rag_engine")
    if rag_app is None:
        raise HTTPException(
            status_code=500, detail="QA 시스템이 초기화되지 않았습니다."
        )

    async def event_generator():
        # [강화] 세션 컨텍스트 강제 재확립
        SessionManager.set_session_id(sid)
        SessionManager.init_session()

        logger.debug(f"[API] Streaming started for session: {sid}")

        llm = await RAGResourceManager.get_llm()
        # 스트리밍 시에도 명시적 세션 바인딩
        run_config = {"configurable": {"llm": llm, "session_id": sid, "thread_id": sid}}

        handler = get_streaming_handler()
        controller = get_adaptive_controller()
        sse_handler = ServerSentEventsHandler()

        try:
            async for chunk in handler.stream_graph_events(
                rag_app.astream_events(
                    {"input": request.query}, config=run_config, version="v2"
                ),
                adaptive_controller=controller,
            ):
                # 1. 메시지(답변 및 사고 과정) 처리
                if chunk.content:
                    yield sse_handler.format_sse_event(
                        "message", {"content": chunk.content}
                    )

                if chunk.thought:
                    yield sse_handler.format_sse_event(
                        "thought", {"content": chunk.thought}
                    )

                # 2. 메타데이터(문서) 처리
                if chunk.metadata and "documents" in chunk.metadata:
                    docs = [
                        {
                            "page": d.metadata.get("page"),
                            "content": d.page_content[:100],
                        }
                        for d in chunk.metadata["documents"]
                    ]
                    yield sse_handler.format_sse_event("sources", {"documents": docs})

            yield sse_handler.format_sse_event("end", {"status": "done"})
        except Exception as e:
            logger.error(f"Streaming error (Session: {sid}): {e}")
            yield sse_handler.format_sse_error(str(e))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.on_event("startup")
async def startup_event():
    """서버 시작 시 백그라운드 태스크 설정"""

    async def session_cleaner():
        while True:
            await asyncio.sleep(600)  # 10분마다 실행
            SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)

    asyncio.create_task(session_cleaner())
    logger.info("[API] 세션 자동 정리 태스크 시작됨 (주기: 10분)")


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
