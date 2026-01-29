"""
FastAPI 기반 RAG 시스템 백엔드 서버
UI와 독립적으로 RAG 기능을 외부 API로 제공합니다.
"""

import os
import time
import logging
import tempfile
from typing import Optional
from pathlib import Path

import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header
from fastapi.responses import StreamingResponse

from common.config import DEFAULT_OLLAMA_MODEL, AVAILABLE_EMBEDDING_MODELS
from core.model_loader import load_llm, load_embedding_model
from core.rag_core import build_rag_pipeline
from api.schemas import QueryRequest, QueryResponse
from core.session import SessionManager

# 로깅 설정
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Ollama와 LangGraph 기반의 고도화된 RAG 시스템 API",
    version="2.0.0",
)


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
async def get_session_context(x_session_id: Optional[str] = Header(None)) -> str:
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
    file: UploadFile = File(...), session_id: str = Form("default")
):
    """
    PDF 문서를 업로드하고 해당 세션에 인덱싱합니다.
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

            # RAG 파이프라인 구축 (인덱싱 포함)
            msg, cache_used = build_rag_pipeline(
                uploaded_file_name=file.filename, file_path=tmp_path, embedder=embedder
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
async def query_rag(request: QueryRequest):
    """
    격리된 세션 컨텍스트에서 질의를 수행합니다.
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
async def stream_query_rag(request: QueryRequest):
    """
    실시간 스트리밍(SSE) 응답 시 세션 격리를 보장합니다.
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
        llm = await RAGResourceManager.get_llm()
        # 스트리밍 시에도 명시적 세션 바인딩
        run_config = {"configurable": {"llm": llm, "session_id": sid, "thread_id": sid}}

        try:
            async for event in rag_app.astream_events(
                {"input": request.query}, config=run_config, version="v2"
            ):
                # (기존 스트리밍 로직 유지)
                kind = event["event"]
                name = event.get("name", "Unknown")
                data = event.get("data", {})

                if kind == "on_custom_event" and name == "response_chunk":
                    chunk_text = data.get("chunk")
                    if chunk_text:
                        yield f"event: message\ndata: {chunk_text}\n\n"

                if kind == "on_chain_end" and name == "retrieve":
                    output = data.get("output")
                    if output and "documents" in output:
                        docs = [
                            {
                                "page": d.metadata.get("page"),
                                "content": d.page_content[:100],
                            }
                            for d in output["documents"]
                        ]
                        import json

                        yield f"event: sources\ndata: {json.dumps(docs)}\n\n"

            yield "event: end\ndata: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Streaming error (Session: {sid}): {e}")
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    # 보안 권고에 따라 기본 호스트를 127.0.0.1로 설정
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port)
