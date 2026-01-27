"""
FastAPI 기반 RAG 시스템 백엔드 서버
UI와 독립적으로 RAG 기능을 외부 API로 제공합니다.
"""

import os
import time
import logging
import tempfile
from typing import List, Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from common.config import OLLAMA_MODEL_NAME, AVAILABLE_EMBEDDING_MODELS
from core.model_loader import load_llm, load_embedding_model
from core.rag_core import build_rag_pipeline, _load_and_build_retrieval_components, _create_ensemble_retriever
from core.graph_builder import build_graph
from api.schemas import QueryRequest, QueryResponse, SearchResponse, SearchResult
from core.session import SessionManager

# 로깅 설정
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG System API",
    description="Ollama와 LangGraph 기반의 고도화된 RAG 시스템 API",
    version="2.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 싱글톤 리소스 관리 (모델 캐싱용) ---
class RAGResourceManager:
    _llm = None
    _embedder = None

    @classmethod
    def get_llm(cls):
        if cls._llm is None:
            cls._llm = load_llm(OLLAMA_MODEL_NAME)
        return cls._llm

    @classmethod
    def get_embedder(cls):
        if cls._embedder is None:
            cls._embedder = load_embedding_model(AVAILABLE_EMBEDDING_MODELS[0])
        return cls._embedder

# --- Endpoints ---

@app.get("/api/v1/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "model": OLLAMA_MODEL_NAME
    }

@app.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("default")
):
    """
    PDF 문서를 업로드하고 인덱싱합니다.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 세션 ID 설정
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
            embedder = RAGResourceManager.get_embedder()
            
            # RAG 파이프라인 구축 (인덱싱 포함)
            msg, cache_used = build_rag_pipeline(
                uploaded_file_name=file.filename,
                file_path=tmp_path,
                embedder=embedder
            )

            SessionManager.set("last_uploaded_file_name", file.filename)
            logger.info(f"[System] [API] 문서 업로드 및 인덱싱 완료: {file.filename} (Session: {session_id}, 캐시 사용: {cache_used})")
            
            return {
                "message": msg,
                "filename": file.filename,
                "session_id": session_id,
                "cache_used": cache_used
            }
        finally:
            # 작업 완료 후 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                logger.debug(f"임시 파일 삭제 완료: {tmp_path}")

    except Exception as e:
        logger.error(f"업로드 중 오류 (Session: {session_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """
    질문에 대해 RAG 파이프라인을 실행하여 답변을 생성합니다.
    """
    SessionManager.set_session_id(request.session_id)
    SessionManager.init_session()

    if SessionManager.get("last_uploaded_file_name") is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    start_time = time.time()
    try:
        llm = RAGResourceManager.get_llm()
        rag_app = SessionManager.get("rag_engine")
        
        if rag_app is None:
             raise HTTPException(status_code=500, detail="QA 시스템이 초기화되지 않았습니다. 문서를 먼저 업로드하세요.")

        config = {"configurable": {"llm": llm}}
        result = await rag_app.ainvoke({"input": request.query}, config=config)
        
        execution_time = (time.time() - start_time) * 1000
        
        # 출처 정보 추출
        sources = []
        for doc in result.get("documents", []):
            sources.append({
                "page": doc.metadata.get("page"),
                "content": doc.page_content[:200] + "..."
            })

        return QueryResponse(
            answer=result["response"],
            sources=sources,
            execution_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"질의 중 오류 (Session: {request.session_id}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/stream_query")
async def stream_query_rag(request: QueryRequest):
    """
    질문에 대한 답변을 실시간 스트리밍(SSE)으로 반환합니다.
    """
    SessionManager.set_session_id(request.session_id)
    SessionManager.init_session()

    if SessionManager.get("last_uploaded_file_name") is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    rag_app = SessionManager.get("rag_engine")
    if rag_app is None:
        raise HTTPException(status_code=500, detail="QA 시스템이 초기화되지 않았습니다.")

    async def event_generator():
        llm = RAGResourceManager.get_llm()
        run_config = {"configurable": {"llm": llm}}
        
        try:
            async for event in rag_app.astream_events(
                {"input": request.query}, 
                config=run_config, 
                version="v1"
            ):
                kind = event["event"]
                name = event.get("name", "Unknown")
                data = event.get("data", {})

                # 1. 텍스트 청크 스트리밍
                chunk_text = None
                if kind == "on_parser_stream":
                    chunk_text = data.get("chunk")
                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if hasattr(chunk, "content"):
                        chunk_text = chunk.content
                elif kind == "on_custom_event" and name == "response_chunk":
                    chunk_text = data.get("chunk")

                if chunk_text:
                    yield f"event: message\ndata: {chunk_text}\n\n"

                # 2. 문서 정보 전송 (retrieve 완료 시)
                if kind == "on_chain_end" and name == "retrieve":
                    output = data.get("output")
                    if output and "documents" in output:
                        docs = [
                            {"page": d.metadata.get("page"), "content": d.page_content[:100]} 
                            for d in output["documents"]
                        ]
                        import json
                        yield f"event: sources\ndata: {json.dumps(docs)}\n\n"

            yield "event: end\ndata: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming API error (Session: {request.session_id}): {e}")
            yield f"event: error\ndata: {str(e)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)