"""
채팅/세션 라우트 모듈.

- POST /api/v1/query
- POST /api/v1/stream_query (SSE)
- DELETE /api/v1/session/{session_id}

공통 헬퍼는 api/_deps 를 통해 공유합니다. 테스트가 api_server 모듈
네임스페이스에서 패치하는 RAGSystem/SessionManager 는 ``_app_server_module()``
으로 호출 시점에 읽습니다(모듈 수준 import 시 순환 참조 + 패치 미적용 문제 회피).
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api._deps import (
    MAX_QUERY_LENGTH,
    _app_server_module,
    _doc_to_source,
    _owners_lock,
    _require_session_owner,
    _session_owners,
    _validate_session_id,
    get_session_context,
    verify_token,
)
from api.schemas import QueryRequest, QueryResponse
from api.streaming_handler import (
    ServerSentEventsHandler,
    get_adaptive_controller,
    get_streaming_handler,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/v1/query", response_model=QueryResponse)
async def query_rag(
    request: QueryRequest,
    user_id: str = Depends(verify_token),
    _session_ctx: str = Depends(get_session_context),
):
    """
    인증된 세션 컨텍스트에서 질의를 수행합니다.
    """
    srv = _app_server_module()
    sid = request.session_id or "default"
    _validate_session_id(sid)
    if len(request.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"질문은 {MAX_QUERY_LENGTH}자 이하여야 합니다."
        )
    srv.SessionManager.init_session(session_id=sid)
    _require_session_owner(sid, user_id)

    if srv.SessionManager.get("last_uploaded_file_name", session_id=sid) is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    start_time = time.time()
    try:
        # [개선] RAGSystem 클래스를 통해 통합된 인터페이스 호출 (설정 및 리소스 관리 자동화)
        rag_sys = srv.RAGSystem(session_id=sid)
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


@router.post("/api/v1/stream_query")
async def stream_query_rag(
    request: QueryRequest,
    fastapi_request: Request,
    user_id: str = Depends(verify_token),
    _session_ctx: str = Depends(get_session_context),
):
    """
    인증된 세션에 대해 실시간 스트리밍(SSE) 응답을 제공합니다.
    """
    srv = _app_server_module()
    sid = request.session_id or "default"
    _validate_session_id(sid)
    if len(request.query) > MAX_QUERY_LENGTH:
        raise HTTPException(
            status_code=400, detail=f"질문은 {MAX_QUERY_LENGTH}자 이하여야 합니다."
        )
    srv.SessionManager.init_session(session_id=sid)
    _require_session_owner(sid, user_id)

    file_name = srv.SessionManager.get("last_uploaded_file_name", session_id=sid)
    logger.debug(f"[TEST] Session ID: {sid}, last_uploaded_file_name: {file_name}")

    if file_name is None:
        raise HTTPException(status_code=400, detail="먼저 문서를 업로드해주세요.")

    rag_app = srv.SessionManager.get("rag_engine", session_id=sid)
    if rag_app is None:
        raise HTTPException(
            status_code=500, detail="QA 시스템이 초기화되지 않았습니다."
        )

    async def event_generator():
        logger.debug(f"[API] Streaming started for session: {sid}")

        # [개선] RAGSystem 클래스를 통해 통합된 인터페이스 호출 (설정 및 리소스 관리 자동화)
        rag_sys = srv.RAGSystem(session_id=sid)

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
                    srv.SessionManager.add_status_log(chunk.status, session_id=sid)

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
        # fail-closed for the stream: any unexpected exception becomes a
        # user-visible SSE [error] event instead of breaking the response body
        # mid-stream. Broad by design; exc_info=True keeps it diagnosable.
        except Exception as e:
            logger.error(f"Streaming error (Session: {sid}): {e}", exc_info=True)
            yield sse_handler.format_sse_error(str(e))
            # 클라이언트가 [error] 후 [end]를 받아야 연결을 해제(블로킹 해제)한다.
            yield sse_handler.format_sse_event("end", {"status": "done"}, event_counter)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 버퍼링 방지
        },
    )


@router.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str, user_id: str = Depends(verify_token)):
    """특정 세션의 데이터를 삭제하고 메모리를 해제합니다. (소유자만 가능)"""
    srv = _app_server_module()
    _validate_session_id(session_id)
    _require_session_owner(session_id, user_id)
    success = srv.SessionManager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
    with _owners_lock:
        _session_owners.pop(session_id, None)
    return {"message": f"Session {session_id} deleted successfully"}
