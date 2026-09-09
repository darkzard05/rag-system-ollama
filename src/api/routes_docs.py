"""
문서 업로드/PDF 서빙 라우트 모듈.

- POST /api/v1/upload
- GET /api/v1/pdf/{file_hash}

공통 헬퍼는 api/_deps 를 통해 공유합니다. 테스트가 api_server 모듈
네임스페이스에서 패치하는 RAGSystem/SessionManager/PDF_STORAGE_DIR/
MAX_PDF_SIZE_BYTES 는 ``_app_server_module()`` 으로 호출 시점에 읽습니다
(모듈 수준 import 시 순환 참조 + 패치 미적용 문제 회피).
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from api._deps import (
    _app_server_module,
    _bind_file_owner,
    _bind_session_owner,
    _require_file_owner,
    _require_session_owner,
    _resolve_pdf_path,
    _validate_session_id,
    verify_token,
)
from common.config import DEFAULT_EMBEDDING_MODEL
from core.document_processor import compute_file_hash
from core.resource_manager import get_resource_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/v1/upload")
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    embedding_model: str | None = Form(None),
    user_id: str = Depends(verify_token),
):
    """
    인증된 사용자의 PDF 문서를 업로드하고 해당 세션에 인덱싱합니다.
    """
    srv = _app_server_module()
    _validate_session_id(session_id)

    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")

    # 다른 사용자가 소유한 세션으로의 업로드를 차단 (교차 사용자 내용 주입 방지)
    _require_session_owner(session_id, user_id)
    # [TOCTOU 방지] 소유권 바인딩을 핸들러 최상단으로 이동: 첫 업로드가 진행되는 동안
    # 다른 사용자의 동일 세션 업로드가 교차 주입되는 경합을 차단한다.
    _bind_session_owner(session_id, user_id)

    # 명시적으로 세션 초기화 (최적화: 직접 호출)
    srv.SessionManager.init_session(session_id=session_id)

    try:
        # PDF를 file_hash 기반 스토리지에 영구 보존 (PDF 서빙 + 좌표 하이드레이션용)
        # [입력 경계] 파일 크기 제한: 메모리 고갈 방지를 위해 청크 단위로 읽으며 상한을 초과하면 거부
        content = b""
        total = 0
        while chunk := await file.read(1024 * 1024):  # 1MB 청크
            total += len(chunk)
            if total > srv.MAX_PDF_SIZE_BYTES:
                raise HTTPException(
                    status_code=413, detail="파일 크기가 50MB를 초과합니다."
                )
            content += chunk
        file_hash = compute_file_hash("", data=content)
        # [TOCTOU 방지] 파일 소유권 바인딩을 파일 저장 직전으로 이동.
        _bind_file_owner(file_hash, user_id)
        storage = Path(srv.PDF_STORAGE_DIR)
        storage.mkdir(parents=True, exist_ok=True)
        persisted_path = storage / f"{file_hash}.pdf"
        persisted_path.write_bytes(content)

        # [수정] 동적으로 결정된 임베딩 모델 사용 (모델 로딩은 무거우므로 스레드 유지)
        embedder = await get_resource_manager().get_embedder_for_session(
            session_id, embedding_model
        )

        # [중요] RAGSystem 클래스를 통해 세션 격리 보장
        rag_sys = srv.RAGSystem(session_id=session_id)
        # [수정] 빌드 전체 동안 embedder 를 pin 하여 모델 축출 use-after-free 방지.
        # 명시적 model_name 으로 획득+pin (key_for_object 역산 의존 제거).
        async with get_resource_manager().use_embedder(
            model_name=embedding_model or DEFAULT_EMBEDDING_MODEL
        ):
            msg, cache_used = await rag_sys.build_pipeline(
                file_path=str(persisted_path),
                file_name=file.filename,
                embedder=embedder,
            )

        srv.SessionManager.set(
            "last_uploaded_file_name", file.filename, session_id=session_id
        )
        srv.SessionManager.set("file_hash", file_hash, session_id=session_id)
        srv.SessionManager.set(
            "pdf_file_path", str(persisted_path), session_id=session_id
        )
        srv.SessionManager.set(
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


@router.get("/api/v1/pdf/{file_hash}")
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
