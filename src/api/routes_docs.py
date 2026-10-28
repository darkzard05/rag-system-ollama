"""
문서 업로드/PDF 서빙 라우트 모듈.

- POST /api/v1/upload
- GET /api/v1/pdf/{file_hash}

공통 헬퍼는 api/_deps 를 통해 공유합니다. 테스트가 api_server 모듈
네임스페이스에서 패치하는 RAGSystem/SessionManager/PDF_STORAGE_DIR/
MAX_PDF_SIZE_BYTES 는 ``_app_server_module()`` 으로 호출 시점에 읽습니다
(모듈 수준 import 시 순환 참조 + 패치 미적용 문제 회피).
"""

import asyncio
import logging
import os
import threading
import time
import uuid
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

# PDF 매직바이트: 접미사(.pdf)만 검사하면 비-PDF 바이트가 스토리지에 영속화되어
# 빌드 경로로 전달된다. 저장·바인딩 전에 매직바이트를 강제한다.
_PDF_MAGIC = b"%PDF"

# Tier1.5: 페이지 수 상한. 50MB 바이트 상한만으로는 고압축·다페이지 폭탄을
# 막지 못해 파서·FAISS·좌표캐시가 OOM된다. xref만 읽는 가벼운 검사로
# 업로드 시점에 조기 거부한다 (fail loudly, 무음 잘림 금지).
MAX_PDF_PAGES = 2000


def _ensure_valid_pdf(content: bytes, filename: str | None) -> None:
    """업로드 바이트가 PDF 매직바이트로 시작하는지 검증한다. 아니면 400을 발생시킨다."""
    if not content or not content.lstrip().startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")


# 동일 file_hash 경로의 동시 쓰기 경합 방지용 프로세스 내 락 테이블.
# 내용-주소 저장소는 동일 해시 → 동일 경로이므로 per-path 락 + 원자적
# 교체(tmp + os.replace)로 찢긴 읽기(torn read)를 방지한다.
_persist_locks_guard = threading.Lock()
_persist_locks: dict[str, threading.Lock] = {}


def _persist_lock_for(key: str) -> threading.Lock:
    with _persist_locks_guard:
        lock = _persist_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _persist_locks[key] = lock
        return lock


def persist_pdf_atomically(persisted_path: Path, content: bytes) -> None:
    """PDF 바이트를 원자적으로 영속화한다.

    ``Path.write_bytes``는 제자리 절단+쓰기라서 동시 리더(빌드 파이프라인)가
    부분 바이트를 읽을 수 있다. 고유 임시파일에 쓴 뒤 ``os.replace``로
    원자 교체하면 리더는 항상 완전한 구버전 또는 완전한 신버전만 본다.
    """
    persisted_path.parent.mkdir(parents=True, exist_ok=True)
    with _persist_lock_for(str(persisted_path)):
        tmp_path = persisted_path.with_name(
            f"{persisted_path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            tmp_path.write_bytes(content)
            # Windows는 리더가 파일을 여는 순간 교체가 WinError 5로
            # 거부되므로 충분히 재시도한다 (POSIX는 첫 시도에 성공).
            # 프로덕션 리더(빌드 파이프라인)는 단발성 읽기라 경합이 낮고,
            # 테스트의 타이트 루프 리더에도 1초 재시도로 대응한다.
            last_error: OSError | None = None
            for _ in range(200):
                try:
                    os.replace(tmp_path, persisted_path)
                    last_error = None
                    break
                except OSError as e:
                    last_error = e
                    time.sleep(0.005)
            if last_error is not None:
                raise last_error
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass


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
        # [입력 검증] 저장·소유권 바인딩 전에 실제 PDF인지 검증한다.
        # 검증 실패 시 스토리지/레지스트리를 오염시키지 않고 400으로 거부한다.
        _ensure_valid_pdf(content, file.filename)
        # Tier1.5: 페이지 수 상한 검사. 파싱 불가(None)는 빌드 단계에서
        # 확정 판정하므로 여기서 거부하지 않고 통과시킨다.
        from common.pdf_utils import get_pdf_page_count_from_bytes

        page_count = get_pdf_page_count_from_bytes(content)
        if page_count is not None and page_count > MAX_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF 페이지 수가 상한({MAX_PDF_PAGES})을 초과합니다.",
            )
        file_hash = compute_file_hash("", data=content)
        # [TOCTOU 방지] 파일 소유권 바인딩을 파일 저장 직전으로 이동.
        _bind_file_owner(file_hash, user_id)
        storage = Path(srv.PDF_STORAGE_DIR)
        storage.mkdir(parents=True, exist_ok=True)
        persisted_path = storage / f"{file_hash}.pdf"
        # 이벤트 루프 블로킹 방지를 위해 worker 스레드로 오프로드
        await asyncio.to_thread(persist_pdf_atomically, persisted_path, content)

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
    # owner-first: 존재 여부(resolve 404)보다 소유권(403)을 먼저 검사한다.
    # resolve-first 순서는 유효 토큰 보유자에게 임의 file_hash의 존재 여부를
    # 404/403 차분으로 노출하는 오라클이 된다. 타인·미등록 파일은 존재 여부와
    # 무관하게 일률 403으로 응답해 열거를 차단한다.
    _require_file_owner(file_hash, user_id)
    pdf_path = _resolve_pdf_path(file_hash)
    if pdf_path is None:
        raise HTTPException(status_code=404, detail="PDF 문서를 찾을 수 없습니다.")
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
