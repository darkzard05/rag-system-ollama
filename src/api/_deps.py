"""
API 레이어 공통 의존성/헬퍼 모듈.

api_server / routes_chat / routes_docs 가 공유하는 인증·소유권·세션 컨텍스트
헬퍼와 전역 상태를 보유합니다. 라우트 모듈이 api_server 를 모듈 수준에서
import 하면 순환 참조가 되므로, api_server 의 네임스페이스 바인딩을 읽어야 하는
경우(테스트의 ``patch("src.api.api_server.X")`` 호환 포함)는
``_app_server_module()`` 지연 조회를 사용합니다.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.document_processor import compute_file_hash
from security.auth_system import AuthenticationManager

logger = logging.getLogger(__name__)

# --- 보안 및 인증 설정 (api_server/login/admin 과 공유되는 단일 인스턴스) ---
auth_scheme = HTTPBearer()
auth_manager = AuthenticationManager()

# --- 세션/파일 소유권 레지스트리 ---
# session_id/file_hash -> (소유 user_id, 바인딩 시각 epoch).
# 값 타입: 바인딩 시각을 포함한 tuple(만료 항목 스윕용).
_OWNER_TTL_SECONDS = 7 * 24 * 3600
# PDF 라이브러리 보존 기간: 세션 정리와 분리되어 오래된 업로드 파일을 수거
_PDF_RETENTION_DAYS = 30

# 입력 경계 (MAX_PDF_SIZE_BYTES 는 api_server 에 정의 — 테스트가 해당 네임스페이스를 패치)
MAX_QUERY_LENGTH = 4000
MAX_SESSION_ID_LENGTH = 64

_session_owners: dict[str, tuple[str, float]] = {}
_file_owners: dict[str, tuple[str, float]] = {}
_owners_lock = threading.RLock()

# 프로세스 단위 부트스트랩 게이트.
# api_server 는 진입 경로에 따라 ``api.api_server`` / ``src.api.api_server``
# 두 모듈 객체로 중복 로드될 수 있다. auth_manager 는 본 모듈 단일 인스턴스를
# 공유하므로, 두 번째 로드의 부트스트랩이 admin 크리덴셜을 덮어쓰지 않도록
# 첫 부트스트랩 결과를 캐시한다. (테스트가 직접 _bootstrap_credentials 를
# 호출하는 경로에는 영향을 주지 않는다.)
_bootstrap_result: tuple[str, str] | None = None


def run_bootstrap_once(_factory: Any) -> tuple[str, str]:
    """모듈 레벨 자격증명 부트스트랩을 프로세스당 1회만 수행합니다."""
    global _bootstrap_result
    if _bootstrap_result is None:
        _bootstrap_result = _factory()
    assert _bootstrap_result is not None
    return _bootstrap_result


def _app_server_module() -> Any:
    """실행 중인 ``api.api_server`` 모듈 객체를 반환합니다.

    진입 경로에 따라 api_server 는 ``api.api_server``(PYTHONPATH=src 실행) 또는
    ``src.api.api_server``(pytest 루트 실행) 로 로드됩니다. sys.modules 에서
    이미 로드된 단일 인스턴스를 찾아 반환하므로 모듈 복제(전역 상태 이중화)를
    막고, 테스트가 ``src.api.api_server.<이름>`` 을 패치한 경우에도 라우트의
    호출 시점 조회가 동일 객체를 보게 되어 패치가 적용됩니다.
    """
    for _name in ("src.api.api_server", "api.api_server"):
        _mod = sys.modules.get(_name)
        if _mod is not None:
            return _mod
    # (방어적 폴백 — 라우트 호출 시점에는 항상 로드 완료 상태)
    import api.api_server

    return api.api_server


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
        storage = Path(_app_server_module().PDF_STORAGE_DIR)
        for file_hash, entry in list(_file_owners.items()):
            stale = now - entry[1] > _OWNER_TTL_SECONDS
            if stale or not (storage / f"{file_hash}.pdf").exists():
                del _file_owners[file_hash]


def _sweep_expired_library_files() -> None:
    """보존 기간(_PDF_RETENTION_DAYS)을 초과한 PDF 라이브러리 파일과 그 소유권 항목을 제거합니다."""
    storage = Path(_app_server_module().PDF_STORAGE_DIR)
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


# --- PDF 서빙 및 참조 좌표 노출 ---
def _resolve_pdf_path(file_hash: str) -> Path | None:
    """스토리지 루트 내에서 file_hash와 일치하는 PDF 경로를 반환합니다.

    경로 주입을 방지하기 위해 file_hash를 파일명 안전 문자열로 정규화하고,
    최종 후보 경로가 항상 스토리지 루트 내부에 위치하는지 검증합니다.
    """
    storage = Path(_app_server_module().PDF_STORAGE_DIR)
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
    for key in (
        "pages",
        "page_coords",
        "word_coords",
        "file_hash",
        "coord_cache_error",
    ):
        if metadata.get(key) is not None:
            source[key] = metadata[key]
    return source
