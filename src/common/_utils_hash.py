"""Domain utilities extracted from ``common.utils`` (Batch 5.4).
Log namespace preserved as ``common.utils`` for compatibility.
"""

import hashlib
import logging
from typing import Any

logger = logging.getLogger("common.utils")


def fast_hash(text: str, length: int = 16) -> str:
    """
    보안이 필요 없는 단순 식별용 고속 해시 함수.
    SHA256보다 훨씬 빠른 MD5를 사용하고 결과 길이를 조절합니다.
    """
    if not text:
        return "0" * length
    if isinstance(text, bytes):
        text = text.decode(errors="ignore")
    elif not isinstance(text, str):
        text = str(text)
    # usedforsecurity=False: 보안 진단 도구(Bandit 등)에 이 해시가
    # 암호화나 보안 목적으로 사용되지 않음을 알립니다.
    return hashlib.md5(text.encode(errors="ignore"), usedforsecurity=False).hexdigest()[
        :length
    ]


def doc_stable_id(
    doc: Any,
    *,
    page_content_key: str = "page_content",
) -> str:
    """문서의 안정 식별자(doc_id 메타 우선, 없으면 page_content 해시)를 반환합니다.

    ``Document`` 객체와 ``dict``(``{page_content, metadata}``) 를 모두 지원합니다.
    안정 키는 ``metadata.get("doc_id")`` 를 우선 사용하고, 없으면
    ``fast_hash(page_content)`` 로 계산합니다. 반환값은 항상 ``str`` 입니다.
    """
    if isinstance(doc, dict):
        meta = doc.get("metadata", {}) or {}
        content = doc.get(page_content_key, "") or ""
    else:
        meta = getattr(doc, "metadata", None) or {}
        content = getattr(doc, page_content_key, "") or ""
    doc_id = meta.get("doc_id")
    if doc_id is not None:
        return str(doc_id)
    return fast_hash(str(content))


def compute_file_hash(
    file_path: str,
    *,
    data: bytes | None = None,
    algorithm: str = "sha256",
) -> str:
    """파일 또는 바이트 데이터의 해시를 계산합니다. (R: Group 4 중복 통합)

    ``document_processor.compute_file_hash`` 와 ``cache_security`` 의
    ``compute_file_hash`` 를 단일 구현으로 통합한 공용 함수입니다.

    - ``data`` 가 주어지면 파일을 읽지 않고 해당 바이트를 해시합니다.
    - ``algorithm`` 은 ``hashlib.new`` 가 지원하는 알고리즘 이름입니다.
    - 두 기존 구현의 안전 계약을 모두 유지합니다:
      파일이 존재하지 않으면 ``FileNotFoundError`` 를, 그 외 I/O/해시 오류는
      ``""`` (빈 문자열) 을 반환합니다.
    """
    try:
        hasher = hashlib.new(algorithm)
        if data is not None:
            hasher.update(data)
            return hasher.hexdigest()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        raise
    except OSError as e:
        logger.error(f"파일 해시 계산 실패 ({file_path}): {e}")
        return ""
