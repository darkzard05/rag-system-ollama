"""
D20: CORS allow_origins 설정 검증.

- 기본값 [] = 모든 브라우저 교차-오리진 차단 (API는 키 인증 + 로컬 사용).
- cors.allow_origins에 명시적으로 등록된 오리진만 허용된다.

api_server.app 은 모듈 import 시점에 CORS_ALLOW_ORIGINS를 읽어 미들웨어를
배선하므로, "설정된 오리진" 케이스는 import 시점에 고정된 실제 앱 대신
config 파싱 시점(common.config._parse_allow_origins)에서 검증한다.
"""

import pytest
import src.api.api_server as srv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

import common.config
from common.config import CORS_ALLOW_ORIGINS, _parse_allow_origins


def _cors_middleware(app):
    """app.user_middleware 에서 CORSMiddleware 항목을 찾아 반환한다."""
    return next(m for m in app.user_middleware if m.cls is CORSMiddleware)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("http://x", ["http://x"]),
        (["a", "b"], ["a", "b"]),
        (None, []),
        ([], []),
    ],
)
def test_cors_parse_handles_string_and_list(raw, expected):
    """cors.allow_origins 파싱: str→[str], list→list[str], None/[]→[]."""
    assert _parse_allow_origins(raw) == expected


def test_cors_parse_filters_blank_entries():
    """빈 문자열/None 항목은 허용 목록에서 제외된다."""
    assert _parse_allow_origins(["http://a", "", None]) == ["http://a"]


def test_cors_default_empty_blocks_cross_origin():
    """기본 설정: 허용 목록이 비어 있어 실제 앱의 브라우저 교차-오리진이 차단된다."""
    assert CORS_ALLOW_ORIGINS == []
    mw = _cors_middleware(srv.app)
    assert mw.kwargs["allow_origins"] == []
    assert mw.kwargs["allow_credentials"] is True

    client = TestClient(srv.app)
    resp = client.options(
        "/api/v1/health",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


def test_cors_allowed_origins_from_config(monkeypatch):
    """구성된 오리진이 파싱·배선된다 (실제 앱 미들웨어는 import 시점 값 사용)."""
    configured = ["http://localhost:8501"]
    monkeypatch.setattr(common.config, "CORS_ALLOW_ORIGINS", configured)
    assert _parse_allow_origins(common.config.CORS_ALLOW_ORIGINS) == configured

    # 배선 불변식: 실제 앱 미들웨어 파라미터는 import 시점 CORS_ALLOW_ORIGINS와 동일.
    mw = _cors_middleware(srv.app)
    assert mw.kwargs["allow_origins"] == list(CORS_ALLOW_ORIGINS)
    assert mw.kwargs["allow_credentials"] is True
