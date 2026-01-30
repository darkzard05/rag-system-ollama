"""
FastAPI 애플리케이션에서 OpenAPI 스펙(openapi.json)을 자동으로 추출하는 스크립트입니다.
CI/CD 파이프라인에서 문서 자동화에 사용됩니다.
"""

import json
import sys
from pathlib import Path

# src 디렉토리를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

from api.api_server import app


def export_openapi():
    """OpenAPI 스펙을 JSON 형식으로 표준 출력에 기록합니다."""
    openapi_schema = app.openapi()
    print(json.dumps(openapi_schema, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    export_openapi()
