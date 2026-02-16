# CI 검증 자동화 스크립트 (Windows PowerShell용)
Write-Host "🚀 CI 파이프라인 시뮬레이션을 시작합니다..." -ForegroundColor Cyan

# 1. 환경 설정
$env:PYTHONPATH += ";$(Get-Location)\src"
$env:IS_CI_TEST = "true"

# 2. Ruff (Lint & Format)
Write-Host "`n[1/4] Ruff 체크 중..." -ForegroundColor Yellow
ruff check . --fix
ruff format .
if ($LASTEXITCODE -ne 0) { Write-Error "Ruff 검사 실패!"; exit 1 }

# 3. Mypy (Type Check)
Write-Host "`n[2/4] Mypy 타입 체크 중..." -ForegroundColor Yellow
mypy --ignore-missing-imports --disable-error-code import-untyped src/
if ($LASTEXITCODE -ne 0) { Write-Error "Mypy 검사 실패!"; exit 1 }

# 4. Pytest (Integrity & Unit)
Write-Host "`n[3/4] 통합 및 단위 테스트 실행 중..." -ForegroundColor Yellow
python -m pytest tests/integration/test_rag_integration.py tests/integration/test_caching_system.py tests/unit/ --disable-warnings
if ($LASTEXITCODE -ne 0) { Write-Error "테스트 실패!"; exit 1 }

# 5. OpenAPI (Doc Gen)
Write-Host "`n[4/4] OpenAPI 스펙 추출 확인 중..." -ForegroundColor Yellow
python scripts/export_openapi.py > $null
if ($LASTEXITCODE -ne 0) { Write-Error "OpenAPI 추출 실패!"; exit 1 }

Write-Host "`n✅ 모든 검증을 통과했습니다! 이제 안심하고 푸시하세요." -ForegroundColor Green
