"""
프로젝트의 전체 무결성을 로컬에서 검증하는 자동화 스크립트입니다.
정적 분석, 단위 테스트, 통합 테스트, UI 테스트 및 실전 RAG 파이프라인을 순차적으로 실행합니다.
"""

import os
import sys
import subprocess
import time
import shutil
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 경로 설정
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(str(ROOT_DIR / "src"))

# 출력 색상 정의 (Windows PowerShell 호환)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"""\n{Colors.HEADER}{Colors.BOLD}{'='*20} {text} {'='*20}{Colors.ENDC}""")

def run_command(command: list[str], name: str, timeout: int = 300) -> bool:
    """명령어를 실행하고 결과를 반환합니다."""
    print(f"{Colors.CYAN}[RUNNING]{Colors.ENDC} {name}...")
    start_time = time.time()
    
    try:
        # shell=True는 Windows에서 명령행 인자 전달을 위해 필요할 수 있음
        result = subprocess.run(
            command,
            cwd=str(ROOT_DIR),
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"{Colors.GREEN}[SUCCESS]{Colors.ENDC} {name} ({elapsed:.1f}s)")
            return True
        else:
            print(f"{Colors.FAIL}[FAILED]{Colors.ENDC} {name} ({elapsed:.1f}s)")
            if result.stdout:
                print(f"""{Colors.WARNING}--- STDOUT ---{Colors.ENDC}\n{result.stdout[:500]}...""")
            if result.stderr:
                print(f"""{Colors.FAIL}--- STDERR ---{Colors.ENDC}\n{result.stderr[:1000]}""")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"{Colors.FAIL}[TIMEOUT]{Colors.ENDC} {name} (Limit: {timeout}s)")
        return False
    except Exception as e:
        print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} {name}: {e}")
        return False

def check_ollama() -> bool:
    """Ollama 서버 상태를 점검합니다."""
    import requests
    from common.config import OLLAMA_BASE_URL
    
    print(f"{Colors.CYAN}[CHECK]{Colors.ENDC} Ollama Server Status ({OLLAMA_BASE_URL})...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            print(f"{Colors.GREEN}[READY]{Colors.ENDC} Ollama is running. Models: {', '.join(models[:5])}...")
            return True
        else:
            print(f"{Colors.FAIL}[ERROR]{Colors.ENDC} Ollama returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"{Colors.WARNING}[WARNING]{Colors.ENDC} Ollama server not reachable: {e}")
        print(f"{Colors.WARNING}실전 파이프라인 테스트(Quick Verify)는 실패할 수 있습니다.{Colors.ENDC}")
        return False

def main():
    start_all = time.time()
    results = {}
    
    print_header("RAG System Integrity Verification")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Root: {ROOT_DIR}")
    
    # 0. 환경 점검
    results["Environment: Ollama"] = check_ollama()
    
    # 1. 정적 분석 (Static Analysis)
    print_header("Step 1: Static Analysis")
    results["Static: Ruff Check"] = run_command(["ruff", "check", "."], "Ruff Linter")
    
    # mypy 실행 (pyproject.toml 설정을 따르며, 외부 라이브러리는 skip함)
    os.environ["PYTHONPATH"] = str(ROOT_DIR / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
    results["Static: Mypy Typing"] = run_command(
        ["mypy", "src"], 
        "Mypy Type Checker",
        timeout=120  # 속도 최적화(skip) 적용으로 2분이면 충분함
    )
    
    # 2. 단위 및 통합 테스트 (Pytest)
    print_header("Step 2: Core & Integration Tests")
    # 고속 단위 테스트
    results["Test: Unit Tests"] = run_command(["pytest", "tests/unit"], "Unit Tests")
    # UI 통합 테스트
    results["Test: UI Flow"] = run_command(["pytest", "tests/integration/test_streamlit_app.py"], "Streamlit UI Test")
    # 스트리밍 프로토콜 테스트
    results["Test: Streaming"] = run_command(["pytest", "tests/integration/test_streaming_response.py"], "Streaming Protocol Test")
    # 보안 및 캐시 시스템 테스트 (CI 동기화)
    results["Test: Security"] = run_command(["pytest", "tests/security/test_cache_security.py"], "Cache Security Test")
    results["Test: Caching"] = run_command(["pytest", "tests/integration/test_caching_system.py"], "Caching System Test")
    # RAG 시스템 통합 테스트
    results["Test: RAG Core"] = run_command(["pytest", "tests/integration/test_rag_integration.py"], "RAG Core Integration")
    
    # 3. 실전 파이프라인 테스트 (E2E)
    print_header("Step 3: End-to-End Verification")
    if results.get("Environment: Ollama"):
        results["E2E: Quick Verify"] = run_command(["python", "scripts/quick_verify_rag.py"], "RAG Pipeline E2E", timeout=600)
    else:
        print(f"{Colors.WARNING}[SKIPPED]{Colors.ENDC} Quick Verify skipped due to Ollama status")
        results["E2E: Quick Verify"] = False

    # 4. 문서 자동 업데이트 (Documentation)
    print_header("Step 4: Documentation Update")
    results["Docs: README Update"] = run_command(["python", "scripts/maintenance/update_readme.py"], "README Auto-update")

    # 5. 결과 요약 리포트
    print_header("Verification Summary")
    all_passed = True
    for task, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if passed else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        print(f"{task.ljust(30)}: {status}")
        if not passed:
            all_passed = False
            
    total_elapsed = time.time() - start_all
    print("-" * 50)
    if all_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CHECKS PASSED!{Colors.ENDC} (Total: {total_elapsed:.1f}s)")
        print(f"{Colors.BLUE}You are ready to commit/push. 🚀{Colors.ENDC}")
    else:
        print(f"{Colors.FAIL}{Colors.BOLD}❌ SOME CHECKS FAILED.{Colors.ENDC} (Total: {total_elapsed:.1f}s)")
        print(f"{Colors.WARNING}위의 FAIL 항목들을 먼저 수정해 주세요.{Colors.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
