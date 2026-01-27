import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.append(str(Path(__file__).parent.parent / "src"))

def test_session_manager_fallback():
    print("Testing SessionManager fallback (No Streamlit context)...")
    from core.session import SessionManager
    
    # 1. 초기화 확인
    SessionManager.init_session()
    print("✓ init_session() passed.")
    
    # 2. 값 저장 및 조회
    SessionManager.set("test_key", "hello_api")
    val = SessionManager.get("test_key")
    if val == "hello_api":
        print(f"✓ set/get passed: {val}")
    else:
        print(f"✗ set/get failed: {val}")
        return False

    # 3. 로그 추가
    SessionManager.add_status_log("API starting")
    logs = SessionManager.get("status_logs")
    if "API starting" in logs:
        print(f"✓ add_status_log passed: {logs}")
    else:
        print(f"✗ add_status_log failed: {logs}")
        return False
        
    print("✅ SessionManager Fallback Test PASSED.\n")
    return True

def test_rag_core_progress_callback():
    print("Testing rag_core progress callback (NameError fix)...")
    from core.rag_core import _load_and_build_retrieval_components
    from unittest.mock import MagicMock
    
    # 모의 객체 설정
    mock_embedder = MagicMock()
    mock_embedder.model_name = "test-model"
    
    # 실제 빌드를 수행하지 않고 함수 호출 구조만 테스트하기 위해 
    # _load_pdf_docs 등을 모킹할 수 있으나, 여기서는 NameError 발생 여부가 핵심.
    # build_rag_pipeline 내부에서 on_progress를 전달하는 로직 확인.
    
    progress_called = False
    def my_progress():
        nonlocal progress_called
        progress_called = True
        
    # 실제 PDF 로딩은 생략하고 progress 변수 전달이 에러를 유발하는지만 체크 (간접 검증)
    print("Checking function signature...")
    import inspect
    sig = inspect.signature(_load_and_build_retrieval_components)
    if "on_progress" in sig.parameters:
        print("✓ 'on_progress' parameter exists in signature.")
    else:
        print("✗ 'on_progress' parameter MISSING in signature.")
        return False

    print("✅ Progress Callback Signature Test PASSED.\n")
    return True

if __name__ == "__main__":
    s1 = test_session_manager_fallback()
    s2 = test_rag_core_progress_callback()
    
    if s1 and s2:
        print("🚀 ALL TARGETED TESTS PASSED!")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED.")
        sys.exit(1)
