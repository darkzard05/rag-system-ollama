import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.session import SessionManager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

def test_session_standalone():
    print("=== Standalone SessionManager Test ===")
    
    # 1. 초기화 확인
    SessionManager.init_session()
    print(f"Initialized: {SessionManager.get('_initialized')}")
    
    # 2. 값 설정 및 가져오기
    SessionManager.set("test_key", "hello_world")
    val = SessionManager.get("test_key")
    print(f"Set/Get Test: {val} (Expected: hello_world)")
    
    # 3. 상태 확인 (Streamlit 사용 여부)
    stats = SessionManager.get_stats()
    print(f"Stats: {stats}")
    
    # 4. 복물 설정 확인
    SessionManager.add_status_log("Test Log 1")
    SessionManager.add_status_log("Test Log 2")
    logs = SessionManager.get("status_logs")
    print(f"Logs: {logs}")
    
    assert val == "hello_world"
    assert not stats["using_streamlit"]
    print("\n✅ Standalone SessionManager test passed!")

if __name__ == "__main__":
    try:
        test_session_standalone()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
