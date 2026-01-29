import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath("src"))

from core.session import SessionManager
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)


def test_session_fallback():
    print("=== SessionManager Fallback Test ===")

    # 1. 세션 초기화 (Streamlit 없이)
    try:
        SessionManager.init_session()
        print("✅ SessionManager initialized successfully without Streamlit context.")
    except Exception as e:
        print(f"❌ SessionManager initialization failed: {e}")
        return

    # 2. 값 설정 및 가져오기 테스트
    SessionManager.set("test_key", "test_value")
    val = SessionManager.get("test_key")
    if val == "test_value":
        print(f"✅ Set/Get successful: {val}")
    else:
        print(f"❌ Set/Get failed: expected 'test_value', got '{val}'")

    # 3. 새로운 키 명칭(rag_engine) 확인
    SessionManager.set("rag_engine", "mock_engine")
    if SessionManager.get("rag_engine") == "mock_engine":
        print("✅ 'rag_engine' key works correctly.")
    else:
        print("❌ 'rag_engine' key test failed.")

    # 4. ready_for_chat 상태 확인 (기본값 False 예상)
    ready = SessionManager.is_ready_for_chat()
    print(f"ℹ️ is_ready_for_chat: {ready} (expected False without PDF processing)")


if __name__ == "__main__":
    test_session_fallback()
