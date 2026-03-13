
import sys
import os
import time
import logging

# 프로젝트 루트를 경로에 추가
sys.path.append(os.path.abspath("src"))

from core.session import SessionManager

# 로그 설정
logging.basicConfig(level=logging.INFO)

def test_session_cleanup():
    print("--- Session Cleanup Test Start ---")
    
    # 1. 세션 생성 및 데이터 설정
    sid1 = "test_session_active"
    sid2 = "test_session_expired"
    
    SessionManager.set("data", "active", session_id=sid1)
    SessionManager.set("data", "to_be_expired", session_id=sid2)
    
    print(f"Created two sessions: {sid1}, {sid2}")
    
    # 2. 만료 세션의 시간을 과거로 조작 (테스트를 위해)
    state2 = SessionManager._get_state(sid2)
    state2["last_accessed"] = time.time() - 5000 # 1시간 이상 전으로 설정
    
    print(f"Set {sid2} last_accessed to 5000 seconds ago.")
    
    # 3. 정리 수행 (TTL 3600초)
    print("Running cleanup...")
    SessionManager.cleanup_expired_sessions(max_idle_seconds=3600)
    
    # 4. 결과 확인
    active_exists = sid1 in SessionManager._fallback_sessions
    expired_exists = sid2 in SessionManager._fallback_sessions
    
    print(f"Active session exists: {active_exists} (Expected: True)")
    print(f"Expired session exists: {expired_exists} (Expected: False)")
    
    if active_exists and not expired_exists:
        print("✅ Success: Expired session was correctly removed.")
    else:
        print("❌ Failure: Session cleanup logic is not working as expected.")
        sys.exit(1)

if __name__ == "__main__":
    test_session_cleanup()
