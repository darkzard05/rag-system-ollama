
import os
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.session import SessionManager
from common.config import PARSING_CONFIG
from core.document_processor import load_pdf_docs, compute_file_hash

def test_session_id():
    print("[TEST] 세션 메시지 고유 ID 부여 확인...")
    SessionManager.reset_all_state()
    SessionManager.add_message("user", "안녕하세요")
    messages = SessionManager.get_messages()
    if messages and "msg_id" in messages[0]:
        print(f"✅ 성공: 메시지 ID '{messages[0]['msg_id']}' 확인")
    else:
        print("❌ 실패: 메시지에 msg_id가 없습니다.")

def test_parsing_config():
    print("[TEST] PARSING_CONFIG 업데이트 확인...")
    if "table_strategy" in PARSING_CONFIG and PARSING_CONFIG["table_strategy"] == "lines_strict":
        print(f"✅ 성공: table_strategy='{PARSING_CONFIG['table_strategy']}' 적용됨")
    else:
        print(f"❌ 실패: PARSING_CONFIG가 예상과 다릅니다: {PARSING_CONFIG}")
    
    if PARSING_CONFIG.get("extract_words") == True:
        print("✅ 성공: extract_words=True (하이라이트 기능 활성화)")
    else:
        print("❌ 실패: extract_words가 비활성화되어 있습니다.")

if __name__ == "__main__":
    test_session_id()
    test_parsing_config()
