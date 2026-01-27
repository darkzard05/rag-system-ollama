import os
import sys
import pytest
import time
from streamlit.testing.v1 import AppTest

# src 디렉토리를 경로에 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

def test_rag_full_cycle():
    """
    PDF 업로드 -> RAG 빌드 -> 질문 -> 답변 생성의 전체 사이클을 테스트합니다.
    """
    at = AppTest.from_file("src/main.py", default_timeout=300)
    
    print("🚀 앱 초기화 중...")
    at.run()
    
    # [디버깅] 현재 모든 위젯의 Key 출력
    print("\n🔍 가용 위젯 키 목록:")
    found_keys = []
    # AppTest는 내부적으로 위젯 맵을 관리함
    for attr in dir(at):
        try:
            val = getattr(at, attr)
            if hasattr(val, "key"):
                found_keys.append(f"{attr}(key={val.key})")
            elif isinstance(val, list) and len(val) > 0 and hasattr(val[0], "key"):
                found_keys.append(f"{attr}[0](key={val[0].key})")
        except:
            pass
    print(f"   {found_keys}")

    # 사이드바 내용 텍스트로 확인
    print("\n🔍 사이드바 내용:")
    for i, e in enumerate(at.sidebar):
        print(f"   [{i}] Type: {type(e).__name__}, Label: {getattr(e, 'label', 'N/A')}, Key: {getattr(e, 'key', 'N/A')}")

    # 1. 위젯 찾기 (Key로 직접 시도)
    print("\n🔍 위젯 탐색 중...")
    uploader = at.get("pdf_uploader")
    if isinstance(uploader, list) and len(uploader) > 0:
        uploader = uploader[0]
        
    if not uploader:
        # label로 찾기
        for e in at.sidebar:
            if "pdf" in str(getattr(e, "label", "")).lower():
                uploader = e
                break
                
    if not uploader:
        # expander 내부 탐색 (특수)
        for exp in at.sidebar.expander:
            for item in exp:
                if "pdf" in str(getattr(item, "label", "")).lower():
                    uploader = item
                    break
            if uploader: break

    if not uploader:
        pytest.fail("file_uploader 위젯을 찾을 수 없습니다. (위 로그를 확인하세요)")

    # 2. PDF 업로드
    pdf_path = os.path.join("tests", "2201.07520v1.pdf")
    print(f"📂 파일 업로드 중: {pdf_path}")
    with open(pdf_path, "rb") as f:
        uploader.upload(f).run()
    
    # 3. RAG 빌드 대기
    print("⚙️ 파일 업로드 완료, RAG 빌드 대기 중...")
    success = False
    for i in range(20):
        at.run()
        # chat_message는 비교적 잘 잡힘
        msgs = [m.content for m in at.chat_message]
        if any("문서 처리" in m or "캐시" in m for m in msgs):
            success = True
            print("✨ RAG 빌드 완료 확인")
            break
        print(f"   ⌛ 대기 중... ({i+1}/20)")
        time.sleep(3)
        
    if not success:
        pytest.fail("RAG 시스템 구축 실패")

    # 4. 질문 입력
    chat_input = at.get("chat_input_main")
    if isinstance(chat_input, list) and len(chat_input) > 0:
        chat_input = chat_input[0]
        
    if not chat_input:
        chat_input = at.chat_input[0] if at.chat_input else None
        
    if not chat_input:
        pytest.fail("chat_input 위젯을 찾을 수 없습니다.")
        
    chat_input.set_value("이 논문의 핵심 내용을 요약해줘.").submit().run()
    
    # 5. 답변 수신 대기
    print("⏳ 답변 수신 중...")
    final_answer = ""
    for i in range(40):
        at.run()
        assistant_msgs = [m.content for m in at.chat_message if m.role == "assistant"]
        if len(assistant_msgs) >= 2:
            current = assistant_msgs[-1]
            if len(current) > len(final_answer):
                final_answer = current
                print(f"   📥 수신 중... ({len(final_answer)} 자)")
            elif len(final_answer) > 50 and i > 10:
                break
        time.sleep(3)
        
    if not final_answer:
        pytest.fail("답변 수신 실패")
        
    print("\n" + "="*50)
    print("🤖 최종 답변:")
    print(final_answer)
    print("="*50)
    print("\n✨ E2E 테스트 성공!")

if __name__ == "__main__":
    test_rag_full_cycle()
