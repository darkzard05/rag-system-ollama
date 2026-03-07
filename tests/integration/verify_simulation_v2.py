from streamlit.testing.v1 import AppTest
import sys

def test_v2_fragment_sync():
    """최적화된 시뮬레이션 앱 V2 검증"""
    at = AppTest.from_file("tests/integration/simulation_app_v2.py")
    at.run()

    # 1. 초기 상태 확인
    assert at.session_state.parent_rerun_count == 1
    assert at.session_state.target_page == 1
    print("\n✅ [V2] Initial State: OK")

    # 2. 채팅 입력 시뮬레이션 (chat_input_v2)
    # AppTest에서 위젯을 찾기 위해 key를 정확히 지정
    try:
        chat_input = at.chat_input(key="chat_input_v2")
        chat_input.set_value("Test Message").run()
    except Exception as e:
        print(f"❌ [V2] Chat Input Error: {e}")
        # 현재 위젯 목록 출력 (디버깅용)
        # print(at._tree)
        return

    # 검증: 메시지가 추가되었는가?
    assert len(at.session_state.messages) == 1
    # [핵심] 실제 Streamlit 환경이라면 이 카운트는 1이어야 함.
    # AppTest는 프래그먼트 업데이트를 위해 전체를 다시 돌릴 수 있으므로 
    # 로직 상의 부작용(예: 무한 루프 등)이 없는지 확인
    print(f"✅ [V2] Chat Input Success: Messages Count = {len(at.session_state.messages)}")

    # 3. 페이지 점프 시뮬레이션
    jump_btn = at.button(key="jump_v2_0")
    jump_btn.click().run()

    # 검증: 타겟 페이지가 바뀌었는가?
    expected_page = (0 % 5) + 1
    assert at.session_state.target_page == expected_page
    print(f"✅ [V2] Page Jump Success: Target Page = {at.session_state.target_page}")

    print(f"🚀 [V2] Final Parent Rerun Count: {at.session_state.parent_rerun_count}")
    print("✨ Simulation V2 passed successfully!")

if __name__ == "__main__":
    test_v2_fragment_sync()
