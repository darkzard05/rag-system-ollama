from streamlit.testing.v1 import AppTest
import sys

def test_v3_execution():
    """V3 시뮬레이션 앱의 에러 없는 실행 검증"""
    print("\n🚀 [V3] Starting Verification...")
    at = AppTest.from_file("tests/integration/simulation_app_v3.py")
    
    # 1. 초기 런
    at.run()
    if at.exception:
        print(f"❌ Initial run failed: {at.exception[0].message}")
        return

    # 2. 채팅 입력 (프래그먼트 내부에서 st.rerun() 발생)
    at.chat_input(key="input_v3").set_value("Test Message").run()
    if at.exception:
        print(f"❌ Chat Input failed: {at.exception[0].message}")
        return
    
    # 메시지 개수 확인
    assert len(at.session_state.messages) == 1
    print("✅ Chat Input success (No exception)")

    # 3. 페이지 점프 클릭
    at.button(key="btn_v3_0").click().run()
    if at.exception:
        print(f"❌ Jump Button failed: {at.exception[0].message}")
        return
    
    # 결과 확인
    assert at.session_state.target_page == 5
    print("✅ Jump Button success (No exception)")

    print(f"📊 Parent Reruns observed: {at.session_state.parent_rerun_count}")
    print("✨ Simulation V3 passed without any Streamlit errors!")

if __name__ == "__main__":
    test_v3_execution()
