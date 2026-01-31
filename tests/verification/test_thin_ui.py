import time

import httpx
import streamlit as st

# --- 중요: 무거운 라이브러리(torch, langchain, core 등) 임포트 절대 금지 ---

st.set_page_config(page_title="Thin UI Speed Test", layout="wide")


def get_api_status():
    """백엔드 서버 상태 체크"""
    try:
        with httpx.Client(timeout=1.0) as client:
            response = client.get("http://127.0.0.1:8000/api/v1/health")
            return response.json()
    except Exception:
        return None


def main():
    st.title("⚡ Thin UI 부팅 속도 테스트")

    start_time = time.time()

    st.sidebar.header("Backend Status")
    status = get_api_status()

    if status:
        st.sidebar.success(f"✅ Connected: {status.get('status')}")
        st.sidebar.info(f"Model: {status.get('model', 'Unknown')}")
    else:
        st.sidebar.error("❌ Backend Offline (Run api_server.py first)")

    st.write("### 🏎️ 속도 측정 결과")
    load_time = time.time() - start_time
    st.metric("UI Interaction Latency", f"{load_time:.4f}s")

    st.info("""
    이 화면이 뜨는 속도를 기존 main.py와 비교해보세요.
    기존 앱이 'torch'와 'langchain'을 로드하느라 5~10초 걸릴 때,
    이 앱은 임포트 오버헤드가 없어 즉시 실행됩니다.
    """)

    # 단순 채팅 시뮬레이션
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요 (백엔드로 전달됨)"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            st.write("백엔드 서버로 요청을 보내는 중... (실제 구현 시 API 호출)")


if __name__ == "__main__":
    main()
