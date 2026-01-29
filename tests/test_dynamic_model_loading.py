import time
import threading
import sys
from pathlib import Path

# 프로젝트 루트 및 src를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from core.session import SessionManager


# 가짜 로딩 함수 (지연 시간 시뮬레이션)
def mock_load_llm(name):
    print(f"[LLM] 로딩 시작: {name} (3초 예상)")
    time.sleep(3)
    print(f"[LLM] 로딩 완료: {name}")
    return f"Object_LLM_{name}"


def mock_load_embedder(name):
    print(f"[Embedder] 로딩 시작: {name} (2초 예상)")
    time.sleep(2)
    print(f"[Embedder] 로딩 완료: {name}")
    return f"Object_Embedder_{name}"


def test_parallel_loading_logic():
    print("--- 모델 병렬 로딩 로직 무결성 테스트 ---")
    SessionManager.init_session()

    selected_llm = "qwen2.5:7b"
    selected_emb = "bge-m3"

    # 내부 함수 정의
    def task_llm():
        res = mock_load_llm(selected_llm)
        SessionManager.set("llm", res)

    def task_emb():
        res = mock_load_embedder(selected_emb)
        SessionManager.set("embedder", res)

    start_time = time.time()

    t1 = threading.Thread(target=task_llm)
    t2 = threading.Thread(target=task_emb)

    t1.start()
    t2.start()
    threads = [t1, t2]

    print("메인 스레드: 모든 모델 로딩 대기 중...")
    while any(t.is_alive() for t in threads):
        time.sleep(0.5)

    total_time = time.time() - start_time

    llm_obj = SessionManager.get("llm")
    emb_obj = SessionManager.get("embedder")

    print("\n" + "=" * 50)
    print(f"LLM 객체 확인: {llm_obj}")
    print(f"임베딩 객체 확인: {emb_obj}")
    print(f"총 소요 시간: {total_time:.2f}초")
    print("=" * 50)

    assert llm_obj == f"Object_LLM_{selected_llm}"
    assert emb_obj == f"Object_Embedder_{selected_emb}"

    if total_time < 4.0:
        print(
            "✅ 결과: 병렬 로딩이 성공적으로 작동했습니다. (순차 5초 대비 약 40% 시간 절약)"
        )
    else:
        print("❌ 결과: 병렬 로딩이 제대로 작동하지 않았습니다.")


if __name__ == "__main__":
    test_parallel_loading_logic()
