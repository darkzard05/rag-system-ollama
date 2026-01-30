import sys
import warnings

# Streamlit bare mode 경고 무시 (Import 전에 설정)
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

from pathlib import Path

from langchain_core.documents import Document

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from core.session import SessionManager


def test_history_lightweight():
    print("🚀 [Test] 채팅 히스토리 경량화 및 무결성 검증 시작")

    # 1. 초기화
    SessionManager.reset_all_state()

    # 2. 테스트용 문서 준비
    doc1 = Document(page_content="인공지능은 매우 유용합니다.", metadata={"page": 1})
    doc2 = Document(page_content="RAG는 검색을 활용합니다.", metadata={"page": 2})

    # 3. 메시지 추가 (중복 문서 포함)
    print("\n📝 메시지 추가 중 (중복 문서 포함)...")
    # 첫 번째 대화: doc1, doc2 참조
    SessionManager.add_message("assistant", "첫 번째 답변", documents=[doc1, doc2])
    # 두 번째 대화: doc1만 참조 (중복 발생 시나리오)
    SessionManager.add_message("assistant", "두 번째 답변", documents=[doc1])

    # 4. 검증: doc_pool 상태 확인
    pool = SessionManager.get("doc_pool")
    print(f"📊 [Pool] 저장된 유니크 문서 수: {len(pool)} (기대값: 2)")

    if len(pool) == 2:
        print("✅ 결과: 중복 문서가 성공적으로 풀링되었습니다.")
    else:
        print(f"❌ 결과: 풀링 실패 (문서 수: {len(pool)})")

    # 5. 검증: 메시지 구조 및 복원 확인
    messages = SessionManager.get_messages()
    print("\n🔍 [Message 1] 복원 테스트 시작...")
    msg1 = messages[0]

    if "documents" not in msg1 and "doc_ids" in msg1:
        print("✅ 결과: 메시지 내 원본 문장 제거 및 ID 변환 완료.")

        # ID로 실제 내용 복원 테스트
        doc_ids = msg1["doc_ids"]
        restored_texts = [pool[d_id].page_content for d_id in doc_ids if d_id in pool]

        if len(restored_texts) == 2 and restored_texts[0] == doc1.page_content:
            print("✅ 결과: ID를 통한 원본 데이터 복원 성공 (무결성 통과).")
        else:
            print("❌ 결과: 데이터 복원 실패.")
    else:
        print("❌ 결과: 메시지 구조가 기대와 다릅니다.")

    # 6. 최종 요약
    print("\n" + "=" * 40)
    print("🏆 최종 결과: 채팅 히스토리 최적화 및 무결성 검증 완료")
    print("=" * 40)


if __name__ == "__main__":
    test_history_lightweight()
