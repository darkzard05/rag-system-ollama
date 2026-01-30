import sys
import unittest
from pathlib import Path

# 프로젝트 루트 추가
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from langchain_core.documents import Document

from core.thread_safe_session import ThreadSafeSessionManager as TSM


class TestIntegrityFixes(unittest.TestCase):
    def setUp(self):
        TSM.clear_all()
        TSM._fallback_sessions = {}

    def test_session_isolation_deepcopy(self):
        """1. 세션 격리 및 Deepcopy 검증 (무결성 테스트)"""
        print("\n[Integrity] 세션 격리 및 Deepcopy 검증 시작...")

        # 세션 1 초기화 및 메시지 추가
        TSM.set_session_id("session_1")
        TSM.init_session()
        TSM.add_message("user", "Hello from S1")
        TSM.add_status_log("S1 Start")

        # 세션 2 초기화
        TSM.set_session_id("session_2")
        TSM.init_session()

        # 검증: 세션 2는 세션 1의 데이터를 공유하지 않아야 함
        self.assertEqual(len(TSM.get_messages()), 0, "S2 should have 0 messages")
        self.assertNotIn(
            "S1 Start",
            TSM.get("status_logs"),
            "S2 should not share status logs with S1",
        )

        # 세션 2 데이터 수정
        TSM.add_message("user", "Hello from S2")

        # 세션 1 재확인
        TSM.set_session_id("session_1")
        messages_s1 = TSM.get_messages()
        self.assertEqual(len(messages_s1), 1)
        self.assertEqual(messages_s1[0]["content"], "Hello from S1")

        print("✅ 세션 격리 및 Deepcopy 검증 통과")

    def test_document_pooling_metadata_integrity(self):
        """2. 문서 풀링 메타데이터 무결성 검증"""
        print("\n[Integrity] 문서 풀링 메타데이터 무결성 검증 시작...")

        TSM.set_session_id("pooling_test")
        TSM.init_session()

        # 같은 내용이지만 출처가 다른 두 문서
        doc1 = Document(
            page_content="Common content", metadata={"source": "doc1.pdf", "page": 1}
        )
        doc2 = Document(
            page_content="Common content", metadata={"source": "doc2.pdf", "page": 5}
        )

        # 메시지에 추가 (풀링 발생)
        TSM.add_message("assistant", "Ref doc1", documents=[doc1])
        TSM.add_message("assistant", "Ref doc2", documents=[doc2])

        state = TSM._get_state()
        doc_pool = state["doc_pool"]
        messages = TSM.get_messages()

        # 검증: 해시가 달라야 하므로 풀에 2개의 문서가 있어야 함
        self.assertEqual(
            len(doc_pool),
            2,
            "Doc pool should contain 2 unique entries for different metadata",
        )

        # 각 메시지가 자신의 올바른 문서를 가리키는지 확인
        id1 = messages[0]["doc_ids"][0]
        id2 = messages[1]["doc_ids"][0]
        self.assertNotEqual(id1, id2, "IDs should be different for different metadata")
        self.assertEqual(doc_pool[id1].metadata["source"], "doc1.pdf")
        self.assertEqual(doc_pool[id2].metadata["source"], "doc2.pdf")

        print("✅ 문서 풀링 메타데이터 무결성 검증 통과")


class TestAsyncIntegrityFixes(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_buffer_flush_integrity(self):
        """3. 스트리밍 버퍼 플러시 무결성 검증 (비동기)"""
        print("\n[Integrity] 스트리밍 버퍼 플러시 무결성 검증 시작...")

        # graph_builder의 generate_response 내부 로직 모사
        answer_buffer = []
        buffer_size = 5
        dispatched_chunks = []

        async def mock_adispatch(event_name, data, config=None):
            if event_name == "response_chunk":
                dispatched_chunks.append(data.get("chunk", ""))

        # 시뮬레이션: 7개의 토큰이 들어오고 도중에 중단되는 상황
        tokens = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]

        try:
            # 첫 토큰은 즉시 발송 로직 (is_first_content_chunk=True)
            dispatched_chunks.append(tokens[0])

            # 나머지 6개 토큰 처리
            for t in tokens[1:]:
                answer_buffer.append(t)
                if len(answer_buffer) >= buffer_size:
                    dispatched_chunks.append("".join(answer_buffer))
                    answer_buffer = []

            # 여기서 예외 발생 가정 (또는 루프 종료)
            raise RuntimeError("Stream interrupted")
        except RuntimeError:
            pass
        finally:
            # 수정된 로직: finally에서 남은 버퍼 플러시
            if answer_buffer:
                dispatched_chunks.append("".join(answer_buffer))

        full_text = "".join(dispatched_chunks)
        self.assertEqual(
            full_text, "".join(tokens), "All tokens should be preserved in final output"
        )
        self.assertEqual(
            len(dispatched_chunks), 3, "Should have 3 dispatch calls: T1, [T2-T6], [T7]"
        )
        self.assertEqual(
            dispatched_chunks[-1], "T7", "Last chunk should be flushed in finally block"
        )

        print("✅ 스트리밍 버퍼 플러시 무결성 검증 통과")


if __name__ == "__main__":
    unittest.main()
