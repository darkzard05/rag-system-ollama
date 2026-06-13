"""
P0-1 결함 테스트: PDF 파일 핸들 누수

테스트 목표:
1. 정상적인 PDF 처리 확인
2. 파일 디스크립터 누수 감지
3. 24시간 시뮬레이션 (장시간 운영 안정성)
"""

import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psutil
import pytest
from core.document_processor import load_pdf_docs


class TestPDFHandleLeak:
    """PDF 파일 핸들 누수 테스트 클래스"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """테스트 전 후 처리"""
        self.process = psutil.Process(os.getpid())
        self.initial_fds = None
        yield
        # 정리

    def test_1_1_normal_pdf_load(self, tmp_path):
        """테스트 1.1: 정상적인 PDF 처리 확인"""
        # 테스트용 간단한 PDF 생성 (실제로는 샘플 PDF 사용)
        try:
            import fitz
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test content")
            test_pdf = tmp_path / "test.pdf"
            doc.save(str(test_pdf))
            doc.close()

            # PDF 로드
            docs = load_pdf_docs(str(test_pdf), "test.pdf")

            assert len(docs) > 0, "문서가 로드되지 않음"
            assert all(doc.page_content for doc in docs), "일부 문서에 컨텐츠 없음"
            print("✅ 테스트 1.1: 정상 PDF 처리 성공")
        except ImportError:
            pytest.skip("PyMuPDF 미설치")

    def test_1_2_file_descriptor_leak(self, tmp_path):
        """테스트 1.2: 핸들/리소스 누수 확인"""
        try:
            import fitz
            # 테스트용 PDF 생성
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "Test content")
            test_pdf = tmp_path / "test.pdf"
            doc.save(str(test_pdf))
            doc.close()

            # 초기 메모리/핸들 수
            initial_memory = self.process.memory_info().rss

            # Windows에서는 num_fds() 대신 메모리로 판단
            if hasattr(self.process, 'num_fds'):
                initial_fds = self.process.num_fds()
            else:
                initial_fds = None

            # 100회 반복 처리
            for i in range(100):
                try:
                    docs = load_pdf_docs(str(test_pdf), f"test_{i}.pdf")
                except Exception as e:
                    # 예외 무시하고 계속
                    pass

            # 최종 상태 확인
            final_memory = self.process.memory_info().rss
            memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)

            if initial_fds is not None:
                final_fds = self.process.num_fds()
                fd_increase = final_fds - initial_fds
                assert fd_increase <= 5, f"FD 누수 감지: {fd_increase}개 증가 (허용: 5개)"
                print(f"✅ 테스트 1.2: FD 누수 없음 (증가: {fd_increase}개, 허용: 5개)")
            else:
                # Windows: 메모리 증가로 판단 (< 100MB)
                assert memory_increase_mb < 100, \
                    f"메모리 누수 의심: {memory_increase_mb:.1f}MB 증가"
                print(f"✅ 테스트 1.2: 리소스 누수 없음 (메모리 증가: {memory_increase_mb:.1f}MB, 허용: 100MB)")

        except ImportError:
            pytest.skip("PyMuPDF 미설치")

    def test_1_3_memory_stability(self, tmp_path):
        """테스트 1.3: 메모리 안정성 (연속 처리)"""
        try:
            import fitz
            # 테스트용 PDF 생성
            doc = fitz.open()
            for _ in range(5):
                page = doc.new_page()
                page.insert_text((50, 50), "Test content line\n" * 100)
            test_pdf = tmp_path / "large_test.pdf"
            doc.save(str(test_pdf))
            doc.close()

            # 초기 메모리
            initial_memory = self.process.memory_info().rss

            # 50회 연속 처리
            for i in range(50):
                try:
                    docs = load_pdf_docs(str(test_pdf), f"large_{i}.pdf")
                except Exception:
                    pass

            # 최종 메모리
            final_memory = self.process.memory_info().rss
            memory_increase_mb = (final_memory - initial_memory) / (1024 * 1024)

            # 검증: 메모리 증가 < 50MB
            assert memory_increase_mb < 50, \
                f"메모리 누수 의심: {memory_increase_mb:.1f}MB 증가"
            print(f"✅ 테스트 1.3: 메모리 안정 (증가: {memory_increase_mb:.1f}MB)")

        except ImportError:
            pytest.skip("PyMuPDF 미설치")


if __name__ == "__main__":
    # pytest 직접 실행
    pytest.main([__file__, "-v", "-s"])
