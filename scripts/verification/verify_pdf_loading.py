import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.getcwd(), "src"))

from core.rag_core import _load_pdf_docs

class TestPDFLoadingStrategy(unittest.TestCase):
    
    def setUp(self):
        import logging
        logging.getLogger("core.rag_core").setLevel(logging.INFO)

    @patch("core.rag_core.fitz.open")
    @patch("os.path.getsize")
    @patch("builtins.open")
    @patch("core.rag_core.concurrent.futures.ThreadPoolExecutor")
    def test_small_file_loading(self, mock_executor_cls, mock_open, mock_getsize, mock_fitz_open):
        """작은 파일(50MB 이하)은 메모리로 읽어들여야 한다."""
        print("\n[Test] Small File Strategy 실행 중...")
        
        # 1. 상황 설정: 10MB
        mock_getsize.return_value = 10 * 1024 * 1024
        
        mock_file_handle = MagicMock()
        mock_file_handle.read.return_value = b"fake_pdf_content"
        mock_open.return_value.__enter__.return_value = mock_file_handle
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        mock_fitz_open.return_value = mock_doc
        
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance
        
        mock_future = MagicMock()
        mock_future.result.return_value = [] 
        mock_executor_instance.submit.return_value = mock_future
        
        # 2. 실행
        with patch("concurrent.futures.as_completed", return_value=[mock_future]):
            try:
                _load_pdf_docs("small.pdf", "small.pdf")
            except Exception:
                pass # 결과 없음 에러 무시

        # 3. 검증
        if mock_executor_instance.submit.called:
            args, _ = mock_executor_instance.submit.call_args
            file_bytes_arg = args[1]
            
            print(f"  - 전달된 file_bytes: {file_bytes_arg[:10]}...")
            
            self.assertIsNotNone(file_bytes_arg, "작은 파일은 file_bytes가 None이면 안 됩니다.")
            self.assertEqual(file_bytes_arg, b"fake_pdf_content", "파일 내용이 일치해야 합니다.")
            print("  ✅ [PASS] Small File Test")
        else:
            self.fail("Executor.submit이 호출되지 않았습니다.")

    @patch("core.rag_core.fitz.open")
    @patch("os.path.getsize")
    @patch("builtins.open")
    @patch("core.rag_core.concurrent.futures.ThreadPoolExecutor")
    def test_large_file_loading(self, mock_executor_cls, mock_open, mock_getsize, mock_fitz_open):
        """큰 파일(50MB 초과)은 파일 경로만 전달해야 한다."""
        print("\n[Test] Large File Strategy 실행 중...")
        
        # 1. 상황 설정: 100MB
        mock_getsize.return_value = 100 * 1024 * 1024
        
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 10
        mock_fitz_open.return_value = mock_doc
        
        mock_executor_instance = MagicMock()
        mock_executor_cls.return_value.__enter__.return_value = mock_executor_instance
        
        mock_future = MagicMock()
        mock_future.result.return_value = []
        mock_executor_instance.submit.return_value = mock_future
        
        # 2. 실행
        with patch("concurrent.futures.as_completed", return_value=[mock_future]):
            try:
                _load_pdf_docs("large.pdf", "large.pdf")
            except Exception:
                pass # 결과 없음 에러 무시

        # 3. 검증
        if mock_executor_instance.submit.called:
            args, _ = mock_executor_instance.submit.call_args
            file_bytes_arg = args[1]
            
            print(f"  - 전달된 file_bytes: {file_bytes_arg}")
            
            self.assertIsNone(file_bytes_arg, "큰 파일은 file_bytes가 None이어야 합니다.")
            print("  ✅ [PASS] Large File Test")
        else:
            self.fail("Executor.submit이 호출되지 않았습니다.")

if __name__ == "__main__":
    unittest.main()