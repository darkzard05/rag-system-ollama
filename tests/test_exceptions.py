"""
커스텀 예외 클래스 단위 테스트.

테스트 대상:
- PDFProcessingError: 기본 예외 클래스
- EmptyPDFError: 빈 PDF
- InsufficientChunksError: 불충분한 청크
- VectorStoreError: 벡터 저장소 오류
- LLMInferenceError: LLM 추론 오류
- EmbeddingModelError: 임베딩 모델 오류

실행: pytest tests/test_exceptions.py -v
"""

import pytest
import sys
from pathlib import Path

# src 디렉토리를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.exceptions import (
    PDFProcessingError,
    EmptyPDFError,
    InsufficientChunksError,
    VectorStoreError,
    LLMInferenceError,
    EmbeddingModelError,
)


class TestPDFProcessingError:
    """기본 예외 클래스 테스트"""

    def test_basic_exception_creation(self):
        """기본 예외 생성"""
        exc = PDFProcessingError("테스트 오류")
        assert exc.message == "테스트 오류"
        assert exc.details == {}

    def test_exception_with_details(self):
        """상세 정보 포함 예외 생성"""
        details = {"file": "test.pdf", "line": 10}
        exc = PDFProcessingError("테스트 오류", details=details)
        assert exc.message == "테스트 오류"
        assert exc.details == details

    def test_exception_str_representation(self):
        """예외의 문자열 표현"""
        exc = PDFProcessingError("테스트 오류")
        assert str(exc) == "테스트 오류"

    def test_exception_str_with_details(self):
        """상세 정보 포함 문자열 표현"""
        details = {"file": "test.pdf", "line": 10}
        exc = PDFProcessingError("테스트 오류", details=details)
        str_repr = str(exc)
        assert "테스트 오류" in str_repr
        assert "file=test.pdf" in str_repr
        assert "line=10" in str_repr

    def test_exception_is_exception(self):
        """PDFProcessingError가 Exception을 상속"""
        exc = PDFProcessingError("테스트")
        assert isinstance(exc, Exception)


class TestEmptyPDFError:
    """EmptyPDFError 테스트"""

    def test_empty_pdf_error_without_filename(self):
        """파일명 없는 빈 PDF 예외"""
        exc = EmptyPDFError()
        assert "텍스트가 없습니다" in exc.message

    def test_empty_pdf_error_with_filename(self):
        """파일명 있는 빈 PDF 예외"""
        exc = EmptyPDFError(filename="document.pdf")
        assert "document.pdf" in exc.message
        assert exc.details["filename"] == "document.pdf"

    def test_empty_pdf_error_with_details(self):
        """추가 정보 포함"""
        details = {"pages": 5, "format": "image_only"}
        exc = EmptyPDFError(filename="scan.pdf", details=details)
        assert "scan.pdf" in exc.message
        assert exc.details["filename"] == "scan.pdf"

    def test_empty_pdf_error_inheritance(self):
        """EmptyPDFError가 PDFProcessingError를 상속"""
        exc = EmptyPDFError("test.pdf")
        assert isinstance(exc, PDFProcessingError)
        assert isinstance(exc, Exception)


class TestInsufficientChunksError:
    """InsufficientChunksError 테스트"""

    def test_insufficient_chunks_basic(self):
        """기본 불충분 청크 예외"""
        exc = InsufficientChunksError()
        assert "부족합니다" in exc.message

    def test_insufficient_chunks_with_counts(self):
        """청크 개수 포함"""
        exc = InsufficientChunksError(chunk_count=3, min_required=10)
        assert "3" in exc.message
        assert "10" in exc.message
        assert exc.details["chunk_count"] == 3
        assert exc.details["min_required"] == 10

    def test_insufficient_chunks_with_details(self):
        """추가 정보 포함"""
        details = {"file_size": "2MB", "reason": "짧은 문서"}
        exc = InsufficientChunksError(chunk_count=2, min_required=5, details=details)
        assert exc.details["file_size"] == "2MB"

    def test_insufficient_chunks_inheritance(self):
        """InsufficientChunksError가 PDFProcessingError를 상속"""
        exc = InsufficientChunksError(1, 10)
        assert isinstance(exc, PDFProcessingError)


class TestVectorStoreError:
    """VectorStoreError 테스트"""

    def test_vector_store_error_basic(self):
        """기본 벡터 저장소 예외"""
        exc = VectorStoreError()
        assert "벡터 저장소" in exc.message

    def test_vector_store_error_with_operation(self):
        """작업 정보 포함"""
        exc = VectorStoreError(operation="create", reason="메모리 부족")
        assert "create" in exc.message
        assert "메모리 부족" in exc.message
        assert exc.details["operation"] == "create"

    def test_vector_store_error_with_reason_only(self):
        """이유만 포함"""
        exc = VectorStoreError(reason="디스크 쓰기 실패")
        assert "디스크 쓰기 실패" in exc.message

    def test_vector_store_error_with_details(self):
        """추가 정보 포함"""
        details = {"available_memory": "512MB", "required": "1GB"}
        exc = VectorStoreError(operation="load", reason="메모리", details=details)
        assert exc.details["available_memory"] == "512MB"


class TestLLMInferenceError:
    """LLMInferenceError 테스트"""

    def test_llm_inference_error_basic(self):
        """기본 LLM 예외"""
        exc = LLMInferenceError()
        assert "LLM" in exc.message

    def test_llm_inference_error_timeout(self):
        """타임아웃 오류"""
        exc = LLMInferenceError(model="gpt-3", reason="timeout")
        assert "시간 제한" in exc.message
        assert "간단한 질문" in exc.message

    def test_llm_inference_error_connection(self):
        """연결 오류"""
        exc = LLMInferenceError(model="ollama", reason="connection_error")
        assert "Ollama" in exc.message
        assert "연결할 수 없습니다" in exc.message

    def test_llm_inference_error_out_of_memory(self):
        """메모리 부족 오류"""
        exc = LLMInferenceError(model="llama", reason="out_of_memory")
        assert "메모리 부족" in exc.message
        assert "종료" in exc.message

    def test_llm_inference_error_with_details(self):
        """추가 정보 포함"""
        details = {"attempts": 3, "elapsed": "45.2s"}
        exc = LLMInferenceError(model="model", reason="timeout", details=details)
        assert exc.details["attempts"] == 3
        assert exc.details["elapsed"] == "45.2s"


class TestEmbeddingModelError:
    """EmbeddingModelError 테스트"""

    def test_embedding_model_error_basic(self):
        """기본 임베딩 모델 예외"""
        exc = EmbeddingModelError()
        assert "임베딩" in exc.message

    def test_embedding_model_error_with_model_name(self):
        """모델명 포함"""
        exc = EmbeddingModelError(model="sentence-transformers/all-MiniLM-L6-v2")
        assert "all-MiniLM-L6-v2" in exc.message

    def test_embedding_model_error_with_reason(self):
        """이유 포함"""
        exc = EmbeddingModelError(model="bert-base", reason="다운로드 실패")
        assert "bert-base" in exc.message
        assert "다운로드 실패" in exc.message

    def test_embedding_model_error_with_details(self):
        """추가 정보 포함"""
        details = {"url": "https://example.com", "status_code": 404}
        exc = EmbeddingModelError(
            model="model", reason="download_failed", details=details
        )
        assert exc.details["status_code"] == 404


class TestExceptionHierarchy:
    """예외 계층 구조 테스트"""

    def test_all_exceptions_inherit_from_base(self):
        """모든 커스텀 예외가 PDFProcessingError를 상속"""
        exceptions = [
            EmptyPDFError(),
            InsufficientChunksError(),
            VectorStoreError(),
            LLMInferenceError(),
            EmbeddingModelError(),
        ]
        for exc in exceptions:
            assert isinstance(exc, PDFProcessingError)
            assert isinstance(exc, Exception)

    def test_exception_catching_by_base(self):
        """기본 예외로 모든 커스텀 예외 캐치"""
        exceptions = [
            EmptyPDFError("test.pdf"),
            InsufficientChunksError(1, 10),
            VectorStoreError(operation="load"),
            LLMInferenceError(model="test"),
            EmbeddingModelError(model="test"),
        ]

        for exc in exceptions:
            try:
                raise exc
            except PDFProcessingError as e:
                assert e.message is not None


class TestExceptionCatching:
    """예외 처리 패턴 테스트"""

    def test_catch_and_re_raise(self):
        """캐치 후 재발생"""
        with pytest.raises(EmptyPDFError):
            try:
                raise EmptyPDFError("doc.pdf")
            except EmptyPDFError as e:
                assert "doc.pdf" in e.message
                raise

    def test_catch_specific_exception(self):
        """특정 예외 캐치"""
        with pytest.raises(InsufficientChunksError):
            raise InsufficientChunksError(chunk_count=0, min_required=1)

    def test_catch_base_exception(self):
        """기본 예외로 캐치"""
        with pytest.raises(PDFProcessingError):
            raise EmptyPDFError("test.pdf")

    def test_multiple_exception_types(self):
        """다양한 예외 타입 처리"""
        test_cases = [
            (EmptyPDFError(filename="test.pdf"), "추출 가능한 텍스트"),
            (InsufficientChunksError(0, 1), "청크"),
            (VectorStoreError(operation="load"), "벡터 저장소"),
        ]

        for exc, expected_keyword in test_cases:
            try:
                raise exc
            except PDFProcessingError as e:
                assert expected_keyword in e.message or expected_keyword in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
