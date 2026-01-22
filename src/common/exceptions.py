"""
RAG 시스템 전용 예외 클래스 모음.

모든 예외는 PDFProcessingError를 상속하여 통일된 예외 처리가 가능합니다.
예외 타입별로 다른 복구 전략을 적용할 수 있습니다.

예시:
    try:
        process_pdf(file)
    except EmptyPDFError:
        # 빈 PDF 처리: 사용자에게 안내 메시지
        logger.warning("PDF에 추출 가능한 텍스트가 없습니다.")
    except VectorStoreError:
        # 벡터 저장소 오류: 재시도
        retry_with_backoff()
    except PDFProcessingError as e:
        # 다른 모든 예외
        logger.error(f"PDF 처리 오류: {e}")
"""


class PDFProcessingError(Exception):
    """
    PDF 처리 과정의 기본 예외 클래스.
    
    모든 RAG 시스템 관련 예외는 이 클래스를 상속합니다.
    
    Attributes:
        message (str): 오류 메시지
        details (dict): 추가 컨텍스트 정보 (파일명, 라인 번호 등)
    """
    
    def __init__(self, message: str, details: dict = None):
        """
        PDF 처리 예외 초기화.
        
        Args:
            message: 사용자 친화적 오류 메시지
            details: 디버깅용 추가 정보
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """예외를 문자열로 표현."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class EmptyPDFError(PDFProcessingError):
    """
    추출 가능한 텍스트가 없는 PDF 오류.
    
    원인:
    - 이미지로만 구성된 PDF (OCR 미지원)
    - 보호된 PDF 파일
    - 손상된 PDF 파일
    
    복구 전략:
    - 사용자에게 다른 PDF 제공 요청
    - OCR 기능 추가 (향후)
    """
    
    def __init__(self, filename: str = None, details: dict = None):
        """
        빈 PDF 예외 초기화.
        
        Args:
            filename: PDF 파일명
            details: 추가 정보 (압축률, 페이지 수 등)
        """
        message = f"PDF '{filename}'에 추출 가능한 텍스트가 없습니다."
        if not filename:
            message = "PDF에 추출 가능한 텍스트가 없습니다."
        
        final_details = details or {}
        if filename:
            final_details["filename"] = filename
        
        super().__init__(message, final_details)


class InsufficientChunksError(PDFProcessingError):
    """
    청킹 후 텍스트가 너무 적어서 의미 있는 임베딩을 만들 수 없는 오류.
    
    원인:
    - PDF 텍스트가 매우 짧음
    - 청킹 파라미터 설정이 너무 엄격함
    
    복구 전략:
    - 청킹 파라미터 완화 (청크 크기 감소, 오버랩 증가)
    - 더 큰 PDF 제공 요청
    """
    
    def __init__(self, chunk_count: int = None, min_required: int = None, details: dict = None):
        """
        불충분한 청크 예외 초기화.
        
        Args:
            chunk_count: 현재 청크 개수
            min_required: 최소 요구 청크 개수
            details: 추가 정보
        """
        message = "청킹 후 의미 있는 청크가 부족합니다."
        if chunk_count is not None and min_required is not None:
            message = (
                f"청크 개수({chunk_count})가 최소 요구치({min_required})에 미달합니다. "
                f"더 큰 PDF를 업로드하거나 청킹 설정을 조정해주세요."
            )
        
        final_details = details or {}
        if chunk_count is not None:
            final_details["chunk_count"] = chunk_count
        if min_required is not None:
            final_details["min_required"] = min_required
        
        super().__init__(message, final_details)


class VectorStoreError(PDFProcessingError):
    """
    벡터 저장소 생성, 접근, 업데이트 중 발생하는 오류.
    
    원인:
    - FAISS 초기화 실패
    - 임베딩 모델 로드 실패
    - 디스크 공간 부족
    - 캐시 손상
    
    복구 전략:
    - 캐시 재구성
    - 임베딩 모델 재다운로드
    - 디스크 공간 확보
    """
    
    def __init__(self, operation: str = None, reason: str = None, details: dict = None):
        """
        벡터 저장소 오류 초기화.
        
        Args:
            operation: 실패한 작업 (create, load, update, search)
            reason: 실패 원인
            details: 추가 정보
        """
        message = "벡터 저장소 처리 중 오류가 발생했습니다."
        if reason and operation:
            message = f"벡터 저장소 {operation} 실패: {reason}"
        elif reason:
            message = reason
        elif operation:
            message = f"벡터 저장소 {operation} 중 오류가 발생했습니다."
        
        final_details = details or {}
        if operation:
            final_details["operation"] = operation
        if reason:
            final_details["reason"] = reason
        
        super().__init__(message, final_details)


class LLMInferenceError(PDFProcessingError):
    """
    LLM 추론(응답 생성) 중 발생하는 오류.
    
    원인:
    - Ollama 서버 연결 실패
    - 모델 로드 실패
    - 타임아웃
    - 메모리 부족
    - 형식 오류
    
    복구 전략:
    - Ollama 서버 재시작
    - 모델 재다운로드
    - 타임아웃 증가
    - 질문 단순화
    """
    
    def __init__(self, model: str = None, reason: str = None, details: dict = None):
        """
        LLM 추론 오류 초기화.
        
        Args:
            model: LLM 모델명
            reason: 실패 원인 (timeout, connection_error, out_of_memory, etc.)
            details: 추가 정보 (시도 횟수, 경과 시간 등)
        """
        message = "LLM 응답 생성 중 오류가 발생했습니다."
        
        if reason == "timeout":
            message = (
                f"LLM 응답 생성이 시간 제한을 초과했습니다. "
                f"더 간단한 질문으로 시도해주세요."
            )
        elif reason == "connection_error":
            message = (
                f"LLM 서버(Ollama)에 연결할 수 없습니다. "
                f"Ollama가 실행 중인지 확인해주세요."
            )
        elif reason == "out_of_memory":
            message = (
                f"LLM 처리 중 메모리 부족. "
                f"다른 프로세스를 종료하거나 더 간단한 질문으로 시도해주세요."
            )
        elif model and reason:
            message = f"LLM '{model}' 추론 실패: {reason}"
        elif model:
            message = f"LLM '{model}' 추론 중 오류가 발생했습니다."
        
        final_details = details or {}
        if model:
            final_details["model"] = model
        if reason:
            final_details["reason"] = reason
        
        super().__init__(message, final_details)


class EmbeddingModelError(PDFProcessingError):
    """
    임베딩 모델 로드, 실행 중 발생하는 오류.
    
    원인:
    - 모델 다운로드 실패
    - GPU 메모리 부족
    - 모델 파일 손상
    - 지원하지 않는 모델
    
    복구 전략:
    - 모델 재다운로드
    - CPU 모드로 변경
    - 다른 모델 선택
    """
    
    def __init__(self, model: str = None, reason: str = None, details: dict = None):
        """
        임베딩 모델 오류 초기화.
        
        Args:
            model: 모델명
            reason: 실패 원인
            details: 추가 정보
        """
        message = "임베딩 모델 처리 중 오류가 발생했습니다."
        if model and reason:
            message = f"임베딩 모델 '{model}' 실패: {reason}"
        elif model:
            message = f"임베딩 모델 '{model}' 로드 중 오류가 발생했습니다."
        
        final_details = details or {}
        if model:
            final_details["model"] = model
        if reason:
            final_details["reason"] = reason
        
        super().__init__(message, final_details)
