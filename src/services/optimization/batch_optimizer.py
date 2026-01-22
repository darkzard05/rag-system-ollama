"""
배치 사이즈 자동 최적화 모듈.

GPU 메모리를 감지하여 최적 배치 크기를 자동 계산합니다.
이를 통해 OOM (Out of Memory) 방지 및 성능 최적화를 달성합니다.

사용 예시:
    from services.optimization.batch_optimizer import get_optimal_batch_size
    batch_size = get_optimal_batch_size()
    embeddings = model.encode(texts, batch_size=batch_size)
"""

import logging
import torch
from typing import Optional

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> tuple[bool, int]:
    """
    GPU 메모리 정보 조회.
    
    Returns:
        (is_gpu_available, total_memory_mb): GPU 사용 가능 여부, 총 메모리(MB)
    """
    if not torch.cuda.is_available():
        return False, 0
    
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        total_memory_mb = total_memory // (1024 ** 2)
        return True, total_memory_mb
    except Exception as e:
        logger.warning(f"GPU 메모리 정보 조회 실패: {e}")
        return False, 0


def get_available_gpu_memory() -> int:
    """
    현재 사용 가능한 GPU 메모리 조회.
    
    Returns:
        available_memory_mb: 사용 가능 메모리(MB)
    """
    if not torch.cuda.is_available():
        return 0
    
    try:
        torch.cuda.reset_peak_memory_stats()
        available_memory = torch.cuda.mem_get_info()[0]
        available_memory_mb = available_memory // (1024 ** 2)
        return available_memory_mb
    except Exception as e:
        logger.warning(f"GPU 사용 가능 메모리 조회 실패: {e}")
        return 0


def get_optimal_batch_size(
    device: Optional[str] = None,
    model_type: str = "embedding",
    safety_margin: float = 0.8
) -> int:
    """
    GPU 메모리 기반 최적 배치 크기 계산.
    
    메모리 기반 휴리스틱:
    - GPU 8GB 이상: 128
    - GPU 4-8GB: 64
    - GPU 2-4GB: 32
    - GPU < 2GB: 16
    - CPU: 32
    
    Args:
        device: 디바이스 타입 ('cuda', 'cpu', None자동감지)
        model_type: 모델 타입 ('embedding', 'reranker', 'llm')
        safety_margin: 안전 여유율 (0.0~1.0, 기본값 0.8 = 80% 사용)
    
    Returns:
        optimal_batch_size: 추천 배치 크기
    """
    # 디바이스 자동 감지
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # CPU 사용 시
    if device == "cpu":
        logger.info("CPU 디바이스 감지: 배치 크기 32 권장")
        return 32
    
    # GPU 메모리 정보 조회
    is_available, total_memory_mb = get_gpu_memory_info()
    if not is_available:
        logger.warning("GPU를 사용할 수 없음: CPU 배치 크기(32) 반환")
        return 32
    
    # 현재 사용 가능 메모리
    available_memory_mb = get_available_gpu_memory()
    
    # 안전 여유를 고려한 실제 사용 가능 메모리
    usable_memory_mb = int(available_memory_mb * safety_margin)
    
    logger.info(
        f"GPU 메모리: {total_memory_mb}MB 총용량, "
        f"{available_memory_mb}MB 사용가능, "
        f"{usable_memory_mb}MB 안전 범위"
    )
    
    # 모델별 배치 크기 결정 규칙
    if model_type == "embedding":
        # 임베딩 모델: 비교적 가벼움
        if total_memory_mb >= 8192:  # 8GB 이상
            batch_size = 128
            logger.info("Embedding 모델: 배치 크기 128 (GPU >= 8GB)")
        elif total_memory_mb >= 4096:  # 4GB 이상
            batch_size = 64
            logger.info("Embedding 모델: 배치 크기 64 (GPU 4-8GB)")
        elif total_memory_mb >= 2048:  # 2GB 이상
            batch_size = 32
            logger.info("Embedding 모델: 배치 크기 32 (GPU 2-4GB)")
        else:  # 2GB 미만
            batch_size = 16
            logger.info("Embedding 모델: 배치 크기 16 (GPU < 2GB)")
    
    elif model_type == "reranker":
        # Reranker: 중간 정도의 메모리 사용
        if total_memory_mb >= 8192:
            batch_size = 64
            logger.info("Reranker 모델: 배치 크기 64 (GPU >= 8GB)")
        elif total_memory_mb >= 4096:
            batch_size = 32
            logger.info("Reranker 모델: 배치 크기 32 (GPU 4-8GB)")
        elif total_memory_mb >= 2048:
            batch_size = 16
            logger.info("Reranker 모델: 배치 크기 16 (GPU 2-4GB)")
        else:
            batch_size = 8
            logger.info("Reranker 모델: 배치 크기 8 (GPU < 2GB)")
    
    elif model_type == "llm":
        # LLM: 매우 메모리 집약적
        if total_memory_mb >= 16384:  # 16GB 이상
            batch_size = 16
            logger.info("LLM 모델: 배치 크기 16 (GPU >= 16GB)")
        elif total_memory_mb >= 8192:  # 8GB 이상
            batch_size = 8
            logger.info("LLM 모델: 배치 크기 8 (GPU 8-16GB)")
        elif total_memory_mb >= 4096:  # 4GB 이상
            batch_size = 4
            logger.info("LLM 모델: 배치 크기 4 (GPU 4-8GB)")
        else:
            batch_size = 1
            logger.info("LLM 모델: 배치 크기 1 (GPU < 4GB)")
    
    else:
        # 기본값
        batch_size = 32
        logger.warning(f"알 수 없는 모델 타입 '{model_type}': 기본 배치 크기 32 반환")
    
    return batch_size


def estimate_memory_usage(
    batch_size: int,
    sequence_length: int,
    model_type: str = "embedding"
) -> int:
    """
    배치에 소요되는 예상 메모리 계산.
    
    Args:
        batch_size: 배치 크기
        sequence_length: 시퀀스 길이 (토큰)
        model_type: 모델 타입
    
    Returns:
        estimated_memory_mb: 예상 메모리(MB)
    """
    # 기본 메모리 계산 (토큰당 약 4바이트 * sequence_length * batch_size)
    base_memory = batch_size * sequence_length * 4 / (1024 ** 2)
    
    # 모델별 추가 오버헤드
    if model_type == "embedding":
        overhead_factor = 2.5  # 임베딩 모델 오버헤드
    elif model_type == "reranker":
        overhead_factor = 3.0  # Reranker 오버헤드
    elif model_type == "llm":
        overhead_factor = 5.0  # LLM 오버헤드 (매우 큼)
    else:
        overhead_factor = 2.0
    
    estimated_memory = base_memory * overhead_factor
    return int(estimated_memory)


def validate_batch_size(
    batch_size: int,
    device: str = "cuda",
    model_type: str = "embedding"
) -> tuple[bool, str]:
    """
    배치 크기 유효성 검사.
    
    Args:
        batch_size: 검증할 배치 크기
        device: 디바이스 타입
        model_type: 모델 타입
    
    Returns:
        (is_valid, message): 유효 여부, 메시지
    """
    # 최소/최대 배치 크기 제한
    min_batch_size = 1
    max_batch_size = 512
    
    if batch_size < min_batch_size:
        return False, f"배치 크기는 최소 {min_batch_size}이어야 합니다."
    
    if batch_size > max_batch_size:
        return False, f"배치 크기는 최대 {max_batch_size}이어야 합니다."
    
    # GPU 메모리 확인
    if device == "cuda":
        is_available, total_memory_mb = get_gpu_memory_info()
        if not is_available:
            return False, "GPU를 사용할 수 없습니다."
        
        # 예상 메모리 계산 (임의의 시퀀스 길이 512 기준)
        estimated_memory = estimate_memory_usage(batch_size, 512, model_type)
        
        # 현재 사용 가능한 메모리 확인
        available_memory_mb = get_available_gpu_memory()

        if estimated_memory > available_memory_mb:
            return False, (
                f"배치 크기 {batch_size}는 {estimated_memory}MB 메모리 필요하지만, "
                f"현재 GPU 사용 가능 메모리는 {available_memory_mb}MB입니다."
            )
    
    return True, f"배치 크기 {batch_size}는 유효합니다."


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    print("\n=== 배치 최적화 테스트 ===\n")
    
    # GPU 정보 조회
    is_gpu, total_mem = get_gpu_memory_info()
    print(f"GPU 사용 가능: {is_gpu}")
    if is_gpu:
        print(f"총 메모리: {total_mem}MB")
        print(f"사용 가능: {get_available_gpu_memory()}MB")
    
    # 최적 배치 크기 계산
    print("\n--- 모델별 최적 배치 크기 ---")
    embedding_batch = get_optimal_batch_size(model_type="embedding")
    print(f"Embedding: {embedding_batch}")
    
    reranker_batch = get_optimal_batch_size(model_type="reranker")
    print(f"Reranker: {reranker_batch}")
    
    llm_batch = get_optimal_batch_size(model_type="llm")
    print(f"LLM: {llm_batch}")
    
    # 메모리 예상치 계산
    print("\n--- 메모리 사용량 예상 ---")
    emb_mem = estimate_memory_usage(embedding_batch, 512, "embedding")
    print(f"Embedding (배치={embedding_batch}): {emb_mem}MB")
    
    # 유효성 검사
    print("\n--- 배치 크기 유효성 검사 ---")
    is_valid, msg = validate_batch_size(embedding_batch, device="cuda")
    print(f"Embedding 배치 크기 {embedding_batch}: {msg}")
