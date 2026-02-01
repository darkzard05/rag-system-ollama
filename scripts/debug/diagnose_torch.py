import sys

import torch


def diagnose():
    print(f"Python Executable: {sys.executable}")
    print(f"PyTorch Version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Alloc: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("\n❌ 원인 분석:")
        if not hasattr(torch, "cuda"):
            print("- PyTorch가 CUDA 지원 없이 빌드되었습니다.")
        else:
            print(
                "- CUDA 드라이버 또는 런타임이 설치되지 않았거나 PyTorch 버전과 호환되지 않습니다."
            )
            print(
                "- 혹은 CUDA_VISIBLE_DEVICES 환경 변수가 GPU를 가리고 있을 수 있습니다."
            )


if __name__ == "__main__":
    diagnose()
