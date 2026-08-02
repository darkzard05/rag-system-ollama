import os
import sys

# Add src to sys.path so we can import common.config
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)

try:
    from common.config import CONFIG_PATH, _load_config
except ImportError as e:
    print(f"Error importing common.config: {e}")
    sys.exit(1)


def validate():
    if not CONFIG_PATH.exists():
        print(f"Error: {CONFIG_PATH} not found.")
        sys.exit(1)

    print(f"Loading {CONFIG_PATH}...")
    try:
        config = _load_config()
        print("✅ 설정 검증 성공")
        print(f"   - 로드된 설정 항목 수: {len(config)}")
    except Exception as e:
        print(f"❌ 설정 검증 실패:\n{e}")
        sys.exit(1)


if __name__ == "__main__":
    validate()
