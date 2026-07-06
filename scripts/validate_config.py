import yaml
import sys
import os

# Add src to sys.path so we can import common.schemas


try:
    from common.schemas import AppConfig
except ImportError as e:
    print(f"Error importing AppConfig from common.schemas: {e}")
    sys.exit(1)

def validate():
    config_path = "config.yml"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    print(f"Loading {config_path}...")
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            sys.exit(1)

    print("Validating against AppConfig schema (src/common/schemas.py)...")
    try:
        AppConfig(**config_dict)
        print("✅ 설정 검증 성공")
    except Exception as e:
        print(f"❌ 설정 검증 실패:\n{e}")
        sys.exit(1)

if __name__ == "__main__":
    validate()
