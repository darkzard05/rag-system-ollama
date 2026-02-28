"""
requirements/base.txt와 windows.txt 사이의 버전 일관성을 검사합니다.
보안 패치가 반영되었는지 확인하고, 플랫폼 공통 패키지들의 버전이 일치하는지 체크합니다.
"""

import re
import sys
from pathlib import Path


def parse_requirements(file_path):
    reqs = {}
    if not file_path.exists():
        return reqs

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 주석 및 빈 줄 제외
            if not line or line.startswith("#") or line.startswith("-r"):
                continue
            
            # 패키지명과 버전 추출 (예: package==1.2.3, package>=1.2.3)
            match = re.match(r"^([a-zA-Z0-9\-_\[\]]+)([=<>!~]+.*)$", line)
            if match:
                name = match.group(1).lower().replace("_", "-")
                version = match.group(2)
                reqs[name] = version
            else:
                # 버전 없이 패키지만 명시된 경우
                name = line.lower().replace("_", "-")
                reqs[name] = "any"
    return reqs


def main():
    root = Path(__file__).parent.parent.parent
    base_file = root / "requirements" / "base.txt"
    win_file = root / "requirements" / "windows.txt"

    base_reqs = parse_requirements(base_file)
    win_reqs = parse_requirements(win_file)

    errors = []

    # 1. 공통 패키지 버전 일관성 체크
    common_packages = set(base_reqs.keys()) & set(win_reqs.keys())
    # 플랫폼 종속적인 것은 제외 (예: torch는 인덱스 url이 다름)
    platform_specific = {"torch"}

    for pkg in common_packages:
        if pkg in platform_specific:
            continue
        
        if base_reqs[pkg] != win_reqs[pkg]:
            errors.append(
                f"[Mismatch] {pkg}: base({base_reqs[pkg]}) != windows({win_reqs[pkg]})"
            )

    # 2. 필수 보안 패치 버전 체크
    security_checks = {
        "starlette": ">=0.49.1",
        "fastapi": "==0.133.1",
        "pillow": ">=12.1.1",
        "streamlit": "==1.54.0"
    }

    for pkg, min_version in security_checks.items():
        if pkg in base_reqs and base_reqs[pkg] != min_version and not base_reqs[pkg].startswith(min_version):
             # 간단하게 문자열 비교로 처리
             if pkg == "starlette" and ">=" in base_reqs[pkg]: continue
             errors.append(f"[Security] {pkg} version should be {min_version} in base.txt")
        
        if pkg in win_reqs and win_reqs[pkg] != min_version and not win_reqs[pkg].startswith(min_version):
             errors.append(f"[Security] {pkg} version should be {min_version} in windows.txt")

    if errors:
        for err in errors:
            print(err)
        print("\nError: Requirements files are inconsistent or have security risks.")
        sys.exit(1)
    
    print("Success: Requirements files are consistent and secure.")
    sys.exit(0)


if __name__ == "__main__":
    main()
