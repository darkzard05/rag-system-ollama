import os
import re
import glob

# Internal packages to standardize - Added "cache"
PACKAGES = ["core", "ui", "api", "common", "services", "infra", "security", "cache"]
# Regex to match "from src.package" or "import src.package", optional trailing dot
PATTERN = re.compile(r"(from|import)\s+src\.(" + "|".join(PACKAGES) + r")(\.|$)", re.MULTILINE)
REPLACEMENT = r"\1 \2\3"

def standardize():
    target_patterns = [
        "tests/**/*.py",
        "scripts/**/*.py",
        "verify_preprocessing.py"
    ]
    
    fixed_files = 0
    for pattern in target_patterns:
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                new_content = PATTERN.sub(REPLACEMENT, content)
                
                if new_content != content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    fixed_files += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")
                
    print(f"Successfully standardized imports in {fixed_files} files.")

if __name__ == "__main__":
    standardize()
