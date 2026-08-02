import os
import re

# Pattern to match patch("src.package..." or patch('src.package...')
# Group 1: patch( and the opening quote
# Group 2: src. (to be removed)
PATTERN = re.compile(r"(patch\(\s*([\"']))src\.")
REPLACEMENT = r"\1"


def fix_patches():
    fixed_files = 0
    for root, dirs, files in os.walk("."):
        # Prune irrelevant directories
        dirs[:] = [
            d
            for d in dirs
            if d not in {".git", ".codegraph", "__pycache__", "venv", ".venv"}
        ]
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
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

    print(f"Successfully standardized patch paths in {fixed_files} files.")


if __name__ == "__main__":
    fix_patches()
