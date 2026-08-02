import re

# Files identified via grep as having patch("src. ...")
FILES_TO_FIX = [
    "tests/integration/test_state_sync.py",
    "tests/unit/test_structured_nodes.py",
    "scripts/verify_session_refactor.py",
    "tests/unit/test_reranker.py",
    "tests/unit/test_reducer_state.py",
    "tests/unit/test_chat_lifecycle.py",
    "tests/unit/test_background_worker.py",
    "tests/unit/test_chunking.py",
    "tests/test_ui_bridge.py",
    "tests/test_rag_save_count.py",
]

# Pattern to match patch("src.package..." or patch('src.package...')
# Group 1: patch( and the opening quote
PATTERN = re.compile(r"(patch\(\s*([\"']))src\.")
REPLACEMENT = r"\1"


def fix_patches():
    fixed_count = 0
    for relative_path in FILES_TO_FIX:
        try:
            with open(relative_path, "r", encoding="utf-8") as f:
                content = f.read()

            new_content = PATTERN.sub(REPLACEMENT, content)

            if new_content != content:
                with open(relative_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                fixed_count += 1
                print(f"Fixed: {relative_path}")
        except Exception as e:
            print(f"Error processing {relative_path}: {e}")

    print(f"Successfully standardized patch paths in {fixed_count} files.")


if __name__ == "__main__":
    fix_patches()
