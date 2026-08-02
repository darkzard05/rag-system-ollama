import os
import re

# Only target these internal packages
PACKAGES = ["core", "ui", "api", "common", "services", "infra", "security"]
SKIP_DIRS = {".git", ".codegraph", "__pycache__", "venv", ".venv", "node_modules"}

# Pattern to match "from src.package." or "import src.package."
PATTERN = re.compile(r"(from|import)\s+src\.(" + "|".join(PACKAGES) + r")\.")
REPLACEMENT = r"\1 \2."

def standardize():
    fixed_files = 0
    for root, dirs, files in os.walk("."):
        # Prune directories to speed up walk
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            if file.endswith(".py") and file != "standardize_imports_final.py":
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
                    
    print(f"Successfully standardized imports in {fixed_files} files.")

if __name__ == "__main__":
    standardize()
