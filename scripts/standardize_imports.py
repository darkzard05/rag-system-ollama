import os
import re

def standardize_imports(root_dir):
    # Patterns to search for and their replacements
    patterns = {
        r'^from (core|ui|api|common|services|infra|security)\.': r'from src.\1.',
        r'^import (core|ui|api|common|services|infra|security)\.': r'import src.\1.',
    }
    
    # Path hacks to remove
    path_hacks = [
        r'sys\.path\.append\(os\.path\.join\(os\.getcwd\(\), "src"\)\)',
        r'sys\.path\.append\(os\.path\.abspath\(os\.path\.join\(os\.path\.dirname\(__file__\), "..", ".."\)\)',
        r'sys\.path\.insert\(0, "src"\)',
    ]

    files_changed = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                original_content = content
                
                # 1. Replace imports
                for pattern, replacement in patterns.items():
                    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
                
                # 2. Remove path hacks
                for hack in path_hacks:
                    content = re.sub(hack, '', content)

                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    files_changed += 1

    return files_changed

if __name__ == "__main__":
    # Use the workspace root
    root = os.getcwd()
    changed = standardize_imports(root)
    print(f"Successfully standardized imports in {changed} files.")
