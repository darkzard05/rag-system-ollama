import sys
import platform
import importlib
import os

def print_step(step_name):
    print(f"\n{'='*50}")
    print(f"STEP: {step_name}")
    print(f"{ '='*50}")

def check_import(module_name):
    print(f"Attempting to import {module_name}...")
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        path = getattr(module, '__file__', 'unknown')
        print(f"✅ SUCCESS: {module_name} imported. Version: {version}")
        print(f"   Path: {path}")
        return True
    except ImportError as e:
        print(f"❌ FAILED: Could not import {module_name}")
        print(f"   Error: {e}")
        return False
    except OSError as e:
        print(f"❌ CRASH/OS ERROR: {module_name} caused an OS-level error (DLL missing?)")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ UNEXPECTED ERROR: {e}")
        return False

def main():
    print_step("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")
    print(f"Architecture: {platform.machine()}")
    
    # Check 1: Torch
    print_step("Checking PyTorch")
    if not check_import("torch"):
        print("CRITICAL: PyTorch failed. Subsequent checks may fail.")
    
    # Check 2: Torchvision (often the culprit on Windows)
    print_step("Checking Torchvision")
    check_import("torchvision")
    
    # Check 3: Sentence Transformers
    print_step("Checking Sentence Transformers")
    check_import("sentence_transformers")
    
    print_step("Diagnosis Complete")
    print("If you see 'DLL load failed' errors above, it usually indicates missing Visual C++ Redistributables.")
    print("Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")

if __name__ == "__main__":
    main()
