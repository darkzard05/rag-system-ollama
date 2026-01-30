import importlib
import os
import platform
import sys


def print_step(step_name):
    print(f"\n{'=' * 60}")
    print(f" STEP: {step_name}")
    print(f"{'=' * 60}")


def check_vc_redist():
    """Checks if Microsoft Visual C++ Redistributable is likely installed."""
    print("Checking for Visual C++ Redistributable...")
    # System32 should contain vcruntime140.dll for 2015-2022 redist
    system32 = os.path.join(os.environ.get("SystemRoot", "C:\\Windows"), "System32")
    vcr_dll = os.path.join(system32, "vcruntime140.dll")

    if os.path.exists(vcr_dll):
        print(f"✅ Found: {vcr_dll}")
        return True
    else:
        print("❌ MISSING: vcruntime140.dll not found in System32.")
        print("   Please install: https://aka.ms/vs/17/release/vc_redist.x64.exe")
        return False


def check_import(module_name):
    print(f"Attempting to import {module_name}...")
    try:
        # For torch, we also want to check CUDA
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        path = getattr(module, "__file__", "unknown")
        print(f"✅ SUCCESS: {module_name} imported. Version: {version}")

        if module_name == "torch":
            import torch

            cuda_avail = torch.cuda.is_available()
            print(f"   CUDA Available: {cuda_avail}")
            if cuda_avail:
                print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        return True
    except (ImportError, OSError) as e:
        error_code = ""
        if hasattr(e, "winerror"):
            error_code = f" (WinError {e.winerror})"

        print(f"❌ FAILED: {module_name} failed to load{error_code}.")
        print(f"   Error Detail: {e}")

        if "0xc0000139" in str(e) or "DLL load failed" in str(e):
            print(
                "\n   [DIAGNOSIS]: This is a classic DLL conflict or missing runtime error."
            )
            if module_name == "torchvision":
                print(
                    "   [SUGGESTION]: torchvision is often incompatible with the installed torch."
                )
                print("   Try: pip uninstall torchvision")
        return False


def main():
    print_step("System Information")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Python: {sys.version}")

    print_step("Runtime Environment")
    check_vc_redist()

    print_step("Checking Core RAG Dependencies")
    torch_ok = check_import("torch")

    # Check torchvision separately as it's the most common failure point
    print("\nNote: torchvision is optional for this text-based RAG system.")
    check_import("torchvision")

    check_import("sentence_transformers")
    check_import("fitz")  # PyMuPDF
    check_import("faiss")

    print_step("Final Recommendation")
    if not torch_ok:
        print("❌ Your environment is broken. Run the fix script:")
        print("   powershell ./scripts/fix_windows_env.ps1")
    else:
        print(
            "✅ Core dependencies are working. If you see crashes, try uninstalling torchvision."
        )


if __name__ == "__main__":
    if platform.system() != "Windows":
        print("This script is intended for Windows diagnostics.")
    main()
