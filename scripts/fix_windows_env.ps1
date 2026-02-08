# scripts/fix_windows_env.ps1
# Windows RAG Environment Fix Script

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  RAG System Windows Environment Fixer" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan

# 1. Check for VC++ Redistributable
$vcrPath = "$env:SystemRoot\System32\vcruntime140.dll"
if (-not (Test-Path $vcrPath)) {
    Write-Host "[!] Missing VC++ Redistributable 2015-2022." -ForegroundColor Red
    Write-Host "    Opening download link... Please install and restart this script."
    Start-Process "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    exit
}

# 2. Cleanup conflicting packages
Write-Host "[*] Cleaning up conflicting packages (torch, torchvision)..." -ForegroundColor Yellow
pip uninstall -y torch torchvision 2>$null

# 3. Install stable Windows build
Write-Host "[*] Installing stable torch (CPU version by default)..." -ForegroundColor Yellow
Write-Host "    To install GPU version, edit requirements-win.txt."
pip install -r requirements-win.txt

# 4. Final verification
Write-Host "[*] Running diagnostics..." -ForegroundColor Yellow
python scripts/diagnose_windows.py

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  FIX COMPLETED. Try running: streamlit run src/main.py" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
