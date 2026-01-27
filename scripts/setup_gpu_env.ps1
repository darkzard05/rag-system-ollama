# GPU-First Environment Setup Script for RAG System
# This script creates a Conda environment and installs PyTorch with GPU support.

$EnvName = "rag-ollama"
$PythonVersion = "3.10"
$CudaVersion = "12.1"

# Set absolute path for requirements.txt (Project Root is one level up from this script)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

Write-Host "--- Starting GPU Environment Setup for [$EnvName] ---" -ForegroundColor Cyan
Write-Host "Project Root: $ProjectRoot" -ForegroundColor Gray
Write-Host "Requirements: $RequirementsPath" -ForegroundColor Gray

# 1. Check if Conda is installed
if (!(Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Error "Conda is not installed or not in PATH. Please install Miniconda or Anaconda first."
    exit
}

# 2. Create Conda Environment
Write-Host "Step 1: Creating Conda environment ($EnvName) with Python $PythonVersion..." -ForegroundColor Yellow
conda create -n $EnvName python=$PythonVersion -y

# 3. Install PyTorch with GPU Support
Write-Host "Step 2: Installing PyTorch with CUDA $CudaVersion support..." -ForegroundColor Yellow
# Note: Using conda install for better dependency resolution of CUDA toolkit
conda install -n $EnvName pytorch torchvision pytorch-cuda=$CudaVersion -c pytorch -c nvidia -y

# 4. Install remaining requirements via Pip
Write-Host "Step 3: Installing remaining dependencies from requirements.txt..." -ForegroundColor Yellow
$Condabin = "$(conda info --base)\envs\$EnvName\Scripts\pip.exe"
if (!(Test-Path $Condabin)) {
    # Fallback for some conda installations
    $Condabin = "$(conda info --base)\envs\$EnvName\bin\pip"
}

if (Test-Path $RequirementsPath) {
    Write-Host "Running: pip install -r $RequirementsPath" -ForegroundColor Gray
    & $Condabin install -r $RequirementsPath
} else {
    Write-Error "CRITICAL: Could not find requirements.txt at $RequirementsPath"
    exit
}

# 5. Verification
Write-Host "Step 4: Verifying GPU support..." -ForegroundColor Yellow
$VerifyCmd = "import torch; print(f'Torch version: {torch.__version__}'); print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
$PythonExe = "$(conda info --base)\envs\$EnvName\python.exe"
& $PythonExe -c $VerifyCmd

Write-Host "--- Setup Complete! ---" -ForegroundColor Green
Write-Host "To activate the environment, run:" -ForegroundColor White
Write-Host "conda activate $EnvName" -ForegroundColor Cyan