# Mental Health Chatbot - Windows Setup Script
# PowerShell version of setup.sh

param(
    [switch]$SkipModelDownload = $false,
    [switch]$SkipLlamaCpp = $false
)

$ErrorActionPreference = "Stop"

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Mental Health Chatbot - Environment Setup" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# 0. Check for Python
Write-Host "[1/6] Checking Python installation..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ from https://www.python.org/downloads/" -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version
Write-Host "Found: $pythonVersion" -ForegroundColor Green

# Check Python version
$versionMatch = $pythonVersion -match "Python (\d+)\.(\d+)"
if ($versionMatch) {
    $major = [int]$Matches[1]
    $minor = [int]$Matches[2]
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 10)) {
        Write-Host "Error: Python 3.10 or higher is required. Found: $pythonVersion" -ForegroundColor Red
        exit 1
    }
}

# 1. Create and activate virtual environment
Write-Host ""
Write-Host "[2/6] Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "Virtual environment created." -ForegroundColor Green
}
else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
Write-Host ""
Write-Host "[3/6] Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip --quiet

# 3. Install Python Dependencies
Write-Host ""
Write-Host "[4/6] Installing Python dependencies from requirements.txt..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
pip install -r requirements.txt

# Check if CUDA is available
$cudaAvailable = $false
try {
    nvidia-smi 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        $cudaAvailable = $true
        Write-Host "CUDA GPU detected!" -ForegroundColor Green
    }
}
catch {
    Write-Host "No CUDA GPU detected. Will use CPU." -ForegroundColor Yellow
}

# Install llama-cpp-python with appropriate backend
Write-Host "Installing llama-cpp-python..." -ForegroundColor Yellow

# Function to check for compiler tools
function Test-Compiler {
    return (Get-Command "cl.exe" -ErrorAction SilentlyContinue) -or (Get-Command "nmake.exe" -ErrorAction SilentlyContinue)
}

if ($cudaAvailable) {
    if (Test-Compiler) {
        Write-Host "CUDA and Compiler detected. Building from source..." -ForegroundColor Yellow
        $env:CMAKE_ARGS = "-DGGML_CUDA=on"
        try {
            pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
            if ($LASTEXITCODE -ne 0) { throw "Installation failed" }
        }
        catch {
             Write-Host "Source build failed. Falling back to pre-built CUDA wheels..." -ForegroundColor Yellow
             $env:CMAKE_ARGS = ""
             # Try common CUDA versions (12.3, 12.4 usually cover most, or use the generic cu121/cu122)
             pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
        }
    }
    else {
        Write-Host "CUDA detected but no C++ compiler (VS Build Tools) found." -ForegroundColor Yellow
        Write-Host "Attempting to install pre-built binary for CUDA..." -ForegroundColor Yellow
        # Remove CMAKE_ARGS to avoid triggering a build
        $env:CMAKE_ARGS = "" 
        pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
    }
}
else {
    Write-Host "Installing for CPU..." -ForegroundColor Yellow
    $env:CMAKE_ARGS = ""
    pip install llama-cpp-python --upgrade --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
}

# 4. Download and Build llama.cpp TOOLS
if (-not $SkipLlamaCpp) {
    Write-Host ""
    Write-Host "[5/6] Setting up llama.cpp (for GGUF conversion)..." -ForegroundColor Yellow
    
    if (-not (Test-Path "llama.cpp")) {
        Write-Host "Cloning llama.cpp repository..." -ForegroundColor Yellow
        git clone https://github.com/ggerganov/llama.cpp
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Warning: Git clone failed. You may need to install Git." -ForegroundColor Yellow
            Write-Host "Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
        }
        else {
            # Try to build
            Write-Host "Building llama.cpp..." -ForegroundColor Yellow
            Set-Location llama.cpp
            
            # Check for CMake
            if (Get-Command cmake -ErrorAction SilentlyContinue) {
                New-Item -ItemType Directory -Force -Path build | Out-Null
                Set-Location build
                
                if ($cudaAvailable) {
                    cmake .. -DGGML_CUDA=ON
                }
                else {
                    cmake ..
                }
                
                cmake --build . --config Release
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "llama.cpp built successfully!" -ForegroundColor Green
                }
                else {
                    Write-Host "Warning: Build failed. You may need to install Visual Studio Build Tools." -ForegroundColor Yellow
                    Write-Host "Download from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
                }
                
                Set-Location ..\..
            }
            else {
                Write-Host "Warning: CMake not found. Skipping build." -ForegroundColor Yellow
                Write-Host "Install CMake from: https://cmake.org/download/" -ForegroundColor Yellow
                Set-Location ..
            }
        }
    }
    else {
        Write-Host "llama.cpp directory already exists. Skipping clone." -ForegroundColor Green
    }
}
else {
    Write-Host ""
    Write-Host "[5/6] Skipping llama.cpp setup (--SkipLlamaCpp flag)" -ForegroundColor Gray
}

# 5. Download Base Model from HuggingFace
if (-not $SkipModelDownload) {
    Write-Host ""
    Write-Host "[6/6] Downloading Base Model (Llama-3.2-1B-Instruct)..." -ForegroundColor Yellow
    Write-Host "NOTE: You must be logged in to HuggingFace and have access to this model." -ForegroundColor Gray
    Write-Host ""
    
    # Check if huggingface-cli is available
    if (-not (Get-Command huggingface-cli -ErrorAction SilentlyContinue)) {
        Write-Host "Error: huggingface-cli not found." -ForegroundColor Red
        Write-Host "Installing huggingface_hub..." -ForegroundColor Yellow
        pip install huggingface_hub[cli]
    }
    
    # Check if user is logged in
    Write-Host "Checking HuggingFace login status..." -ForegroundColor Yellow
    $loginCheck = huggingface-cli whoami 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "You need to login to HuggingFace." -ForegroundColor Yellow
        Write-Host "Please run: huggingface-cli login" -ForegroundColor Cyan
        Write-Host "Get your token from: https://huggingface.co/settings/tokens" -ForegroundColor Gray
        Write-Host ""
        Write-Host "After logging in, run this script again." -ForegroundColor Yellow
        exit 0
    }
    
    Write-Host "Logged in as: $loginCheck" -ForegroundColor Green
    
    # Download model
    if (-not ((Test-Path "models\base\config.json") -and (Test-Path "models\base\model.safetensors"))) {
        Write-Host "Downloading model (this will take several minutes)..." -ForegroundColor Yellow
        New-Item -ItemType Directory -Force -Path models | Out-Null
        
        huggingface-cli download meta-llama/Llama-3.2-1B-Instruct --local-dir models/base --local-dir-use-symlinks False --exclude "*.pth" "*.pt"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Model downloaded successfully!" -ForegroundColor Green
        }
        else {
            Write-Host "Warning: Model download failed." -ForegroundColor Yellow
            Write-Host "You may need to request access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct" -ForegroundColor Yellow
        }
    }
    else {
        Write-Host "Model files found in 'models\base'. Skipping download." -ForegroundColor Green
    }
}
else {
    Write-Host ""
    Write-Host "[6/6] Skipping model download (--SkipModelDownload flag)" -ForegroundColor Gray
}

# Create necessary directories
Write-Host ""
Write-Host "Creating project directories..." -ForegroundColor Yellow
$directories = @(
    "models\checkpoints",
    "models\lora_final",
    "models\merged_model",
    "models\gguf",
    "rag\chroma_db",
    "logs",
    "outputs",
    "tests",
    "docs",
    "scripts",
    "visualization",
    "monitoring"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}
Write-Host "Directories created." -ForegroundColor Green

# Summary
Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Prepare data:    python data/prepare_data.py" -ForegroundColor White
Write-Host "  2. Build RAG index: python rag/build_index.py" -ForegroundColor White
Write-Host "  3. Train model:     python training/train.py" -ForegroundColor White
Write-Host "  4. Merge adapters:  python training/merge.py" -ForegroundColor White
Write-Host "  5. Convert to GGUF: python training/convert_to_gguf.py" -ForegroundColor White
Write-Host "  6. Run chatbot:     python apps/chat_cli.py" -ForegroundColor White
Write-Host ""
Write-Host "For web interface:    python apps/chat_web.py" -ForegroundColor Cyan
Write-Host ""
