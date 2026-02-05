# Create GGUF Quantizations
# Creates Q8_0 (high quality) and Q4_K_M (fast) versions

Write-Host "Creating GGUF Quantizations..." -ForegroundColor Cyan

$baseModel = "models\gguf\model-f16.gguf"
$q8Model = "models\gguf\model-q8_0.gguf"
$q4Model = "models\gguf\model-q4_k_m.gguf"

if (-not (Test-Path $baseModel)) {
    Write-Host "Error: Base model not found at $baseModel" -ForegroundColor Red
    exit 1
}

# Check if llama.cpp is built
if (-not (Test-Path "llama.cpp\build\bin\Release\llama-quantize.exe")) {
    Write-Host "Building llama.cpp..." -ForegroundColor Yellow
    Push-Location llama.cpp
    
    if (-not (Test-Path "build")) {
        cmake -B build -DCMAKE_BUILD_TYPE=Release
    }
    
    cmake --build build --config Release -j 8
    Pop-Location
}

$quantizeTool = "llama.cpp\build\bin\Release\llama-quantize.exe"

if (-not (Test-Path $quantizeTool)) {
    Write-Host "Error: Quantize tool not found" -ForegroundColor Red
    exit 1
}

# Create Q8_0 (high quality, ~50% smaller)
if (-not (Test-Path $q8Model)) {
    Write-Host "`nCreating Q8_0 quantization (high quality)..." -ForegroundColor Green
    & $quantizeTool $baseModel $q8Model Q8_0
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Q8_0 created successfully!" -ForegroundColor Green
    } else {
        Write-Host "Q8_0 creation failed" -ForegroundColor Red
    }
} else {
    Write-Host "Q8_0 model already exists" -ForegroundColor Yellow
}

# Create Q4_K_M (balanced, ~75% smaller)
if (-not (Test-Path $q4Model)) {
    Write-Host "`nCreating Q4_K_M quantization (fast)..." -ForegroundColor Green
    & $quantizeTool $baseModel $q4Model Q4_K_M
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Q4_K_M created successfully!" -ForegroundColor Green
    } else {
        Write-Host "Q4_K_M creation failed" -ForegroundColor Red
    }
} else {
    Write-Host "Q4_K_M model already exists" -ForegroundColor Yellow
}

Write-Host "`nGGUF Models:" -ForegroundColor Cyan
Get-ChildItem "models\gguf\*.gguf" | Select-Object Name, @{Name="Size (GB)";Expression={[math]::Round($_.Length/1GB, 2)}}

Write-Host "`nDone! You now have 3 GGUF versions:" -ForegroundColor Green
Write-Host "  - model-f16.gguf   (Best quality, slowest)" -ForegroundColor White
Write-Host "  - model-q8_0.gguf  (High quality, balanced)" -ForegroundColor White
Write-Host "  - model-q4_k_m.gguf (Good quality, fastest)" -ForegroundColor White
