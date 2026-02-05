
$ErrorActionPreference = "Stop"

Write-Host "=== Step 1: Training Model ===" -ForegroundColor Green
python training/train.py
if ($LASTEXITCODE -ne 0) { Write-Error "Training failed!"; exit 1 }

Write-Host "`n=== Step 2: Merging Adapters ===" -ForegroundColor Green
python training/merge.py
if ($LASTEXITCODE -ne 0) { Write-Error "Merging failed!"; exit 1 }

Write-Host "`n=== Step 3: Converting to GGUF ===" -ForegroundColor Green
python training/convert_to_gguf.py
if ($LASTEXITCODE -ne 0) { Write-Error "Conversion failed!"; exit 1 }

Write-Host "`n=== Pipeline Complete! You can now chat. ===" -ForegroundColor Cyan
Write-Host "To chat, run: python apps/chat_cli.py"
