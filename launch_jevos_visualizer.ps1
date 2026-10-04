$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONPATH = "src;."

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Launching Pokemon Red RL — Jevos Local Affordance Visualizer" -ForegroundColor Green
Write-Host "  Architecture: Local Yes/No Decisions on Laptop CPU in 25-110 ms" -ForegroundColor Green
Write-Host "  Dashboard URL: http://127.0.0.1:8090" -ForegroundColor Yellow
Write-Host "  Controls: Open browser, or close terminal / Ctrl+C to exit" -ForegroundColor Cyan
Write-Host "=======================================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8090"

$pythonExe = if (Test-Path "C:\ProgramData\anaconda3\python.exe") { "C:\ProgramData\anaconda3\python.exe" } else { "python" }
& $pythonExe -u src/pokemon_rl/visualizer/jevos_server.py
