# launch_v2_hybrid.ps1 -- 1-Click PowerShell Launcher for V2 Weights + Jev Affordances
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$env:PYTHONPATH = "external/PokemonRedExperiments/v2;src;."

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Launching PokemonRL - Baseline V2 Weights + Jev Affordance Steering" -ForegroundColor Green
Write-Host "  Architecture: Residual Policy Steering (PPO 26.2M weights + System 1)" -ForegroundColor Yellow
Write-Host "  Interactive Mode: Real PyBoy Game Boy Window with live steering HUD" -ForegroundColor White
Write-Host "=======================================================================" -ForegroundColor Cyan

python -u scripts/run_v2_jev_hybrid_interactive.py
