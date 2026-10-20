# =========================================================================
# Launch Head-to-Head Comparison: Pretrained Baseline V2 vs Retrained Model
# =========================================================================

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$env:PYTHONPATH = "external/PokemonRedExperiments/v2;src;."

Write-Host "=========================================================================" -ForegroundColor Cyan
Write-Host "  LAUNCHING BASELINE V2 EMPIRICAL COMPARISON DEMO ON GAME BOY" -ForegroundColor Green
Write-Host "  Model 1: Their Pretrained Baseline V2 (26.2M steps PPO)" -ForegroundColor Yellow
Write-Host "  Model 2: Our Retrained Baseline V2 Model" -ForegroundColor Yellow
Write-Host "=========================================================================" -ForegroundColor Cyan

python scripts/compare_baseline_v2_models_demo.py --steps 100 $args
