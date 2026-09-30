# PowerShell 1-Click Launcher for Pokemon Red RL - Baseline V2 (Real RL, Pure PPO)
$repoRoot = $PSScriptRoot
Set-Location $repoRoot
$env:PYTHONPATH = "external/PokemonRedExperiments/v2;src;."

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Launching Pokemon Red RL — Baseline V2 (REAL RL, PURE PPO, NO CHEATS)" -ForegroundColor Green
Write-Host "  Trained Checkpoint: poke_26214400.zip (26.2M Real RL Steps)" -ForegroundColor Cyan
Write-Host "  Controls: Close window or Ctrl+C to exit" -ForegroundColor Yellow
Write-Host "=======================================================================" -ForegroundColor Cyan

$pythonExe = if (Test-Path "C:\ProgramData\anaconda3\python.exe") { "C:\ProgramData\anaconda3\python.exe" } else { "python" }
& $pythonExe -u external/PokemonRedExperiments/v2/run_pretrained_interactive.py
