# launch_pokemonrl_visualizer.ps1
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$env:PYTHONPATH = "src;."

Write-Host "=======================================================================" -ForegroundColor Cyan
Write-Host "  Launching PokemonRL — Autonomous Neuro-Symbolic RL Visualizer" -ForegroundColor Green
Write-Host "  Architecture: Local SMDP Affordance Decisions on CPU (0 API Cost)" -ForegroundColor Yellow
Write-Host "  Dashboard URL: http://127.0.0.1:8090" -ForegroundColor White
Write-Host "  Hardware Core: LR35902 Game Boy Emulation via PyBoy 2.7.0" -ForegroundColor White
Write-Host "=======================================================================" -ForegroundColor Cyan

Start-Process "http://127.0.0.1:8090"
python -u src/pokemon_rl/visualizer/pokemonrl_server.py
