@echo off
title PokemonRL - Baseline V2 Weights + Jev Affordances
cd /d "%~dp0"
set PYTHONPATH=external\PokemonRedExperiments\v2;src;.
echo =======================================================================
echo   Launching PokemonRL - Baseline V2 Weights + Jev Affordance Steering
echo   Architecture: Residual Policy Steering (PPO 26.2M weights + System 1)
echo   Interactive Mode: Real PyBoy Game Boy Window with live steering HUD
echo =======================================================================
python -u scripts\run_v2_jev_hybrid_interactive.py
pause
