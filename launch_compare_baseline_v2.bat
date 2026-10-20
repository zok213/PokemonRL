@echo off
REM =========================================================================
REM Launch Head-to-Head Comparison: Pretrained Baseline V2 vs Retrained Model
REM =========================================================================

cd /d "%~dp0"
set PYTHONPATH=external/PokemonRedExperiments/v2;src;.

echo =========================================================================
echo   LAUNCHING BASELINE V2 EMPIRICAL COMPARISON DEMO ON GAME BOY
echo   Model 1: Their Pretrained Baseline V2 (26.2M steps PPO)
echo   Model 2: Our Retrained Baseline V2 Model
echo =========================================================================

python scripts/compare_baseline_v2_models_demo.py --steps 100 %*

pause
