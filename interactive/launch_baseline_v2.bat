@echo off
cd /d "%~dp0\.."
set PYTHONPATH=external\PokemonRedExperiments\v2;src;.
echo =======================================================================
echo   Launching Pokemon Red RL — Baseline V2 (REAL RL, PURE PPO, NO CHEATS)
echo   Trained Checkpoint: poke_26214400.zip (26.2M Real RL Steps)
echo   Controls: Close window or Ctrl+C to exit
echo =======================================================================

if exist "C:\ProgramData\anaconda3\python.exe" (
    "C:\ProgramData\anaconda3\python.exe" -u external\PokemonRedExperiments\v2\run_pretrained_interactive.py
) else (
    python -u external\PokemonRedExperiments\v2\run_pretrained_interactive.py
)
pause
