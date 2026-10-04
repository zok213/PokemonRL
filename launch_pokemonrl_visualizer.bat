@echo off
cd /d "%~dp0"
set PYTHONPATH=src;.
echo =======================================================================
echo   Launching PokemonRL — Autonomous Neuro-Symbolic RL Visualizer
echo   Architecture: Local SMDP Affordance Decisions on CPU (0 API Cost)
echo   Dashboard URL: http://127.0.0.1:8090
echo   Hardware Core: LR35902 Game Boy Emulation via PyBoy 2.7.0
echo =======================================================================

if exist "C:\ProgramData\anaconda3\python.exe" (
    start http://127.0.0.1:8090
    "C:\ProgramData\anaconda3\python.exe" -u src\pokemon_rl\visualizer\pokemonrl_server.py
) else (
    start http://127.0.0.1:8090
    python -u src\pokemon_rl\visualizer\pokemonrl_server.py
)
pause
