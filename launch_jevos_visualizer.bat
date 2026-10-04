@echo off
cd /d "%~dp0"
set PYTHONPATH=src;.
echo =======================================================================
echo   Launching Pokemon Red RL — Jevos Local Affordance Visualizer
echo   Architecture: Local Yes/No Decisions on Laptop CPU in 25-110 ms
echo   Dashboard URL: http://127.0.0.1:8080
echo   Controls: Open browser, or close terminal / Ctrl+C to exit
echo =======================================================================

if exist "C:\ProgramData\anaconda3\python.exe" (
    start http://127.0.0.1:8080
    "C:\ProgramData\anaconda3\python.exe" -u src\pokemon_rl\visualizer\jevos_server.py
) else (
    start http://127.0.0.1:8080
    python -u src\pokemon_rl\visualizer\jevos_server.py
)
pause
