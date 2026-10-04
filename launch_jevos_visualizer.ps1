# launch_jevos_visualizer.ps1 -> forwards to launch_pokemonrl_visualizer.ps1
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$ScriptDir\launch_pokemonrl_visualizer.ps1"
