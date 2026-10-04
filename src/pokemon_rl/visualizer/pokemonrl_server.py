"""
pokemonrl_server.py — High-Performance PokemonRL Affordance Visualizer Server
=============================================================================
Direct entry point for the PokemonRL neuro-symbolic affordance dashboard.
Runs 100% local cycle-accurate Game Boy emulation with real WRAM grounding.
"""

from __future__ import annotations
import os
from pokemon_rl.visualizer.jevos_server import (
    PokemonRLVisualizerServer,
    SharedVisualizerState,
    EmulationWorker,
    find_free_port,
)

__all__ = ["PokemonRLVisualizerServer", "SharedVisualizerState", "EmulationWorker"]

if __name__ == "__main__":
    env_port = os.environ.get("POKEMONRL_PORT") or os.environ.get("JEVOS_PORT")
    env_state = os.environ.get("POKEMONRL_STATE")
    port = int(env_port) if env_port else find_free_port(8090)
    server = PokemonRLVisualizerServer(port=port, state_path=env_state)
    server.start(blocking=True)
