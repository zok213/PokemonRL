"""
test_go_explore.py — Unit Tests for Go-Explore State Archive & DFD
=================================================================
Verifies delta compression, deterministic restoration, Directed Frontier Distance sampling,
hierarchical cell representations, and stale frontier culling.
"""

import pytest
import numpy as np
from pokemon_rl.exploration.go_explore import (
    GoExploreStateArchive,
    CellRepresentation,
    DeltaStateCompressor,
)


def test_delta_compression_roundtrip():
    """Verify lossless compression and decompression."""
    keyframe = bytes(8192)
    mutated = bytearray(8192)

    # Mutate 25 bytes
    for i in range(25):
        mutated[i * 128] = (i * 7 + 3) & 0xFF

    compressed = DeltaStateCompressor.compress(bytes(mutated), keyframe)
    # Compression ratio should be > 95%
    compression_ratio = (1.0 - len(compressed) / 8192.0) * 100.0
    assert compression_ratio > 95.0

    restored = DeltaStateCompressor.decompress(compressed, keyframe)
    assert bytes(restored) == bytes(mutated)


def test_archive_registration_and_restoration():
    archive = GoExploreStateArchive(max_cells=1000)
    cell = CellRepresentation(map_id=0, x_coarse=2, y_coarse=3, safari_bucket=50)

    state_bytes = bytes(bytearray([i % 256 for i in range(8192)]))
    registered = archive.register_state(
        cell, state_bytes, trajectory_cost=10, score=2.0, remaining_budget=450, badge_count=1
    )
    assert registered is True
    assert len(archive.archive) == 1
    assert cell in archive.active_frontier

    restored = archive.restore_cell(cell)
    assert bytes(restored) == state_bytes


def test_hierarchical_cell_properties():
    """Verify macro-cell and micro-cell projections."""
    cell = CellRepresentation(map_id=4, x_coarse=14, y_coarse=22, safari_bucket=30)
    assert cell.macro_cell == (4, 30)
    assert cell.micro_cell == (7, 11)


def test_stale_frontier_culling():
    """Verify stagnant cells are removed from active sampling frontier."""
    archive = GoExploreStateArchive(max_cells=1000)
    c1 = CellRepresentation(map_id=0, x_coarse=1, y_coarse=1, safari_bucket=50)
    c2 = CellRepresentation(map_id=0, x_coarse=2, y_coarse=2, safari_bucket=50)

    archive.register_state(c1, bytes(8192), badge_count=0)
    archive.register_state(c2, bytes(8192), badge_count=0)

    assert len(archive.active_frontier) == 2

    # Simulate c1 being sampled 20 times without new discovery
    archive.archive[c1].stagnant_samples = 20

    culled = archive.cull_stale_frontier(max_stagnant_samples=15)
    assert culled == 1
    assert c1 not in archive.active_frontier
    assert c2 in archive.active_frontier
    # Preserved in full archive for evaluation/replay
    assert c1 in archive.archive


def test_dfd_frontier_sampling():
    archive = GoExploreStateArchive(max_cells=1000)

    # Register shallow cell (0 badges)
    cell_shallow = CellRepresentation(map_id=0, x_coarse=1, y_coarse=1, safari_bucket=50)
    archive.register_state(cell_shallow, bytes(8192), badge_count=0, remaining_budget=100)

    # Register deep cell (4 badges, high budget)
    cell_deep = CellRepresentation(map_id=5, x_coarse=10, y_coarse=12, safari_bucket=40)
    archive.register_state(cell_deep, bytes(8192), badge_count=4, remaining_budget=400)

    # DFD sampling should strongly prioritize cell_deep
    rng = np.random.default_rng(42)
    deep_count = 0
    N_SAMPLES = 200
    for _ in range(N_SAMPLES):
        chosen_cell, _ = archive.sample_frontier_cell(alpha_progress=3.0, rng=rng)
        if chosen_cell == cell_deep:
            deep_count += 1

    # cell_deep should be chosen > 70% of the time due to DFD exponential progress weighting
    assert deep_count > N_SAMPLES * 0.70
