"""
test_pipeline.py — End-to-End Production Pipeline Tests
======================================================
Verifies integrated coordination of Go-Explore, Dynamic Masking, Combat Head, and GRPO.
"""

import pytest
import numpy as np
from pokemon_rl.systems.production_pipeline import ProductionAgentPipeline, Action, RAMMap


def test_pipeline_initialization():
    pipeline = ProductionAgentPipeline(group_size=8)
    assert pipeline.group_size == 8
    assert pipeline.archive is not None
    assert pipeline.masker is not None
    assert pipeline.combat is not None
    assert pipeline.grpo is not None
    assert pipeline.env is not None


def test_pipeline_short_training_run():
    pipeline = ProductionAgentPipeline(group_size=8)
    metrics = pipeline.run_training_cycle(num_iterations=25)

    assert metrics["total_actions"] == 25 * 8
    assert metrics["throughput_sps"] > 50.0
    assert metrics["archive_unique_cells"] > 0
    assert metrics["mean_delta_compression_ratio"] > 95.0


def test_pipeline_dfd_sampling():
    pipeline = ProductionAgentPipeline(group_size=8)
    pipeline.run_training_cycle(num_iterations=10)

    sampled_cell, state_bytes = pipeline.archive.sample_frontier_cell(alpha_progress=2.0)
    assert sampled_cell is not None
    assert len(state_bytes) > 0


def test_pipeline_hardware_action_mask():
    pipeline = ProductionAgentPipeline(group_size=8)
    mock_reader = lambda addr: 0x01 if addr == RAMMap.JOY_IGNORE else 0
    mask = pipeline.masker.compute_action_mask(mock_reader)
    assert not mask[Action.A]


def test_pipeline_torch_autograd_training():
    pipeline = ProductionAgentPipeline(group_size=4, use_torch_policy=True)
    metrics = pipeline.run_training_cycle(num_iterations=3)

    assert metrics["use_torch_policy"] is True
    assert metrics["total_actions"] == 12
    assert "mean_policy_loss" in metrics
    assert isinstance(metrics["mean_policy_loss"], float)

