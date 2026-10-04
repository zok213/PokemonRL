"""
test_jevos_visualizer.py — Comprehensive Unit Tests for Jevos Web Visualizer & Server
====================================================================================
Tests state snapshotting, base64 frame encoding, affordance formulation,
decision logging, and HTTP endpoint serving.
"""

import json
import threading
import time
import urllib.request
import urllib.error
import numpy as np
import pytest

from pokemon_rl.visualizer.jevos_server import (
    SharedVisualizerState,
    EmulationWorker,
    JevosVisualizerServer,
)


def test_shared_visualizer_state_update_and_snapshot():
    state = SharedVisualizerState()
    
    # Verify initial default state
    snap = state.snapshot()
    assert "frame" in snap
    assert "stats" in snap
    assert snap["stats"]["badges"] == 0
    assert snap["stats"]["best"] == 8
    assert "model_reads" in snap
    assert "model_answers" in snap
    assert "log" in snap

    # Test updating frame with uint8 array
    raw_rgba = np.random.randint(0, 256, (144, 160, 4), dtype=np.uint8)
    state.update_frame(raw_rgba)
    snap2 = state.snapshot()
    assert len(snap2["frame"]) > 50  # Base64 string encoded

    # Test recording a decision
    test_reads = {
        "state": {"location": "Route 1", "coordinates": [12, 28]},
        "questions": {"north": "advance north", "south": "retreat south"}
    }
    test_answers = [
        {"id": "north", "question": "advance north", "prob": 88, "is_chosen": True},
        {"id": "south", "question": "retreat south", "prob": 12, "is_chosen": False}
    ]
    state.record_decision(
        reads=test_reads,
        answers=test_answers,
        chosen_action="north",
        chosen_prob=88,
        context="Route 1 clear path",
        latency_ms=28.5,
        model_source="jevos-q4_k_m"
    )

    snap3 = state.snapshot()
    assert snap3["stats"]["decisions"] == 1
    assert snap3["stats"]["model_source"] == "jevos-q4_k_m"
    assert snap3["model_reads"]["state"]["location"] == "Route 1"
    assert len(snap3["model_answers"]) == 2
    assert snap3["log"][0]["action"] == "north"
    assert snap3["log"][0]["prob"] == 88


def test_emulation_worker_affordance_fork():
    state = SharedVisualizerState()
    worker = EmulationWorker(state, rom_path="roms/non_existent.gb")
    
    # Test synthetic frame generator
    frame = worker._generate_synthetic_gb_frame(42)
    assert frame.shape == (144, 160, 3)
    assert frame.dtype == np.uint8

    # Test evaluating affordance fork on mock overworld
    mock_overworld = {
        "map_id": 0,
        "x": 5,
        "y": 4,
        "in_battle": False,
        "badges": 0,
        "text_box_id": 0,
        "active_mon": {"hp": 20, "max_hp": 20, "level": 5}
    }
    worker._evaluate_affordance_fork(mock_overworld)

    snap = state.snapshot()
    assert snap["stats"]["decisions"] == 1
    assert "questions" in snap["model_reads"]
    assert len(snap["model_answers"]) >= 2
    for ans in snap["model_answers"]:
        assert 0 <= ans["prob"] <= 100

    # Test evaluating affordance fork in battle
    mock_battle = {
        "map_id": 12,
        "x": 10,
        "y": 14,
        "in_battle": True,
        "badges": 0,
        "text_box_id": 0,
        "active_mon": {"hp": 18, "max_hp": 20, "level": 5}
    }
    worker._evaluate_affordance_fork(mock_battle)
    snap2 = state.snapshot()
    assert snap2["stats"]["decisions"] == 2
    assert "scratch" in snap2["model_reads"]["questions"]


def test_jevos_server_http_endpoints():
    test_port = 8995
    server = JevosVisualizerServer(port=test_port, rom_path="roms/non_existent.gb")
    
    # Run server in background thread
    server_thread = threading.Thread(target=server.server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.15)  # Allow server to bind

    base_url = f"http://127.0.0.1:{test_port}"
    try:
        # 1. Test GET / (HTML Dashboard)
        with urllib.request.urlopen(f"{base_url}/", timeout=2.0) as resp:
            assert resp.status == 200
            html = resp.read().decode("utf-8")
            assert "jevos plays Pokémon Red" in html
            assert "WHAT THE MODEL READS" in html
            assert "THE MODEL'S ANSWERS" in html

        # 2. Test GET /api/state (JSON State Snapshot)
        with urllib.request.urlopen(f"{base_url}/api/state", timeout=2.0) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "stats" in data
            assert "model_reads" in data
            assert "model_answers" in data

        # 3. Test POST /api/control (Pause and Resume)
        req_pause = urllib.request.Request(
            f"{base_url}/api/control",
            data=json.dumps({"action": "pause"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_pause, timeout=2.0) as resp:
            assert resp.status == 200
            res = json.loads(resp.read().decode("utf-8"))
            assert res.get("status") == "ok"
        assert server.shared_state.paused is True

        req_resume = urllib.request.Request(
            f"{base_url}/api/control",
            data=json.dumps({"action": "resume"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req_resume, timeout=2.0) as resp:
            assert resp.status == 200
        assert server.shared_state.paused is False

    finally:
        server.server.shutdown()
        server.server.server_close()
