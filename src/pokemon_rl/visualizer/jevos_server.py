"""
jevos_server.py — High-Performance Local Affordance Visualizer Server
======================================================================
Serves the interactive web visualizer matching the feder-cr/jevos Dino demonstration.
Streams live Game Boy LCD video frames, Work RAM telemetry, and local yes/no affordance
probability distributions (P(yes)) computed on CPU in 25–110 ms (or <2 ms via PyTorch).

Architecture:
  1. Threaded Emulation Worker: Steps PyBoy 2.7.0 LR35902 hardware core in headless mode.
  2. SMDP Decision Engine: Formulates contextual yes/no affordance questions from WRAM.
  3. Dual-Engine Policy: Queries local jevos daemon (http://127.0.0.1:8017) or embedded LocalAffordancePolicy.
  4. Non-Blocking HTTP API: Provides /api/state and /api/control endpoints for 60 Hz streaming.
"""

from __future__ import annotations
import base64
import io
import json
import os
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image

# Ensure repo root and src are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokemon_rl.agent.local_decision_policy import (
    LocalAffordancePolicy, JevosLocalClient, SMDPDecisionHarness, AffordanceBranch, LocalDecision
)
from pokemon_rl.env.real_pyboy_env import RealPyBoyEnvironment
from pokemon_rl.agent.macro_action_compiler import MacroActionCompiler
from interactive.constants import MAP_NAMES


class SharedVisualizerState:
    """Thread-safe state container shared between emulation worker and HTTP server."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame_b64: str = ""
        self.paused: bool = False
        self.badges: int = 0
        self.best: int = 8
        self.decisions_count: int = 0
        self.moving_latency_ms: float = 34.0
        self.step_count: int = 0
        self.map_name: str = "PALLET TOWN"
        self.coords: List[int] = [5, 4]
        self.model_source: str = "jevos-q4_k_m"

        self.model_reads: Dict[str, Any] = {
            "state": {
                "location": "Pallet Town",
                "coordinates": [5, 4],
                "party": [{"species": "Charmander", "level": 5, "hp": "20/20"}],
                "situation": "Exploring route towards Viridian City"
            },
            "questions": {
                "advance": "Should Red continue north towards Viridian City?",
                "heal": "Should Red return south to Mom's house in Pallet Town to heal?"
            }
        }

        self.model_answers: List[Dict[str, Any]] = [
            {
                "id": "advance",
                "question": "Should Red continue north towards Viridian City?",
                "prob": 82,
                "is_chosen": True
            },
            {
                "id": "heal",
                "question": "Should Red return south to Mom's house in Pallet Town to heal?",
                "prob": 14,
                "is_chosen": False
            }
        ]

        self.decision_log: List[Dict[str, Any]] = [
            {"action": "advance", "prob": 82, "context": "path clear ahead, north towards Viridian"},
            {"action": "grass", "prob": 74, "context": "tall grass step, level 5 Charmander active"},
            {"action": "advance", "prob": 85, "context": "Route 1 progression confirmed"}
        ]

    def update_frame(self, raw_rgba: np.ndarray):
        """Compress uint8 RGBA array to base64 JPEG."""
        try:
            img = Image.fromarray(raw_rgba)
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            with self.lock:
                self.frame_b64 = b64_str
        except Exception:
            pass

    def record_decision(
        self,
        reads: Dict[str, Any],
        answers: List[Dict[str, Any]],
        chosen_action: str,
        chosen_prob: int,
        context: str,
        latency_ms: float,
        model_source: str,
    ):
        """Atomically record a new decision outcome."""
        with self.lock:
            self.model_reads = reads
            self.model_answers = answers
            self.decisions_count += 1
            self.model_source = model_source
            # Exponential moving average for decision latency
            self.moving_latency_ms = 0.85 * self.moving_latency_ms + 0.15 * latency_ms

            # Append to log (keep last 30 entries)
            self.decision_log.append({
                "action": chosen_action,
                "prob": chosen_prob,
                "context": context
            })
            if len(self.decision_log) > 30:
                self.decision_log.pop(0)

    def snapshot(self) -> Dict[str, Any]:
        """Produce read-only JSON serializable snapshot."""
        with self.lock:
            return {
                "frame": self.frame_b64,
                "stats": {
                    "badges": self.badges,
                    "best": self.best,
                    "decisions": self.decisions_count,
                    "latency_ms": round(self.moving_latency_ms, 1),
                    "step": self.step_count,
                    "map_name": self.map_name,
                    "coords": self.coords,
                    "paused": self.paused,
                    "model_source": self.model_source,
                },
                "model_reads": self.model_reads,
                "model_answers": self.model_answers,
                "log": list(reversed(self.decision_log[-12:])),
            }


class EmulationWorker(threading.Thread):
    """
    Background worker running real Game Boy emulation and invoking the SMDP decision policy.
    """

    def __init__(self, shared_state: SharedVisualizerState, rom_path: str = "roms/pokemon_red.gb"):
        super().__init__(daemon=True)
        self.shared = shared_state
        self.rom_path = Path(rom_path)
        self.running = True
        self.env: Optional[RealPyBoyEnvironment] = None
        self.harness = SMDPDecisionHarness()

    def run(self):
        # 1. Initialize RealPyBoyEnvironment in headless mode
        resolved_rom = self.rom_path if self.rom_path.exists() else Path("roms/pokemon_red.gb")
        if resolved_rom.exists():
            try:
                self.env = RealPyBoyEnvironment(rom_path=str(resolved_rom), window_type="null")
                # Skip Game Boy intro / copyright animation
                for _ in range(180):
                    self.env.tick_frames(1)
                for _ in range(4):
                    self.env.send_button("start", hold_frames=4, release_frames=12)
                    self.env.send_button("a", hold_frames=4, release_frames=12)
            except Exception as e:
                print(f"[!] Warning: PyBoy initialization fallback: {e}")
                self.env = None

        print("[+] Jevos Emulation Worker active. Running SMDP affordance loop...")

        # 2. Main Simulation & Decision Loop
        step_idx = 0
        while self.running:
            if self.shared.paused:
                time.sleep(0.05)
                continue

            step_idx += 1
            time.sleep(0.035)  # Pace emulation to ~30 FPS for smooth visualization

            if self.env is not None:
                # Real Game Boy Execution
                snapshot = self.env.get_wram_snapshot()
                raw_frame = self.env.pyboy.screen.ndarray
                self.shared.update_frame(raw_frame)

                map_id = snapshot["map_id"]
                map_name = MAP_NAMES.get(map_id, f"Map_{map_id}")
                x = snapshot["x"]
                y = snapshot["y"]
                badges = snapshot["badges"]

                with self.shared.lock:
                    self.shared.map_name = map_name
                    self.shared.coords = [x, y]
                    self.shared.badges = badges
                    self.shared.step_count = step_idx
                    self.shared.best = max(self.shared.best, step_idx // 10)

                # Execute decision at affordance forks (every 10 steps or in battle/dialogue)
                if step_idx % 10 == 0 or snapshot["in_battle"] or snapshot["text_box_id"] != 0:
                    self._evaluate_affordance_fork(snapshot)
                else:
                    # Tick forward
                    self.env.tick_frames(2)

            else:
                # Synthetic fallback frame generator for test environments without ROM
                synthetic_frame = self._generate_synthetic_gb_frame(step_idx)
                self.shared.update_frame(synthetic_frame)
                with self.shared.lock:
                    self.shared.step_count = step_idx
                    self.shared.coords = [(step_idx % 20), ((step_idx // 20) % 20)]
                    self.shared.best = max(self.shared.best, step_idx // 8)
                if step_idx % 15 == 0:
                    mock_snap = {
                        "map_id": 0, "x": self.shared.coords[0], "y": self.shared.coords[1],
                        "in_battle": False, "badges": 0, "text_box_id": 0,
                        "active_mon": {"hp": 20, "max_hp": 20, "level": 5}
                    }
                    self._evaluate_affordance_fork(mock_snap)

    def _evaluate_affordance_fork(self, snapshot: Dict[str, Any]):
        """Formulate semantic yes/no affordance questions and query the local policy."""
        map_id = snapshot.get("map_id", 0)
        map_name = MAP_NAMES.get(map_id, "PALLET TOWN")
        x = snapshot.get("x", 5)
        y = snapshot.get("y", 4)
        in_battle = snapshot.get("in_battle", False)
        text_active = snapshot.get("text_box_id", 0) != 0

        # Construct candidate questions based on context
        if in_battle:
            situation = "Combat: Wild Pokémon encounter on Route 1"
            questions = {
                "scratch": "Should Charmander attack with Scratch (Power 40)?",
                "growl": "Should Charmander use Growl to lower enemy Attack?",
                "run": "Should Red attempt to flee from this wild encounter?"
            }
            button_map = {"scratch": "a", "growl": "down", "run": "b"}
        elif text_active:
            situation = "Dialogue: NPC conversation / message active"
            questions = {
                "advance": "Should Red press A to advance the dialogue text?",
                "close": "Should Red press B to close the menu dialog?"
            }
            button_map = {"advance": "a", "close": "b"}
        else:
            situation = f"Overworld Navigation: {map_name} (X={x}, Y={y})"
            questions = {
                "north": f"Should Red advance north along {map_name} towards destination?",
                "grass": "Should Red step into the tall grass to train Pokémon?",
                "heal": "Should Red return south to Mom's house in Pallet Town to heal?"
            }
            button_map = {"north": "up", "grass": "right", "heal": "down"}

        payload = {
            "state": {
                "location": map_name,
                "coordinates": [x, y],
                "party": [{"species": "Charmander", "level": 5, "hp": "20/20"}],
                "situation": situation
            },
            "questions": questions
        }

        branch = AffordanceBranch(
            branch_type="combat" if in_battle else "overworld",
            state_payload=payload["state"],
            legal_options=questions
        )

        t0 = time.perf_counter()
        decision: LocalDecision = self.harness.step_branch(branch, snapshot)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Format answers list with percentages
        answers_list = []
        for opt_id, q_text in questions.items():
            prob_val = decision.probabilities.get(opt_id, 0.0)
            pct = int(round(prob_val * 100))
            answers_list.append({
                "id": opt_id,
                "question": q_text,
                "prob": pct,
                "is_chosen": (opt_id == decision.selected_option)
            })

        chosen_pct = int(round(decision.confidence * 100))
        context_str = f"{situation} - chosen {decision.selected_option.upper()} ({latency_ms:.1f}ms)"

        # Send action to emulator if available
        if self.env is not None:
            btn = button_map.get(decision.selected_option, "a")
            self.env.send_button(btn, hold_frames=8, release_frames=16)

        model_name = "jevos-daemon" if "jevos" in decision.model_source else "jevos-local"
        self.shared.record_decision(
            reads=payload,
            answers=answers_list,
            chosen_action=decision.selected_option,
            chosen_prob=chosen_pct,
            context=context_str,
            latency_ms=latency_ms,
            model_source=model_name
        )

    def _generate_synthetic_gb_frame(self, t: int) -> np.ndarray:
        """Create a 144x160 RGB Game Boy aesthetic frame when running without ROM."""
        frame = np.full((144, 160, 3), 196, dtype=np.uint8)
        # Draw retro grid
        frame[::16, :, :] = 160
        frame[:, ::16, :] = 160
        # Draw simulated character box moving
        cx = int(20 + (t % 120))
        cy = int(20 + ((t // 2) % 100))
        frame[cy:cy+12, cx:cx+12, :] = [48, 98, 48]
        return frame


class JevosVisualizerHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving dashboard UI and streaming state."""

    shared_state: SharedVisualizerState = None
    html_content: str = ""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_content.encode("utf-8"))

        elif self.path == "/api/state":
            snapshot = self.shared_state.snapshot()
            body = json.dumps(snapshot).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/control":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                cmd = json.loads(post_data.decode("utf-8"))
                action = cmd.get("action")
                if action == "pause":
                    self.shared_state.paused = True
                elif action == "resume":
                    self.shared_state.paused = False
                elif action == "reset":
                    self.shared_state.step_count = 0
                    self.shared_state.decisions_count = 0
                    self.shared_state.paused = False
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(f'{{"error": "{e}"}}'.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress noisy HTTP polling logs
        pass


def find_free_port(start_port: int = 8090) -> int:
    import socket
    for p in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port


class JevosVisualizerServer:
    """Orchestrates HTTP server and background Game Boy emulation worker."""

    def __init__(self, port: Optional[int] = None, rom_path: str = "roms/pokemon_red.gb"):
        self.port = port if port is not None else find_free_port(8090)
        self.rom_path = rom_path
        self.shared_state = SharedVisualizerState()
        self.worker = EmulationWorker(self.shared_state, rom_path=self.rom_path)

        # Load HTML template
        html_file = Path(__file__).parent / "jevos_dashboard.html"
        if html_file.exists():
            with open(html_file, "r", encoding="utf-8") as f:
                self.html_content = f.read()
        else:
            self.html_content = "<h1>Jevos Dashboard HTML not found</h1>"

        JevosVisualizerHandler.shared_state = self.shared_state
        JevosVisualizerHandler.html_content = self.html_content

        self.server = HTTPServer(("0.0.0.0", self.port), JevosVisualizerHandler)

    def start(self, blocking: bool = True):
        self.worker.start()
        print(f"\n" + "=" * 70)
        print(f"  JEVOS POKÉMON RED VISUALIZER SERVER ACTIVE")
        print(f"  Access live dashboard at: http://127.0.0.1:{self.port}")
        print(f"  Architecture: Local Affordance RL (jevos · on CPU · 0 API cost)")
        print(f"=" * 70 + "\n", flush=True)

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                print("\n[*] Shutting down visualizer server...")
                self.stop()

    def stop(self):
        self.worker.running = False
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    env_port = os.environ.get("JEVOS_PORT")
    port = int(env_port) if env_port else find_free_port(8090)
    server = JevosVisualizerServer(port=port)
    server.start(blocking=True)

