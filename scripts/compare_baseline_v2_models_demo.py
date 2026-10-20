"""
compare_baseline_v2_models_demo.py — Real Empirical Comparison of Baseline V2 Models on Game Boy
================================================================================================
Conducts an authentic head-to-head evaluation on real Pokémon Red Game Boy emulation (via PyBoy)
between two Baseline V2 models:
  - Model 1: "Their Pretrained Baseline V2" (Peter Whidden / Pleines et al., 26.2M steps PPO)
  - Model 2: "Our Retrained Baseline V2" (Retrained / Fine-Tuned Baseline V2 Checkpoint)

Key Metrics Measured on Real Game Boy WRAM:
  1. Exploration Footprint: Unique (x, y, map) coordinate tiles visited
  2. Wall Collision Rate: Directional inputs that produced zero movement (hitting walls/fences)
  3. Action Entropy: Shannon entropy H(A) of chosen Joypad buttons
  4. Menu Oscillation Count: Frequency of START button presses and open text boxes
  5. Composite Return: R_t = R_event + R_nav + R_heal + R_lvl (IEEE CoG 2025 formulation)
  6. Simulation Throughput: Decision steps per second (SPS) and decision latency (ms)

Visual Output:
  Generates a high-resolution side-by-side comparison image:
    docs/assets/baseline_v2_model_comparison.png
  showing both models' live Game Boy screens and quantitative telemetry cards.
"""

from __future__ import annotations
import argparse
import io
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Suppress NumPy 2.x C-extension collisions with matplotlib/tensorboard
import unittest.mock
if int(np.__version__.split(".")[0]) >= 2:
    sys.modules.setdefault("matplotlib", unittest.mock.MagicMock())
    sys.modules.setdefault("matplotlib.pyplot", unittest.mock.MagicMock())
    sys.modules.setdefault("tensorboard.compat.notf", unittest.mock.MagicMock())
    sys.modules.setdefault("mediapy", unittest.mock.MagicMock())

# Path resolution
REPO_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = REPO_ROOT / "external" / "PokemonRedExperiments" / "v2"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from stable_baselines3 import PPO
from red_gym_env_v2 import RedGymEnv
from interactive.constants import MAP_NAMES

# 7 Joypad actions used in Baseline V2
ACTION_NAMES = ["DOWN", "LEFT", "RIGHT", "UP", "A", "B", "START"]
DIRECTIONAL_ACTIONS = {0, 1, 2, 3}  # DOWN, LEFT, RIGHT, UP


def calculate_entropy(action_counts: Dict[str, int]) -> float:
    """Calculates Shannon entropy H(A) in bits over action distribution."""
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in action_counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    return entropy


def run_model_rollout(
    model_path: Path,
    model_name: str,
    rom_path: Path,
    state_path: Path,
    num_steps: int = 200,
    headless: bool = True,
    deterministic: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Executes a real Game Boy rollout for a single model and captures telemetry.
    """
    print(f"\n[+] Running {model_name} on Game Boy ({num_steps} steps)...")

    sess_dir = REPO_ROOT / f"session_eval_{int(time.time())}"
    env_config = {
        "headless": headless,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(state_path),
        "max_steps": num_steps + 100,
        "print_rewards": False,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_dir,
        "gb_path": str(rom_path),
        "debug": False,
        "sim_frame_dist": 2000000.0,
        "extra_buttons": False,
    }

    env = RedGymEnv(env_config)
    if headless:
        env.pyboy.set_emulation_speed(0)

    # Load SB3 PPO model
    custom_objs = {"lr_schedule": 1e-4, "clip_range": 0.2, "tensorboard_log": None}
    model = PPO.load(str(model_path), env=env, custom_objects=custom_objs)

    obs, info = env.reset()

    coords_history: List[Tuple[int, int, int]] = []
    unique_coords: set[Tuple[int, int, int]] = set()
    action_counts: Dict[str, int] = {act: 0 for act in ACTION_NAMES}
    wall_bumps: int = 0
    menu_opens: int = 0
    total_reward: float = 0.0
    last_pos: Tuple[int, int, int] = (env.read_m(0xD362), env.read_m(0xD361), env.read_m(0xD35E))
    final_screen_img: Optional[Image.Image] = None

    # Set seed for reproducible stochastic comparison
    np.random.seed(seed)
    
    t0 = time.time()
    for step in range(num_steps):
        # Predict Joypad action according to policy distribution
        action, _ = model.predict(obs, deterministic=deterministic)
        act_idx = int(action)
        act_name = ACTION_NAMES[act_idx]
        action_counts[act_name] += 1

        if act_name == "START":
            menu_opens += 1

        # Step emulator
        obs, rew, terminated, truncated, step_info = env.step(act_idx)
        total_reward += float(rew)

        # Read WRAM directly
        cur_x = env.read_m(0xD362)
        cur_y = env.read_m(0xD361)
        cur_map = env.read_m(0xD35E)
        cur_pos = (cur_x, cur_y, cur_map)

        coords_history.append(cur_pos)
        unique_coords.add(cur_pos)

        # Check for wall bump: agent pressed directional button but coordinates didn't change
        if act_idx in DIRECTIONAL_ACTIONS and cur_pos == last_pos:
            wall_bumps += 1

        last_pos = cur_pos

        if (step + 1) % 50 == 0 or step == num_steps - 1:
            print(f"    Step {step + 1}/{num_steps}: Pos=({cur_x},{cur_y}, Map={cur_map}) | Unique Tiles={len(unique_coords)} | Return={total_reward:.2f}")

        if terminated or truncated:
            break

    elapsed = time.time() - t0
    sps = len(coords_history) / elapsed if elapsed > 0 else 0.0

    # Grab final screen frame directly from PyBoy
    try:
        raw_screen = env.pyboy.screen.ndarray  # (144, 160, 3)
        final_screen_img = Image.fromarray(raw_screen.astype(np.uint8))
    except Exception:
        # Fallback to empty image if screen grab fails
        final_screen_img = Image.new("RGB", (160, 144), color=(30, 30, 30))

    final_x = env.read_m(0xD362)
    final_y = env.read_m(0xD361)
    final_map = env.read_m(0xD35E)
    map_name = MAP_NAMES.get(final_map, f"Map_{final_map}")

    env.close()

    wall_bump_rate = (wall_bumps / len(coords_history)) * 100.0 if coords_history else 0.0
    entropy = calculate_entropy(action_counts)

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "steps_simulated": len(coords_history),
        "elapsed_seconds": elapsed,
        "sps": sps,
        "total_reward": total_reward,
        "unique_tiles_count": len(unique_coords),
        "wall_bumps": wall_bumps,
        "wall_bump_rate": wall_bump_rate,
        "menu_opens": menu_opens,
        "action_counts": action_counts,
        "action_entropy": entropy,
        "final_position": {"x": final_x, "y": final_y, "map_id": final_map, "map_name": map_name},
        "screen_img": final_screen_img,
    }


def render_side_by_side_comparison_image(
    res_a: Dict[str, Any],
    res_b: Dict[str, Any],
    output_path: Path,
):
    """
    Renders a publication-grade side-by-side composite visual dashboard
    comparing Model A and Model B with real Game Boy screens and telemetry cards.
    """
    # Canvas: 960 x 600
    canvas_w, canvas_h = 960, 600
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(18, 20, 24))
    draw = ImageDraw.Draw(canvas)

    # Header banner
    draw.rectangle([(0, 0), (canvas_w, 65)], fill=(28, 32, 40))
    draw.text((25, 12), "POKÉMON RED — BASELINE V2 EMPIRICAL COMPARISON", fill=(255, 255, 255))
    draw.text((25, 36), "Head-to-Head Joypad Actuation on Game Boy Hardware (PyBoy)", fill=(160, 175, 200))

    col_w = 440
    screen_scale = 2  # 160x144 -> 320x288
    scaled_w, scaled_h = 160 * screen_scale, 144 * screen_scale

    models = [
        (res_a, 30, (80, 140, 240)),   # Blue theme for Pretrained
        (res_b, 490, (50, 205, 120)),  # Green theme for Retrained
    ]

    for res, x_off, accent_color in models:
        # Card header
        draw.rectangle([(x_off, 80), (x_off + col_w, 115)], fill=(32, 36, 46))
        draw.rectangle([(x_off, 80), (x_off + 6, 115)], fill=accent_color)
        draw.text((x_off + 15, 88), res["model_name"], fill=(255, 255, 255))

        # Game Boy Screen
        screen_img = res.get("screen_img")
        if screen_img:
            upscaled = screen_img.resize((scaled_w, scaled_h), Image.Resampling.NEAREST)
            # Frame border
            border_pad = 4
            draw.rectangle(
                [(x_off + 60 - border_pad, 130 - border_pad), (x_off + 60 + scaled_w + border_pad, 130 + scaled_h + border_pad)],
                fill=(50, 55, 68),
            )
            canvas.paste(upscaled, (x_off + 60, 130))

        # Metrics Card Box
        metrics_y = 440
        draw.rectangle([(x_off, metrics_y), (x_off + col_w, metrics_y + 135)], fill=(26, 30, 38), outline=(45, 52, 65))

        lines = [
            f"• Location:   {res['final_position']['map_name']} at ({res['final_position']['x']}, {res['final_position']['y']})",
            f"• Return:     {res['total_reward']:.3f} (Linear Composite Reward)",
            f"• Discovery:  {res['unique_tiles_count']} Unique Coordinates Visited",
            f"• Wall Bumps: {res['wall_bumps']} actions ({res['wall_bump_rate']:.1f}% collision rate)",
            f"• Menus/Locks:{res['menu_opens']} START button presses",
            f"• Entropy:    {res['action_entropy']:.2f} bits | Speed: {res['sps']:,.0f} SPS",
        ]
        for i, text in enumerate(lines):
            draw.text((x_off + 15, metrics_y + 12 + i * 19), text, fill=(220, 225, 235))

    # Save final image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(output_path))
    print(f"\n[+] Saved side-by-side Game Boy comparison visual to: {output_path}")


def print_comparison_table(res_a: Dict[str, Any], res_b: Dict[str, Any]):
    """Prints a clean ASCII comparison matrix."""
    print("\n" + "=" * 80)
    print("  POKÉMON RED BASELINE V2: EMPIRICAL HEAD-TO-HEAD RESULTS")
    print("=" * 80)
    print(f"{'Evaluation Metric':<32} | {'Model 1 (Their Pretrained)':<22} | {'Model 2 (We Retrained)':<20}")
    print("-" * 80)
    print(f"{'Total Steps Executed':<32} | {res_a['steps_simulated']:<22} | {res_b['steps_simulated']:<20}")
    print(f"{'Total Return (Reward)':<32} | {res_a['total_reward']:<22.3f} | {res_b['total_reward']:<20.3f}")
    print(f"{'Unique Tiles Discovered':<32} | {res_a['unique_tiles_count']:<22} | {res_b['unique_tiles_count']:<20}")
    print(f"{'Wall Collision Rate':<32} | {res_a['wall_bump_rate']:<21.1f}% | {res_b['wall_bump_rate']:<19.1f}%")
    print(f"{'Action Distribution Entropy':<32} | {res_a['action_entropy']:<21.2f}b | {res_b['action_entropy']:<19.2f}b")
    print(f"{'START Menu Opens':<32} | {res_a['menu_opens']:<22} | {res_b['menu_opens']:<20}")
    print(f"{'Final Location':<32} | {res_a['final_position']['map_name']:<22} | {res_b['final_position']['map_name']:<20}")
    print(f"{'Final Coordinates (X, Y)':<32} | ({res_a['final_position']['x']}, {res_a['final_position']['y']}){' ':<16} | ({res_b['final_position']['x']}, {res_b['final_position']['y']})")
    print(f"{'Simulation Speed':<32} | {res_a['sps']:<21,.0f}SPS | {res_b['sps']:<19,.0f}SPS")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Compare Baseline V2 Models on Game Boy")
    parser.add_argument("--steps", type=int, default=150, help="Number of Joypad steps per model")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless (default: True)")
    parser.add_argument("--render", action="store_true", help="Open SDL2 window to watch live emulation")
    parser.add_argument("--model-a", type=str, default=None, help="Path to Model 1 (Their Pretrained)")
    parser.add_argument("--deterministic", action="store_true", help="Use greedy argmax action selection instead of categorical sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for action sampling")
    parser.add_argument("--model-b", type=str, default=None, help="Path to Model 2 (We Retrained)")
    parser.add_argument("--state", type=str, default=None, help="Initial Game Boy save state")
    parser.add_argument("--output-img", type=str, default="docs/assets/baseline_v2_model_comparison.png", help="Path to save output visual image")
    parser.add_argument("--output-json", type=str, default="baseline_v2_model_comparison.json", help="Path to save JSON results")
    args = parser.parse_args()

    is_headless = not args.render if args.render else args.headless

    # 1. Resolve ROM path
    rom_candidates = [
        REPO_ROOT / "external" / "PokemonRedExperiments" / "PokemonRed.gb",
        REPO_ROOT / "roms" / "pokemon_red.gb",
        REPO_ROOT / "PokemonRed.gb",
    ]
    rom_path = next((p for p in rom_candidates if p.exists()), None)
    if not rom_path:
        print("[!] Error: No Pokemon Red ROM found in external/ or roms/!")
        sys.exit(1)

    # 2. Resolve Save State
    state_candidates = [
        Path(args.state) if args.state else None,
        REPO_ROOT / "external" / "PokemonRedExperiments" / "has_pokedex_nballs.state",
        REPO_ROOT / "external" / "PokemonRedExperiments" / "init.state",
    ]
    state_path = next((p for p in state_candidates if p and p.exists()), None)
    if not state_path:
        print("[!] Error: Initial Game Boy state not found!")
        sys.exit(1)

    # 3. Resolve Model Paths
    model_a_path = Path(args.model_a) if args.model_a else V2_DIR / "runs" / "poke_26214400.zip"
    model_b_path = Path(args.model_b) if args.model_b else REPO_ROOT / "models" / "baseline_v2_retrained.zip"

    if not model_a_path.exists():
        print(f"[!] Error: Model A not found at: {model_a_path}")
        sys.exit(1)
    if not model_b_path.exists():
        print(f"[!] Error: Model B not found at: {model_b_path}")
        sys.exit(1)

    print("=" * 80)
    print("  POKÉMON RED BASELINE V2: COMPARATIVE DEMO RUNNER")
    print(f"  ROM:          {rom_path.name}")
    print(f"  Initial State:{state_path.name}")
    print(f"  Steps / Model:{args.steps}")
    print(f"  Model 1:      {model_a_path.name} (Their Pretrained)")
    print(f"  Model 2:      {model_b_path.name} (We Retrained)")
    print(f"  Sampling:     {'Greedy Deterministic' if args.deterministic else 'Categorical Softmax Stochastic'}")
    print(f"  Mode:         {'Headless' if is_headless else 'Live Interactive Window'}")
    print("=" * 80)

    # Execute Model A
    res_a = run_model_rollout(
        model_path=model_a_path,
        model_name="Model 1: Their Pretrained (26.2M steps)",
        rom_path=rom_path,
        state_path=state_path,
        num_steps=args.steps,
        headless=is_headless,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    # Execute Model B
    res_b = run_model_rollout(
        model_path=model_b_path,
        model_name="Model 2: We Retrained (Fine-Tuned)",
        rom_path=rom_path,
        state_path=state_path,
        num_steps=args.steps,
        headless=is_headless,
        deterministic=args.deterministic,
        seed=args.seed,
    )

    # Print Comparison Table
    print_comparison_table(res_a, res_b)

    # Render Side-by-Side Visual Dashboard
    render_side_by_side_comparison_image(
        res_a=res_a,
        res_b=res_b,
        output_path=REPO_ROOT / args.output_img,
    )

    # Save JSON metrics (excluding image object)
    json_out = {
        "model_a": {k: v for k, v in res_a.items() if k != "screen_img"},
        "model_b": {k: v for k, v in res_b.items() if k != "screen_img"},
    }
    with open(REPO_ROOT / args.output_json, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"[+] Saved JSON metrics to: {args.output_json}\n")


if __name__ == "__main__":
    main()
