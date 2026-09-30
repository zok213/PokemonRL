import os
from os.path import exists
from pathlib import Path
import uuid
import time
import glob
from red_gym_env_v2 import RedGymEnv
from stable_baselines3 import A2C, PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        #env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def get_most_recent_zip_with_age(folder_path):
    # Get all zip files in the folder
    zip_files = glob.glob(os.path.join(folder_path, "*.zip"))
    
    if not zip_files:
        return None, None  # Return None if no zip files are found
    
    # Find the most recently modified zip file
    most_recent_zip = max(zip_files, key=os.path.getmtime)
    
    # Calculate how old the file is in hours
    current_time = time.time()
    modification_time = os.path.getmtime(most_recent_zip)
    age_in_hours = (current_time - modification_time) / 3600  # Convert seconds to hours
    
    return most_recent_zip, age_in_hours

if __name__ == '__main__':
    v2_dir = Path(__file__).resolve().parent
    repo_root = v2_dir.parent.parent.parent

    # Resolve ROM path
    rom_candidates = [
        v2_dir.parent / "PokemonRed.gb",
        repo_root / "roms" / "pokemon_red.gb",
        repo_root / "PokemonRed.gb",
    ]
    resolved_rom = next((str(p) for p in rom_candidates if p.exists()), None)
    if not resolved_rom:
        print("[!] Error: No Pokemon Red ROM found!")
        sys.exit(1)

    # Resolve init state
    state_candidates = [
        v2_dir.parent / "init.state",
        v2_dir.parent / "has_pokedex_nballs.state",
        repo_root / "saves" / "has_pokedex_nballs.state",
    ]
    resolved_state = next((str(p) for p in state_candidates if p.exists()), None)

    # Resolve checkpoint
    runs_dir = v2_dir / "runs"
    most_recent_checkpoint, time_since = get_most_recent_zip_with_age(str(runs_dir))
    if not most_recent_checkpoint:
        print("[!] Error: No checkpoint found in v2/runs!")
        sys.exit(1)

    file_name = most_recent_checkpoint
    print(f"[*] Using V2 Real RL Pretrained Checkpoint: {Path(file_name).name} ({time_since:.1f} hours old)")
    print(f"[*] ROM: {resolved_rom}")
    print(f"[*] Initial State: {resolved_state}")

    sess_path = v2_dir / f"session_{str(uuid.uuid4())[:8]}"
    ep_length = 2**23

    env_config = {
        'headless': False,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': resolved_state,
        'max_steps': ep_length, 
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,
        'gb_path': resolved_rom,
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'extra_buttons': False
    }

    env = make_env(0, env_config)()

    print('\n[*] Loading PPO policy weights into neural network...')
    model = PPO.load(
        file_name,
        env=env,
        custom_objects={'lr_schedule': 0, 'clip_range': 0, 'tensorboard_log': None}
    )
    print('[+] Real PPO Policy Loaded! Launching SDL2 Game Boy Window...')

    toggle_file = v2_dir / "agent_enabled.txt"
    if not toggle_file.exists():
        toggle_file.write_text("yes\n")

    obs, info = env.reset()
    step_count = 0

    print("=" * 70)
    print("  POKEMON RED — BASELINE V2 REAL RL INTERACTIVE RUNNER")
    print("  Algorithm: Pure PPO MultiInputPolicy (26.2M steps trained)")
    print("  Environment: Raw Gym (NO Cheats, NO Level 100, NO Injected Flags)")
    print("  Controls: [M] Toggle in agent_enabled.txt | Close window to exit")
    print("=" * 70, flush=True)

    while True:
        try:
            with open(toggle_file, "r") as f:
                agent_enabled = f.readlines()[0].strip().lower().startswith("yes")
        except Exception:
            agent_enabled = True

        if agent_enabled:
            action, _states = model.predict(obs, deterministic=False)
            obs, rewards, terminated, truncated, info = env.step(action)
            step_count += 1
            if step_count % 10 == 0:
                cur_x = env.read_m(0xD362)
                cur_y = env.read_m(0xD361)
                cur_map = env.read_m(0xD35E)
                hp_val = env.read_hp(0xD16C)
                max_hp = env.read_hp(0xD18D)
                lvl = env.read_m(0xD18C)
                print(f"Step {step_count:5d} | Map={cur_map:2d} Pos=({cur_x:2d},{cur_y:2d}) | HP={hp_val}/{max_hp} Lv={lvl} | Act={action} Rew={rewards:+.3f}", flush=True)
        else:
            env.pyboy.tick(1, True)
            obs = env._get_obs()
            truncated = env.step_count >= env.max_steps - 1

        env.render()
        if truncated:
            break
    env.close()



