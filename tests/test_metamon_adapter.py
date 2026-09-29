"""
test_metamon_adapter.py — Unit Tests for Metamon Neural Battle Adapter
======================================================================
Verifies checkpoint loading, WRAM battle feature extraction, and tactical move selection.
"""

from pathlib import Path
import pytest
from pokemon_rl.combat.metamon_adapter import MetamonBattleAdapter
from pokemon_rl.env.wram_map import Action


@pytest.fixture
def metamon_checkpoint_path() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    path = (
        repo_root
        / "external"
        / "metamon"
        / "metamon"
        / "baselines"
        / "model_based"
        / "pretrained_models"
        / "replays_v2_small_trial1_BEST.pt"
    )
    return path


def test_metamon_adapter_initialization(metamon_checkpoint_path: Path):
    adapter = MetamonBattleAdapter(checkpoint_path=metamon_checkpoint_path)
    assert adapter is not None


def test_metamon_adapter_wram_feature_extraction():
    adapter = MetamonBattleAdapter()
    raw_wram = bytearray(8192)

    # Set player HP = 120/120 (0xD015, 0xD023)
    off_p_hp = 0xD015 - 0xC000
    off_p_max = 0xD023 - 0xC000
    raw_wram[off_p_hp] = 0x00
    raw_wram[off_p_hp + 1] = 120
    raw_wram[off_p_max] = 0x00
    raw_wram[off_p_max + 1] = 120

    # Set enemy HP = 40/80 (0xCFE6, 0xCFF4)
    off_e_hp = 0xCFE6 - 0xC000
    off_e_max = 0xCFF4 - 0xC000
    raw_wram[off_e_hp] = 0x00
    raw_wram[off_e_hp + 1] = 40
    raw_wram[off_e_max] = 0x00
    raw_wram[off_e_max + 1] = 80

    feats = adapter.extract_battle_features(bytes(raw_wram))
    assert feats["hp_player"] == 120
    assert feats["max_hp_player"] == 120
    assert feats["hp_ratio_player"] == 1.0
    assert feats["hp_enemy"] == 40
    assert feats["max_hp_enemy"] == 80
    assert feats["hp_ratio_enemy"] == 0.5


def test_metamon_adapter_decision_and_menu_path():
    adapter = MetamonBattleAdapter()
    battle_state = {
        "opponent_type": "FIRE",
        "user_type": "WATER",
        "user_base_speed": 43,  # Squirtle
        "available_moves": [
            {"name": "Tackle", "power": 40, "accuracy": 1.0, "type": "NORMAL", "pp": 35},
            {"name": "Water Gun", "power": 40, "accuracy": 1.0, "type": "WATER", "pp": 25},
        ]
    }
    slot = adapter.select_battle_move(battle_state)
    assert slot == 1  # Water Gun is slot 1

    menu_actions = adapter.plan_menu_navigation(slot)
    assert menu_actions == [Action.RIGHT, Action.A]
