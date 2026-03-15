"""Tests for GameConfig loading from YAML."""

from pathlib import Path

from reach_avoid_game.config import GameConfig

YAML_PATH = Path(__file__).resolve().parents[2] / "config" / "game_params.yaml"


def test_load_config():
    config = GameConfig.from_yaml(str(YAML_PATH))
    assert config.room.x_max == 45.0
    assert config.defender.max_speed_horizontal == 6.0
    assert config.attacker.max_speed_horizontal == 3.0
    assert config.capture.d_h == 3.0
    assert config.capture.d_z == 1.0


def test_config_target_region():
    config = GameConfig.from_yaml(str(YAML_PATH))
    assert config.target_region.type == "box"
    assert config.target_region.x_min == 38.0


def test_config_obstacles():
    config = GameConfig.from_yaml(str(YAML_PATH))
    assert len(config.obstacles) == 1
    assert config.obstacles[0].type == "box"
    assert config.obstacles[0].x_min == 15.0


def test_config_grid():
    config = GameConfig.from_yaml(str(YAML_PATH))
    # Dev preset should override vertical grid points
    assert config.grid.vertical.z_rel_points == 51
    assert config.grid.solver.time_horizon == 15.0


def test_config_defender_params():
    config = GameConfig.from_yaml(str(YAML_PATH))
    assert config.defender.k_z == 1.5
    assert config.defender.k_x == 0.7
    assert config.defender.k_y == 0.7


def test_config_grid_preset():
    config = GameConfig.from_yaml(str(YAML_PATH))
    assert config.grid_preset == "dev"
    assert "dev" in config.grid_presets
    assert "paper" in config.grid_presets
