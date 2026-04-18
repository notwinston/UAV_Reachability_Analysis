"""Tests for paper horizontal Phi_h set construction."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("odp")

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.grid_utils import (
    create_horizontal_game_grid,
    create_horizontal_relative_grid,
)
from reach_avoid_game.solvers.horizontal_solver import (
    _make_attacker_target_set,
    _make_paper_horizontal_avoid_set,
    _make_paper_horizontal_reach_set,
)
from reach_avoid_game.solvers.value_function_io import ValueFunctionData


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "game_params.yaml"


@pytest.fixture(scope="module")
def config():
    return GameConfig.from_yaml(CONFIG_PATH)


@pytest.fixture(scope="module")
def game_grid(config):
    return create_horizontal_game_grid(config)


def _nearest_idx(grid, dim: int, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(grid.grid_points[dim]) - value)))


def _game_idx(grid, x_d, y_d, vx_d, vy_d, x_a, y_a):
    return (
        _nearest_idx(grid, 0, x_d),
        _nearest_idx(grid, 1, y_d),
        _nearest_idx(grid, 2, vx_d),
        _nearest_idx(grid, 3, vy_d),
        _nearest_idx(grid, 4, x_a),
        _nearest_idx(grid, 5, y_a),
    )


def test_attacker_target_set_negative_inside_target(game_grid, config):
    target = _make_attacker_target_set(game_grid, config)
    inside = _game_idx(game_grid, 22.5, 12.5, 0.0, 0.0, 39.0, 12.5)
    outside = _game_idx(game_grid, 22.5, 12.5, 0.0, 0.0, 5.0, 12.5)

    assert target[inside] <= 0.0
    assert target[outside] > 0.0


def test_paper_reach_set_includes_attacker_target(game_grid, config):
    reach = _make_paper_horizontal_reach_set(game_grid, config)
    idx = _game_idx(game_grid, 22.5, 12.5, 0.0, 0.0, 39.0, 12.5)

    assert reach[idx] <= 0.0


def test_paper_reach_set_includes_defender_obstacle(game_grid, config):
    reach = _make_paper_horizontal_reach_set(game_grid, config)
    idx = _game_idx(game_grid, 17.0, 12.5, 0.0, 0.0, 5.0, 12.5)

    assert reach[idx] <= 0.0


def test_paper_avoid_set_uses_bh_only_before_target(game_grid, config):
    rel_grid = create_horizontal_relative_grid(config)
    values = np.full(tuple(rel_grid.pts_each_dim), config.capture.d_h + 1.0)
    center = tuple(int(n // 2) for n in rel_grid.pts_each_dim)
    values[center] = 0.0
    v_h_t_data = ValueFunctionData(
        values=values,
        grid_min=rel_grid.min,
        grid_max=rel_grid.max,
        grid_shape=values.shape,
    )

    avoid = _make_paper_horizontal_avoid_set(
        game_grid, config, v_h_t_data, config.capture.d_h,
    )
    before_target = _game_idx(game_grid, 5.0, 12.5, 0.0, 0.0, 5.0, 12.5)
    in_target = _game_idx(game_grid, 39.0, 12.5, 0.0, 0.0, 39.0, 12.5)

    assert avoid[before_target] <= 0.0
    assert avoid[in_target] > 0.0
