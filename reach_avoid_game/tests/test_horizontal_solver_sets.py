"""Tests for paper horizontal Phi_h set construction."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

pytest.importorskip("odp")

from reach_avoid_game.config import GameConfig
from reach_avoid_game.solvers.grid_utils import (
    create_horizontal_game_grid,
)
from reach_avoid_game.solvers.horizontal_solver import (
    _make_attacker_target_set,
    _make_paper_horizontal_avoid_set,
    _make_paper_horizontal_reach_set,
    compute_invariant_set_Bh,
)
from reach_avoid_game.solvers.value_function_io import ValueFunctionData, load_value_function, save_value_function


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "generated_calibrated_game_params.yaml"


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


def _target_grid_point(grid, config) -> Tuple[float, float]:
    """Pick an attacker grid point inside the configured target box."""
    x_points = np.asarray(grid.grid_points[4])
    y_points = np.asarray(grid.grid_points[5])
    x_inside = x_points[
        (x_points >= config.target_region.x_min)
        & (x_points <= config.target_region.x_max)
    ]
    y_inside = y_points[
        (y_points >= config.target_region.y_min)
        & (y_points <= config.target_region.y_max)
    ]
    x_safe = x_inside[
        (x_inside > config.room.x_min + 0.5)
        & (x_inside < config.room.x_max - 0.5)
    ]
    y_safe = y_inside[
        (y_inside > config.room.y_min + 0.5)
        & (y_inside < config.room.y_max - 0.5)
    ]
    if x_safe.size:
        x_inside = x_safe
    if y_safe.size:
        y_inside = y_safe
    assert x_inside.size > 0
    assert y_inside.size > 0
    center_x = 0.5 * (config.target_region.x_min + config.target_region.x_max)
    center_y = 0.5 * (config.target_region.y_min + config.target_region.y_max)
    x = x_inside[np.argmin(np.abs(x_inside - center_x))]
    y = y_inside[np.argmin(np.abs(y_inside - center_y))]
    return float(x), float(y)


def test_attacker_target_set_negative_inside_target(game_grid, config):
    target = _make_attacker_target_set(game_grid, config)
    target_x, target_y = _target_grid_point(game_grid, config)
    inside = _game_idx(game_grid, 22.5, 12.5, 0.0, 0.0, target_x, target_y)
    outside = _game_idx(game_grid, 22.5, 12.5, 0.0, 0.0, 5.0, 12.5)

    assert target[inside] <= 0.0
    assert target[outside] > 0.0


def test_paper_reach_set_includes_attacker_target(game_grid, config):
    reach = _make_paper_horizontal_reach_set(game_grid, config)
    target_x, target_y = _target_grid_point(game_grid, config)
    idx = _game_idx(game_grid, 22.5, 12.5, 0.0, 0.0, target_x, target_y)

    assert reach[idx] <= 0.0


def test_paper_reach_set_includes_defender_obstacle(game_grid, config):
    reach = _make_paper_horizontal_reach_set(game_grid, config)
    idx = _game_idx(game_grid, 17.0, 12.5, 0.0, 0.0, 5.0, 12.5)

    assert reach[idx] <= 0.0


def test_paper_avoid_set_uses_bh_only_before_target(game_grid, config):
    values = np.zeros(tuple(game_grid.pts_each_dim))
    v_h_t_data = ValueFunctionData(
        values=values,
        grid_min=game_grid.min,
        grid_max=game_grid.max,
        grid_shape=values.shape,
    )

    avoid = _make_paper_horizontal_avoid_set(
        game_grid, config, v_h_t_data, config.capture.d_h,
    )
    target_x, target_y = _target_grid_point(game_grid, config)
    before_target = _game_idx(game_grid, 5.0, 12.5, 0.0, 0.0, 5.0, 12.5)
    in_target = _game_idx(game_grid, target_x, target_y, 0.0, 0.0, target_x, target_y)

    assert avoid[before_target] <= 0.0
    assert avoid[in_target] > 0.0


def test_paper_avoid_set_rejects_4d_tracking_value(game_grid, config):
    values = np.zeros((3, 3, 3, 3))
    v_h_t_data = ValueFunctionData(
        values=values,
        grid_min=np.array([-1.0, -1.0, -1.0, -1.0]),
        grid_max=np.array([1.0, 1.0, 1.0, 1.0]),
        grid_shape=values.shape,
    )

    with pytest.raises(ValueError, match="requires an obstacle-aware 6D"):
        _make_paper_horizontal_avoid_set(
            game_grid, config, v_h_t_data, config.capture.d_h,
        )


def test_compute_bh_uses_requested_threshold_without_expansion(tmp_path):
    values = np.array([
        [[[0.5]], [[4.0]]],
        [[[4.0]], [[4.0]]],
    ])
    vf_data = ValueFunctionData(
        values=values,
        grid_min=np.array([0.0, 0.0, 0.0, 0.0]),
        grid_max=np.array([1.0, 1.0, 0.0, 0.0]),
        grid_shape=values.shape,
    )
    source = tmp_path / "V_h_T.npz"
    save_value_function(source, vf_data)

    output = compute_invariant_set_Bh(source, d_h=3.0, output_dir=tmp_path)
    b_h = load_value_function(output)

    np.testing.assert_array_equal(b_h.values, (values <= 3.0).astype(float))
    assert "d_h_effective" not in b_h.params


def test_compute_bh_raises_when_requested_threshold_is_empty(tmp_path):
    values = np.full((2, 2, 1, 1), 4.0)
    vf_data = ValueFunctionData(
        values=values,
        grid_min=np.array([0.0, 0.0, 0.0, 0.0]),
        grid_max=np.array([1.0, 1.0, 0.0, 0.0]),
        grid_shape=values.shape,
    )
    source = tmp_path / "V_h_T.npz"
    save_value_function(source, vf_data)

    with pytest.raises(ValueError, match="threshold will not be expanded"):
        compute_invariant_set_Bh(source, d_h=3.0, output_dir=tmp_path)
