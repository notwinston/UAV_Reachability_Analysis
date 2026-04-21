"""Regression tests for paper-figure attacker/defender behavior policies."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from reach_avoid_game.config import GameConfig


def _load_numerical_sim_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "numerical_sim.py"
    spec = importlib.util.spec_from_file_location("numerical_sim", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_attacker_target_policy_points_toward_paper_target():
    sim = _load_numerical_sim_module()
    config = GameConfig.from_yaml(Path(__file__).resolve().parents[2] / "config" / "game_params.yaml")

    d_x, d_y = sim._extract_target_reaching_disturbance(config, x_a=5.0, y_a=20.0, u_a_h=3.0)

    assert d_x > 0.0
    assert d_y < 0.0
    assert math.hypot(d_x, d_y) == pytest.approx(3.0)


def test_defender_rejects_diagonal_hj_command_when_pursuit_closes_better():
    sim = _load_numerical_sim_module()

    # Launch geometry: same x, attacker above defender in y. A coarse HJ command
    # of (4.24, 4.24) does close the gap, but much worse than straight pursuit.
    x_rel = 0.0
    y_rel = -7.5
    hj_x, hj_y = 4.24, 4.24
    pid_x, pid_y = 0.0, 6.0

    assert not sim._horizontal_command_is_useful(hj_x, hj_y, pid_x, pid_y, x_rel, y_rel)


def test_defender_accepts_hj_command_aligned_with_pursuit():
    sim = _load_numerical_sim_module()

    x_rel = 0.0
    y_rel = -7.5
    hj_x, hj_y = 0.0, 6.0
    pid_x, pid_y = 0.0, 6.0

    assert sim._horizontal_command_is_useful(hj_x, hj_y, pid_x, pid_y, x_rel, y_rel)


def test_combined_sim_stops_at_first_terminal_event():
    sim = _load_numerical_sim_module()
    config = GameConfig.from_yaml(Path(__file__).resolve().parents[2] / "config" / "game_params.yaml")

    result = sim.run_combined_sim(
        config,
        str(Path(__file__).resolve().parents[2] / "data" / "value_functions"),
        dt=0.01,
        T=40.0,
        initial_defender_pos=[32.0, 22.0, 7.0],
        initial_attacker_pos=[3.0, 2.0, 4.0],
    )

    assert result["terminal_time"] is not None
    assert len(result["x_d"]) == len(result["u_x"]) + 1
    assert result["T"] == pytest.approx(len(result["u_x"]) * result["dt"])
    assert len(result["u_x"]) < int(40.0 / 0.01)
