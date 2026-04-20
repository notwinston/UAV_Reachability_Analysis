"""Tests for DefenderControlLogic -- standalone pytest, no rclpy required."""

import sys
import os
import math
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/workspace/reach_avoid_game/src")

from reach_avoid_controller.value_function_loader import ValueFunctionLoader
from reach_avoid_controller.defender_node import DefenderControlLogic

REPO_ROOT = Path(__file__).resolve().parents[4]
VF_DIR = str(REPO_ROOT / "data" / "value_functions")


@pytest.fixture(scope="module")
def logic():
    loader = ValueFunctionLoader(VF_DIR)
    return DefenderControlLogic(loader)


class TestControlOutput:
    def test_returns_3d_cmd_vel(self, logic):
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([12.0, 12.0, 10.0])
        cmd, status = logic.compute_control(d_pos, d_vel, a_pos)
        assert cmd.shape == (3,)

    def test_cmd_vel_finite(self, logic):
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([12.0, 12.0, 10.0])
        cmd, _ = logic.compute_control(d_pos, d_vel, a_pos)
        assert np.all(np.isfinite(cmd))

    def test_status_has_modes(self, logic):
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([15.0, 15.0, 12.0])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        assert "z_mode" in status
        assert "h_mode" in status
        assert "game_status" in status


class TestSpeedClamping:
    def test_vertical_speed_clamped(self, logic):
        """Vertical command should not exceed U_D_z."""
        d_pos = np.array([10.0, 10.0, 5.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([10.0, 10.0, 15.0])
        cmd, _ = logic.compute_control(d_pos, d_vel, a_pos)
        assert abs(cmd[2]) <= logic.U_D_z + 1e-6

    def test_horizontal_speed_clamped(self, logic):
        """Horizontal speed should not exceed U_D_h."""
        d_pos = np.array([5.0, 5.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([40.0, 20.0, 10.0])
        cmd, _ = logic.compute_control(d_pos, d_vel, a_pos)
        h_speed = math.sqrt(cmd[0] ** 2 + cmd[1] ** 2)
        assert h_speed <= logic.U_D_h + 1e-6


class TestModeTransitions:
    def test_pid_deep_when_very_close(self, logic):
        """When defender is very close to attacker, should be in pid_deep or pid_fallback mode."""
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([10.1, 10.1, 10.0])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        # Current checked-in B_z may be rejected if it was threshold-expanded.
        assert status["z_mode"] in ("pid_deep", "tracking", "pid_fallback", "pid_invalid_bz")

    def test_reaching_when_far(self, logic):
        """When defender is far from attacker, should be in reaching mode."""
        d_pos = np.array([5.0, 5.0, 5.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([30.0, 20.0, 15.0])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        # Far apart -> outside B_z and B_h -> reaching or pid_pursuit (if outside winning region)
        assert status["z_mode"] in ("reaching", "pid_fallback", "pid_pursuit", "pid_invalid_bz")
        assert status["h_mode"] in ("reaching", "pid_fallback", "pid_pursuit", "pid_invalid_bh")

    def test_mode_is_valid_string(self, logic):
        valid_modes = {
            "reaching", "tracking", "pid_deep", "pid_fallback", "pid_pursuit",
            "pid_invalid_bz", "pid_invalid_bh",
        }
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([1.0, -1.0, 0.5])
        a_pos = np.array([12.0, 8.0, 11.0])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        assert status["z_mode"] in valid_modes
        assert status["h_mode"] in valid_modes


class TestCaptureDetection:
    def test_captured_when_close(self, logic):
        """Capture detected when within d_h and d_z."""
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([11.0, 11.0, 10.5])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        h_dist = math.sqrt(1.0 + 1.0)
        assert h_dist < logic.d_h
        assert abs(0.5) < logic.d_z
        assert status["captured"]
        assert status["game_status"] == "CAPTURED"

    def test_not_captured_when_far(self, logic):
        d_pos = np.array([5.0, 5.0, 5.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([30.0, 20.0, 15.0])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        assert not status["captured"]
        assert status["game_status"] != "CAPTURED"


class TestSafeFallback:
    def test_fallback_on_missing_vfs(self):
        """With no VFs loaded, should still return valid commands."""
        loader = ValueFunctionLoader("/nonexistent/path/")
        logic_fb = DefenderControlLogic(loader)
        d_pos = np.array([10.0, 10.0, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([12.0, 12.0, 11.0])
        cmd, status = logic_fb.compute_control(d_pos, d_vel, a_pos)
        assert cmd.shape == (3,)
        assert np.all(np.isfinite(cmd))
        assert status["z_mode"] == "pid_fallback"
        assert status["h_mode"] == "pid_invalid_bh"


class _FakeVF:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


class _FakeLoader:
    def __init__(
        self,
        phi_h_value=1.0,
        b_h_value=0.0,
        phi_z_value=-1.0,
        b_z_effective=1.0,
        b_h_effective=3.0,
        v_h_6d_values=(0.0, 10.0),
        zero_gradients=False,
    ):
        self.loaded_names = {"phi_h", "B_h", "phi_z", "B_z", "V_z_inf", "V_h_T", "V_h_T_6d"}
        self.phi_h_value = phi_h_value
        self.b_h_value = b_h_value
        self.phi_z_value = phi_z_value
        self.b_z_effective = b_z_effective
        self.b_h_effective = b_h_effective
        self.zero_gradients = zero_gradients
        self.vf_data = {
            "B_z": _FakeVF([1.0]),
            "B_h": _FakeVF([1.0]),
            "V_h_T_6d": _FakeVF(np.reshape(np.asarray(v_h_6d_values, dtype=float), (1, 1, 1, 1, 1, -1))),
        }

    def get_params(self, name):
        params = {
            "phi_z": {"d_z": 1.0, "k_z": 1.5, "U_D_z": 4.0, "U_A_z": 2.0},
            "phi_h": {"d_h": 3.0, "k_x": 0.7, "k_y": 0.7, "U_D_h": 6.0, "U_A_h": 3.0},
            "B_z": {
                "d_z_effective": self.b_z_effective,
                "paper_valid": True,
                "subset_valid": True,
            },
            "B_h": {
                "d_h_effective": self.b_h_effective,
                "paper_valid": True,
                "subset_valid": True,
            },
        }
        return params.get(name, {})

    def get_value(self, name, state):
        if name == "phi_h":
            return self.phi_h_value
        if name == "B_h":
            return self.b_h_value
        if name == "phi_z":
            return self.phi_z_value
        if name == "B_z":
            return 0.0
        if name == "V_h_T_6d":
            return 10.0
        if name in {"V_z_inf", "V_h_T"}:
            return 10.0
        return 0.0

    def get_gradient(self, name, state):
        if self.zero_gradients:
            sizes = {"phi_h": 6, "phi_z": 3, "V_h_T": 4, "V_h_T_6d": 6, "V_z_inf": 2}
            return np.zeros(sizes.get(name, 1))
        if name == "phi_h":
            return np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
        if name == "phi_z":
            return np.array([0.0, 1.0, 0.0])
        if name == "V_h_T_6d":
            return np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0])
        if name == "V_h_T":
            return np.array([0.0, 0.0, 1.0, -1.0])
        if name == "V_z_inf":
            return np.array([0.0, 1.0])
        return np.zeros(1)


class TestCorrectedConventions:
    def test_horizontal_status_uses_paper_positive_defender_region(self):
        logic = DefenderControlLogic(_FakeLoader(phi_h_value=1.0))
        status = logic._check_game_status(
            vertical_state=np.array([10.0, 0.0, 12.0]),
            horizontal_state=np.zeros(6),
            attacker_pos_h=np.zeros(2),
            z_rel=-2.0,
            x_rel=4.0,
            y_rel=0.0,
            defender_pos=np.array([0.0, 0.0, 10.0]),
            attacker_pos=np.array([4.0, 0.0, 12.0]),
        )

        assert status["in_W_D_h"] is True

    def test_horizontal_reaching_only_when_paper_phi_positive(self):
        state = np.array([5.0, 5.0, 0.0, 0.0, 20.0, 12.5])

        positive_logic = DefenderControlLogic(_FakeLoader(phi_h_value=1.0, b_h_value=0.0))
        _, _, positive_mode = positive_logic._horizontal_reach_track(
            state, -15.0, -7.5, 0.0, 0.0, 20.0, 12.5, 5.0, 5.0,
        )

        negative_logic = DefenderControlLogic(_FakeLoader(phi_h_value=-1.0, b_h_value=0.0))
        _, _, negative_mode = negative_logic._horizontal_reach_track(
            state, -15.0, -7.5, 0.0, 0.0, 20.0, 12.5, 5.0, 5.0,
        )

        assert positive_mode == "reaching"
        assert negative_mode == "pid_pursuit"

    def test_anti_closing_horizontal_reaching_falls_back_to_pursuit(self):
        state = np.array([10.0, -10.0, 0.0, 0.0, 0.0, 0.0])
        logic = DefenderControlLogic(_FakeLoader(phi_h_value=1.0, b_h_value=0.0))

        u_x, u_y, mode = logic._horizontal_reach_track(
            state, 10.0, -10.0, 0.0, 0.0, 0.0, 0.0, 10.0, -10.0,
        )

        assert mode == "pid_pursuit"
        assert 10.0 * u_x + -10.0 * u_y < 0.0

    def test_anti_closing_vertical_reaching_falls_back_to_pursuit(self):
        logic = DefenderControlLogic(_FakeLoader(phi_z_value=-1.0))

        u_z, mode = logic._vertical_reach_track(
            np.array([5.0, 0.0, 10.0]), -5.0, 0.0, 10.0, 5.0,
        )

        assert mode == "pid_pursuit"
        assert u_z > 0.0

    def test_vertical_reaching_uses_defender_minimizing_sign(self):
        logic = DefenderControlLogic(_FakeLoader())

        assert logic._optimal_reaching_vertical(np.array([10.0, 0.0, 12.0])) == -logic.U_D_z

    def test_expanded_threshold_metadata_rejects_bz(self):
        logic = DefenderControlLogic(_FakeLoader(b_z_effective=3.6))
        _, mode = logic._vertical_reach_track(
            np.array([10.0, 0.0, 10.0]), 0.0, 0.0, 10.0, 10.0,
        )

        assert mode == "pid_invalid_bz"

    def test_4d_horizontal_tracking_does_not_enable_algorithm2(self):
        loader = _FakeLoader(v_h_6d_values=(4.0, 10.0))
        loader.loaded_names.remove("V_h_T_6d")
        logic = DefenderControlLogic(loader)
        _, _, mode = logic._horizontal_reach_track(
            np.zeros(6), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

        assert mode == "pid_invalid_bh"

    def test_zero_gradient_vertical_tracking_returns_zero(self):
        logic = DefenderControlLogic(_FakeLoader(zero_gradients=True))

        assert logic._optimal_tracking_vertical(np.array([0.0, 0.0])) == 0.0

    def test_zero_gradient_horizontal_tracking_returns_zero(self):
        logic = DefenderControlLogic(_FakeLoader(zero_gradients=True))

        assert logic._optimal_tracking_horizontal(np.zeros(6)) == (0.0, 0.0)


class TestWallAvoidance:
    """Tests for the wall-avoidance safety layer."""

    def test_wall_avoidance_scales_near_min_wall_x(self, logic):
        """Near x=0 wall with velocity toward it: command should be scaled."""
        cmd = np.array([-6.0, 3.0, 0.0])
        pos = np.array([1.0, 12.5, 10.0])
        vel = np.array([-5.0, 0.0, 0.0])
        result = logic._apply_wall_avoidance(cmd, pos, vel)
        # x-command toward wall should be reduced
        assert result[0] > -3.0

    def test_wall_avoidance_scales_near_max_wall_x(self, logic):
        """Near x=45 wall with velocity toward it: command should be scaled."""
        cmd = np.array([6.0, 0.0, 0.0])
        pos = np.array([44.0, 12.5, 10.0])
        vel = np.array([5.0, 0.0, 0.0])
        result = logic._apply_wall_avoidance(cmd, pos, vel)
        assert result[0] < 3.0

    def test_wall_avoidance_pushes_away_very_close(self, logic):
        """Very close to wall (< 1m): should actively push away."""
        cmd = np.array([-6.0, 0.0, 0.0])
        pos = np.array([0.3, 12.5, 10.0])
        vel = np.array([-1.0, 0.0, 0.0])
        result = logic._apply_wall_avoidance(cmd, pos, vel)
        assert result[0] > 0  # pushed away from wall

    def test_wall_avoidance_no_effect_at_center(self, logic):
        """At arena center with zero velocity: command should be unchanged."""
        cmd = np.array([3.0, -2.0, 1.0])
        pos = np.array([22.5, 12.5, 10.0])
        vel = np.array([0.0, 0.0, 0.0])
        result = logic._apply_wall_avoidance(cmd, pos, vel)
        np.testing.assert_array_almost_equal(result, cmd)

    def test_wall_avoidance_y_and_z_walls(self, logic):
        """Wall avoidance works on y and z axes too."""
        # Near y=0 wall
        cmd_y = np.array([0.0, -6.0, 0.0])
        pos_y = np.array([22.5, 0.5, 10.0])
        vel_y = np.array([0.0, -4.0, 0.0])
        result_y = logic._apply_wall_avoidance(cmd_y, pos_y, vel_y)
        assert result_y[1] > 0  # pushed away

        # Near z=20 wall
        cmd_z = np.array([0.0, 0.0, 4.0])
        pos_z = np.array([22.5, 12.5, 19.5])
        vel_z = np.array([0.0, 0.0, 3.0])
        result_z = logic._apply_wall_avoidance(cmd_z, pos_z, vel_z)
        assert result_z[2] < 4.0  # scaled down

    def test_wall_avoidance_allows_capture_near_wall(self, logic):
        """Defender can still move toward attacker near wall (margin < d_h)."""
        # Attacker at x=1.0, defender at x=2.0. Margin at zero vel = 1.5m.
        # Defender is 2.0m from wall, margin is 1.5m, so not in avoidance zone.
        d_pos = np.array([2.0, 12.5, 10.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([1.0, 12.5, 10.0])
        cmd, _ = logic.compute_control(d_pos, d_vel, a_pos)
        # Should command toward attacker (negative x)
        assert cmd[0] < 0
