"""Tests for DefenderControlLogic -- standalone pytest, no rclpy required."""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, "/workspace/reach_avoid_game/src")

from reach_avoid_controller.value_function_loader import ValueFunctionLoader
from reach_avoid_controller.defender_node import DefenderControlLogic

VF_DIR = "/workspace/data/value_functions/"


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
        # z_rel = 0 -> inside B_z -> should be pid_deep or tracking
        assert status["z_mode"] in ("pid_deep", "tracking", "pid_fallback")

    def test_reaching_when_far(self, logic):
        """When defender is far from attacker, should be in reaching mode."""
        d_pos = np.array([5.0, 5.0, 5.0])
        d_vel = np.array([0.0, 0.0, 0.0])
        a_pos = np.array([30.0, 20.0, 15.0])
        _, status = logic.compute_control(d_pos, d_vel, a_pos)
        # Far apart -> outside B_z and B_h -> reaching
        assert status["z_mode"] in ("reaching", "pid_fallback")
        assert status["h_mode"] in ("reaching", "pid_fallback")

    def test_mode_is_valid_string(self, logic):
        valid_modes = {"reaching", "tracking", "pid_deep", "pid_fallback"}
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
        assert status["h_mode"] == "pid_fallback"
